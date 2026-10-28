import { describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentContext, McpServer } from "@agentclientprotocol/sdk";
import type { LettaCodeSocketConstructor } from "@letta-ai/letta-agent-sdk";
import {
  LettaAcpAgent,
  SessionRetirementError,
  toSdkMcpServers,
} from "../src/agent.js";
import { PERMISSION_MODE_CONFIG_ID } from "../src/session-modes.js";
import { SessionRegistry } from "../src/session-registry.js";

interface RuntimeScope {
  agent_id: string;
  conversation_id: string;
}

type WireMessage = Record<string, unknown>;
type SocketListener = (event: unknown) => void;

interface ServerSocket {
  readonly channel: "control" | "stream";
  push(message: WireMessage): void;
}

class FakeAppServer {
  readonly runtimeStartRequestIds: string[] = [];
  /** Full runtime_start payloads, to prove what a reopen asked the runtime for. */
  readonly runtimeStartMessages: WireMessage[] = [];
  /** closedSockets observed as each runtime_start arrived, for ordering checks. */
  readonly runtimeStartClosedSockets: number[] = [];
  private runtimeStartGate: Promise<void> | null = null;
  private updateModelGate: Promise<void> | null = null;
  readonly createdAgentBodies: WireMessage[] = [];
  readonly createdAgentPinGlobals: unknown[] = [];
  readonly approvalResponses: WireMessage[] = [];
  readonly conversationRetrieveIds: string[] = [];
  closedSockets = 0;
  readonly updatedModelPayloads: WireMessage[] = [];
  private nextConversation = 0;
  private activeControlSocket: ServerSocket | null = null;
  private activeRuntime: RuntimeScope | null = null;
  private activeCommand: string | null = null;
  private approvalWaiters: Array<(payload: WireMessage) => void> = [];
  private commandWaiters: Array<() => void> = [];
  private cancelablePromptWaiters: Array<() => void> = [];
  private cancelablePromptActive = false;

  socketConstructor(): LettaCodeSocketConstructor {
    const server = this;

    class FakeSocket {
      readyState = 0;
      readonly channel: "control" | "stream";
      private readonly listeners = new Map<string, Set<SocketListener>>();

      constructor(url: string) {
        this.channel = new URL(url).searchParams.get("channel") === "stream"
          ? "stream"
          : "control";
        queueMicrotask(() => {
          this.readyState = 1;
          this.emit("open", {});
        });
      }

      send(data: string): void {
        server.receive(this, JSON.parse(data) as WireMessage);
      }

      close(): void {
        if (this.readyState === 3) return;
        this.readyState = 3;
        server.closedSockets += 1;
        this.emit("close", {});
      }

      addEventListener(type: string, listener: SocketListener): void {
        const listeners = this.listeners.get(type) ?? new Set<SocketListener>();
        listeners.add(listener);
        this.listeners.set(type, listeners);
      }

      removeEventListener(type: string, listener: SocketListener): void {
        this.listeners.get(type)?.delete(listener);
      }

      push(message: WireMessage): void {
        queueMicrotask(() => {
          this.emit("message", { data: JSON.stringify(message) });
        });
      }

      private emit(type: string, event: unknown): void {
        for (const listener of this.listeners.get(type) ?? []) {
          listener(event);
        }
      }
    }

    return FakeSocket as unknown as LettaCodeSocketConstructor;
  }

  /**
   * Holds every later runtime_start until the returned release runs, so a test
   * can act while a session reopen is provably still in flight.
   */
  blockRuntimeStart(): () => void {
    let release = (): void => {};
    this.runtimeStartGate = new Promise<void>((resolve) => {
      release = resolve;
    });
    return release;
  }

  /**
   * Holds every later update_model response until the returned release runs,
   * so a test can act while a reopen is still reapplying the saved model.
   */
  blockUpdateModel(): () => void {
    let release = (): void => {};
    this.updateModelGate = new Promise<void>((resolve) => {
      release = resolve;
    });
    return release;
  }

  /**
   * Pushes an `update_device_status` for the active runtime, the channel a
   * runtime uses to report that it is working or waiting on an approval.
   */
  pushDeviceStatus(
    options: { isProcessing?: boolean; pendingToolName?: string } = {},
  ): void {
    const socket = this.activeControlSocket;
    const runtime = this.activeRuntime;
    if (!socket || !runtime) {
      throw new Error("No active runtime for a device status update");
    }
    socket.push({
      type: "update_device_status",
      runtime,
      device_status: {
        is_online: true,
        is_processing: options.isProcessing ?? false,
        current_permission_mode: "standard",
        current_working_directory: "/tmp/letta-acp-test",
        ...(options.pendingToolName
          ? {
              pending_control_requests: [
                {
                  request_id: "pending-approval-1",
                  request: { tool_name: options.pendingToolName },
                },
              ],
            }
          : {}),
      },
    });
  }

  /** Resolves with the next approval_response the adapter sends back. */
  nextApprovalResponse(): Promise<WireMessage> {
    return new Promise((resolve) => {
      this.approvalWaiters.push(resolve);
    });
  }

  /** Resolves when the adapter dispatches its next execute_command. */
  nextCommand(): Promise<void> {
    return new Promise((resolve) => {
      this.commandWaiters.push(resolve);
    });
  }

  /** Resolves after a prompt has produced activity but before it completes. */
  nextCancelablePrompt(): Promise<void> {
    return new Promise((resolve) => {
      this.cancelablePromptWaiters.push(resolve);
    });
  }

  requestPermissionAfterPrompt(
    overrides: {
      requestId?: string;
      toolName?: string;
      toolCallId?: string;
      input?: Record<string, unknown>;
    } = {},
  ): void {
    const socket = this.activeControlSocket;
    const runtime = this.activeRuntime;
    if (!socket || !runtime) {
      throw new Error("No active runtime for late permission request");
    }
    socket.push({
      type: "control_request",
      request_id: overrides.requestId ?? "approval-request-after-prompt",
      runtime,
      request: {
        subtype: "can_use_tool",
        tool_name: overrides.toolName ?? "Bash",
        tool_call_id: overrides.toolCallId ?? "call-after-prompt",
        input: overrides.input ?? {
          command: "pwd",
          description: "Show working directory after prompt",
        },
      },
    });
  }

  private receive(socket: ServerSocket, message: WireMessage): void {
    if (socket.channel !== "control") {
      throw new Error(`Unexpected command on ${socket.channel} channel`);
    }

    switch (message.type) {
      case "runtime_start": {
        const requestId = requiredString(message.request_id, "runtime_start.request_id");
        const conversationId = `conv-test-${++this.nextConversation}`;
        const createAgent = message.create_agent as
          | { body?: WireMessage; pin_global?: boolean }
          | undefined;
        const agentId =
          typeof message.agent_id === "string"
            ? message.agent_id
            : createAgent
              ? "agent-created"
              : "agent-test";
        const runtime = {
          agent_id: agentId,
          conversation_id: conversationId,
        };
        this.runtimeStartRequestIds.push(requestId);
        this.runtimeStartMessages.push(message);
        this.runtimeStartClosedSockets.push(this.closedSockets);
        if (createAgent?.body) {
          this.createdAgentBodies.push(createAgent.body);
          this.createdAgentPinGlobals.push(createAgent.pin_global);
        }
        const respond = (): void => {
          socket.push({
            type: "runtime_start_response",
            request_id: requestId,
            success: true,
            runtime,
            agent: { id: agentId, model: "test/model", tools: [] },
            conversation: { id: conversationId, agent_id: agentId },
          });
        };
        if (this.runtimeStartGate) {
          void this.runtimeStartGate.then(respond);
        } else {
          respond();
        }
        return;
      }
      case "enable_memfs":
        socket.push({
          type: "enable_memfs_response",
          request_id: requiredString(message.request_id, "enable_memfs.request_id"),
          success: true,
          memory_directory: "/tmp/letta-acp-memory",
        });
        return;
      case "conversation_retrieve": {
        const conversationId = requiredString(
          message.conversation_id,
          "conversation_retrieve.conversation_id",
        );
        this.conversationRetrieveIds.push(conversationId);
        if (conversationId.includes("missing")) {
          socket.push({
            type: "conversation_retrieve_response",
            request_id: requiredString(
              message.request_id,
              "conversation_retrieve.request_id",
            ),
            success: false,
            error: "conversation unavailable",
          });
          return;
        }
        socket.push({
          type: "conversation_retrieve_response",
          request_id: requiredString(
            message.request_id,
            "conversation_retrieve.request_id",
          ),
          success: true,
          conversation: {
            id: conversationId,
            agent_id: conversationId.includes("foreign") ? "agent-other" : "agent-test",
            archived: conversationId.includes("archived"),
          },
        });
        return;
      }
      case "conversation_messages_list":
        socket.push({
          type: "conversation_messages_list_response",
          request_id: requiredString(
            message.request_id,
            "conversation_messages_list.request_id",
          ),
          success: true,
          messages: [],
          has_more: false,
        });
        return;
      case "list_models":
        socket.push({
          type: "list_models_response",
          request_id: requiredString(message.request_id, "list_models.request_id"),
          success: true,
          entries: [
            {
              id: "test-model",
              handle: "test/model",
              label: "Test Model",
              description: "Deterministic test model",
              isDefault: true,
            },
            {
              id: "other-model",
              handle: "test/other-model",
              label: "Other Model",
              description: "Alternate deterministic model",
            },
          ],
          available_handles: ["test/model", "test/other-model"],
        });
        return;
      case "update_model": {
        const payload = message.payload as WireMessage;
        this.updatedModelPayloads.push(payload);
        const modelHandle =
          typeof payload.model_handle === "string" ? payload.model_handle : undefined;
        const modelId =
          typeof payload.model_id === "string"
            ? payload.model_id
            : modelHandle
              ? modelHandle.split("/").at(-1) ?? modelHandle
              : requiredString(payload.model_id, "update_model.model_id");
        const handle =
          modelHandle ??
          (modelId === "other-model" ? "test/other-model" : "test/model");
        const respond = (): void => {
          socket.push({
            type: "update_model_response",
            request_id: requiredString(message.request_id, "update_model.request_id"),
            success: true,
            runtime: message.runtime,
            applied_to: "conversation",
            model_id: modelId,
            model_handle: handle,
          });
        };
        if (this.updateModelGate) {
          void this.updateModelGate.then(respond);
        } else {
          respond();
        }
        return;
      }
      case "input":
        this.handleInput(socket, message);
        return;
      case "execute_command":
        this.handleCommand(socket, message);
        return;
      case "abort_message":
        socket.push({
          type: "abort_message_response",
          request_id: requiredString(
            message.request_id,
            "abort_message.request_id",
          ),
          success: true,
        });
        if (this.activeCommand || this.cancelablePromptActive) {
          socket.push({
            type: "update_loop_status",
            runtime: this.activeRuntime,
            loop_status: {
              status: "WAITING_ON_INPUT",
              active_run_ids: [],
            },
          });
          this.cancelablePromptActive = false;
        }
        return;
      default:
        throw new Error(`Unhandled fake app-server command: ${String(message.type)}`);
    }
  }

  private handleInput(socket: ServerSocket, message: WireMessage): void {
    const runtime = message.runtime as RuntimeScope;
    const payload = message.payload as Record<string, unknown>;
    this.activeControlSocket = socket;
    this.activeRuntime = runtime;

    if (payload.kind === "approval_response") {
      this.approvalResponses.push(payload);
      for (const waiter of this.approvalWaiters.splice(0)) waiter(payload);
      const command = this.activeCommand;
      this.pushDelta(socket, runtime, {
        message_type: "tool_return_message",
        run_id: command ? "run-command-approval" : "run-approval",
        tool_call_id: command ? "call-command-approval" : "call-approval",
        tool_return: "write completed",
        status: "success",
      });
      this.pushDelta(socket, runtime, {
        message_type: "assistant_message",
        run_id: command ? "run-command-approval" : "run-approval",
        content: command ? "COMMAND_APPROVAL_OK" : "APPROVAL_OK",
      });
      socket.push({
        type: "update_loop_status",
        runtime,
        loop_status: {
          status: "WAITING_ON_INPUT",
          active_run_ids: [],
        },
      });
      if (command) {
        this.pushDelta(socket, runtime, {
          message_type: "slash_command_end",
          command_id: command,
          success: true,
          output: "COMMAND_APPROVAL_OK",
        });
        this.activeCommand = null;
      }
      return;
    }

    if (payload.kind !== "create_message") {
      throw new Error(`Unhandled fake input payload: ${String(payload.kind)}`);
    }

    const promptText = JSON.stringify(payload.messages);
    if (promptText.includes("cancel active prompt")) {
      this.cancelablePromptActive = true;
      for (const waiter of this.cancelablePromptWaiters.splice(0)) waiter();
      this.pushDelta(socket, runtime, {
        message_type: "tool_call_message",
        run_id: "run-cancelled",
        tool_calls: [
          {
            id: "call-cancelled-agent",
            name: "Agent",
            arguments: JSON.stringify({
              description: "Inspect test coverage",
              prompt: "Review the repository tests",
            }),
          },
        ],
      });
      return;
    }
    if (promptText.includes("replacement after cancel")) {
      this.pushDelta(socket, runtime, {
        message_type: "assistant_message",
        run_id: "run-replacement",
        content: "REPLACEMENT_OK",
      });
      this.finishTurn(socket, runtime, "run-replacement", "end_turn");
      return;
    }
    if (promptText.includes("fragmented prompt")) {
      this.streamFragmentedToolCall(socket, runtime);
      return;
    }
    if (promptText.includes("busy idle prompt")) {
      this.reportIdleWhileRunning(socket, runtime, { thenContinue: true });
      return;
    }
    if (promptText.includes("silent idle prompt")) {
      this.reportIdleWhileRunning(socket, runtime, { thenContinue: false });
      return;
    }

    const isApprovalPrompt = promptText.includes("approval prompt");
    if (isApprovalPrompt) {
      this.pushDelta(socket, runtime, {
        message_type: "tool_call_message",
        run_id: "run-approval",
        tool_calls: [
          {
            id: "call-approval",
            name: "Write",
            arguments: JSON.stringify({ file_path: "/tmp/test.txt" }),
          },
        ],
      });
      socket.push({
        type: "control_request",
        request_id: "approval-request-1",
        runtime,
        request: {
          subtype: "can_use_tool",
          tool_name: "Write",
          tool_call_id: "call-approval",
          input: { file_path: "/tmp/test.txt" },
        },
      });
      this.finishTurn(socket, runtime, "run-approval", "requires_approval");
      return;
    }

    if (promptText.includes("edit prompt")) {
      this.pushDelta(socket, runtime, {
        message_type: "tool_call_message",
        run_id: "run-edit",
        tool_calls: [
          {
            id: "call-edit",
            name: "Edit",
            arguments: JSON.stringify({
              file_path: "/tmp/example.ts",
              old_string: "const value = 1;",
              new_string: "const value = 2;",
            }),
          },
        ],
      });
      this.pushDelta(socket, runtime, {
        message_type: "tool_return_message",
        run_id: "run-edit",
        tool_call_id: "call-edit",
        tool_return: JSON.stringify({
          message: "Successfully replaced 1 occurrence",
          replacements: 1,
          startLine: 12,
        }),
        status: "success",
      });
      this.finishTurn(socket, runtime, "run-edit", "end_turn");
      return;
    }

    this.pushDelta(socket, runtime, {
      message_type: "assistant_message",
      run_id: "run-prompt",
      content: "PROMPT_OK",
    });
    this.finishTurn(socket, runtime, "run-prompt", "end_turn");
  }

  private handleCommand(socket: ServerSocket, message: WireMessage): void {
    const runtime = message.runtime as RuntimeScope;
    const command = requiredString(
      message.command_id,
      "execute_command.command_id",
    );
    this.activeControlSocket = socket;
    this.activeRuntime = runtime;
    this.activeCommand = command;
    for (const waiter of this.commandWaiters.splice(0)) waiter();

    this.pushDelta(socket, runtime, {
      message_type: "slash_command_start",
      command_id: command,
    });
    if (command === "init") {
      // Stay silent until the test cancels, reproducing an abort before the
      // command's nested turn emits its first substantive event.
      return;
    }
    this.pushDelta(socket, runtime, {
      message_type: "tool_call_message",
      run_id: "run-command-approval",
      tool_calls: [
        {
          id: "call-command-approval",
          name: "Write",
          arguments: JSON.stringify({ file_path: "/tmp/command-test.txt" }),
        },
      ],
    });
    socket.push({
      type: "control_request",
      request_id: "command-approval-request",
      runtime,
      request: {
        subtype: "can_use_tool",
        tool_name: "Write",
        tool_call_id: "call-command-approval",
        input: { file_path: "/tmp/command-test.txt" },
      },
    });
    this.finishTurn(socket, runtime, "run-command-approval", "requires_approval");
  }

  /**
   * Streams one tool call as the app-server really does: an opening chunk
   * carrying the name plus a slice of the arguments, then continuation chunks
   * that carry argument text only (and therefore no tool name).
   */
  private streamFragmentedToolCall(
    socket: ServerSocket,
    runtime: RuntimeScope,
  ): void {
    const fragments = [
      { name: "Bash", arguments: '{"command":"git status"' },
      { arguments: ',"description":"Show the ' },
      { arguments: 'working tree status"}' },
    ];
    for (const fragment of fragments) {
      this.pushDelta(socket, runtime, {
        message_type: "tool_call_message",
        run_id: "run-fragmented",
        tool_calls: [{ id: "call-fragmented", ...fragment }],
      });
    }
    this.pushDelta(socket, runtime, {
      message_type: "tool_return_message",
      run_id: "run-fragmented",
      tool_call_id: "call-fragmented",
      tool_return: "nothing to commit",
      status: "success",
    });
    this.finishTurn(socket, runtime, "run-fragmented", "end_turn");
  }

  /**
   * Reports WAITING_ON_INPUT while still listing an active run — the shape
   * that makes the SDK synthesize a successful, stop-reason-less result even
   * though the agent is mid-step. `thenContinue` decides whether the run
   * really was still going (after a beat, as a real server would take) or
   * whether the status was simply stale and nothing more arrives.
   */
  private reportIdleWhileRunning(
    socket: ServerSocket,
    runtime: RuntimeScope,
    options: { thenContinue: boolean },
  ): void {
    this.pushDelta(socket, runtime, {
      message_type: "assistant_message",
      run_id: "run-first",
      content: "WORKING",
    });
    socket.push({
      type: "update_loop_status",
      runtime,
      loop_status: {
        status: "WAITING_ON_INPUT",
        active_run_ids: ["run-second"],
      },
    });
    if (!options.thenContinue) return;
    // A real server takes a beat before the next step; without the delay the
    // continuation lands in the same microtask batch and hides the race.
    setTimeout(() => {
      this.pushDelta(socket, runtime, {
        message_type: "assistant_message",
        run_id: "run-second",
        content: "STILL_WORKING",
      });
      // The turn tracker was already consumed by the contradicted result, so
      // the run ends the way post-approval rounds do: on a real idle status.
      socket.push({
        type: "update_loop_status",
        runtime,
        loop_status: { status: "WAITING_ON_INPUT", active_run_ids: [] },
      });
    }, 20);
  }

  private pushDelta(
    socket: ServerSocket,
    runtime: RuntimeScope,
    delta: WireMessage,
  ): void {
    socket.push({ type: "stream_delta", runtime, delta });
  }

  /**
   * Agent SDK 0.8 parks a pending terminal on `stop_reason` and completes the
   * turn only after `turn_finished`. Letta Code has emitted that event since
   * 0.30.28; a fake that stops at `stop_reason` leaves the turn open until
   * `requestTimeoutMs`.
   */
  private finishTurn(
    socket: ServerSocket,
    runtime: RuntimeScope,
    runId: string,
    stopReason: string,
  ): void {
    this.pushDelta(socket, runtime, {
      message_type: "stop_reason",
      run_id: runId,
      stop_reason: stopReason,
    });
    socket.push({
      type: "turn_finished",
      runtime,
      run_id: runId,
      stop_reason: stopReason,
    });
  }
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`Missing ${label}`);
  }
  return value;
}

function createAgent(
  server: FakeAppServer,
  overrides: {
    agentId?: string;
    model?: string;
    sessionRegistryDir?: string | null;
    sessionRegistryScope?: string;
    outOfTurnPermissionTimeoutMs?: number;
    prematureResultGraceMs?: number;
    idleEvictMs?: number;
  } = {},
): LettaAcpAgent {
  return new LettaAcpAgent({
    clientOptions: {
      backend: "local",
      appServer: {
        url: "ws://fake-app-server.test/ws",
        WebSocket: server.socketConstructor(),
        requestTimeoutMs: 1_000,
        pinGlobalAgent: false,
      },
    },
    agentId: "agent-test",
    permissionMode: "standard",
    ...overrides,
  });
}

/** The Letta SDK surface the adapter uses to open and release a session. */
interface SdkSessionHandle {
  [Symbol.asyncDispose]?: () => Promise<void>;
  close(): void;
}

interface SdkSessionOptions {
  mcpServers?: Record<string, unknown>;
}

interface SdkClientHandle {
  createSession(agentId: string, options: SdkSessionOptions): SdkSessionHandle;
  resumeSession(
    conversationId: string,
    options: SdkSessionOptions,
  ): SdkSessionHandle;
}

/** A disposal held open, so a test can act while a session is still closing. */
interface DeferredDisposal {
  /** True once the adapter has asked the SDK to dispose a session. */
  readonly started: boolean;
  /** Lets the real disposal proceed. */
  resolve(): void;
  /** Fails the disposal, leaving that runtime unreleased. */
  reject(error: unknown): void;
}

/**
 * Instruments the agent's Letta SDK client — the adapter's only boundary with
 * the SDK — so a test can see what each session open asks for and control when
 * a disposal settles. Sessions are still real SDK sessions driven over the fake
 * app-server socket.
 */
function instrumentSdkSessions(
  agent: LettaAcpAgent,
  options: { stripMcpServers?: boolean } = {},
) {
  const client = (agent as unknown as { client: SdkClientHandle }).client;
  const createSession = client.createSession.bind(client);
  const resumeSession = client.resumeSession.bind(client);
  /** The mcpServers the adapter handed the SDK, one entry per session opened. */
  const mcpServerOptions: Array<Record<string, unknown> | undefined> = [];
  let deferred: { started: boolean; gate: Promise<void> } | null = null;
  let closeFailure: Error | null = null;

  const instrument = (
    sessionOptions: SdkSessionOptions,
    open: (forwarded: SdkSessionOptions) => SdkSessionHandle,
  ): SdkSessionHandle => {
    mcpServerOptions.push(sessionOptions.mcpServers);
    const forwarded = { ...sessionOptions };
    // Letting the SDK connect these for real would spawn a subprocess or reach
    // a network target, which these tests must never do. Record what the
    // adapter asked for and open the session without it.
    if (options.stripMcpServers) delete forwarded.mcpServers;
    const session = open(forwarded);
    const dispose = session[Symbol.asyncDispose]?.bind(session);
    Object.defineProperty(session, Symbol.asyncDispose, {
      configurable: true,
      writable: true,
      value: async (): Promise<void> => {
        const held = deferred;
        if (held) {
          deferred = null;
          held.started = true;
          await held.gate;
        }
        await dispose?.();
      },
    });
    const close = session.close.bind(session);
    Object.defineProperty(session, "close", {
      configurable: true,
      writable: true,
      value: (): void => {
        if (closeFailure) throw closeFailure;
        close();
      },
    });
    return session;
  };

  client.createSession = (agentId, sessionOptions) =>
    instrument(sessionOptions, (forwarded) => createSession(agentId, forwarded));
  client.resumeSession = (conversationId, sessionOptions) =>
    instrument(sessionOptions, (forwarded) =>
      resumeSession(conversationId, forwarded),
    );

  return {
    mcpServerOptions,
    /** Holds the next disposal the adapter starts until the test settles it. */
    deferNextDisposal(): DeferredDisposal {
      let resolveGate = (): void => {};
      let rejectGate = (_error: unknown): void => {};
      const gate = new Promise<void>((resolve, reject) => {
        resolveGate = resolve;
        rejectGate = reject;
      });
      const held = { started: false, gate };
      deferred = held;
      return {
        get started(): boolean {
          return held.started;
        },
        resolve: resolveGate,
        reject: rejectGate,
      };
    },
    /**
     * Makes every subsequent `close()` on these sessions throw, so a test can
     * cover the case where the adapter has no working way left to release a
     * runtime it is holding.
     */
    failCloseWith(error: Error): void {
      closeFailure = error;
    },
  };
}

function createContext(
  options: {
    answerPermissions?: boolean;
    permissionDelayMs?: number;
    permissionOptionId?: "allow_once" | "allow_always" | "reject_once";
  } = {},
) {
  const answerPermissions = options.answerPermissions ?? true;
  const permissionDelayMs = options.permissionDelayMs ?? 0;
  const permissionOptionId = options.permissionOptionId ?? "allow_once";
  const updates: WireMessage[] = [];
  const permissionRequests: WireMessage[] = [];
  const permissionSignals: Array<AbortSignal | undefined> = [];
  const context = {
    notify: async (_method: unknown, params: { update: WireMessage }) => {
      updates.push(params.update);
    },
    request: async (
      _method: unknown,
      params: WireMessage,
      requestOptions?: { cancellationSignal?: AbortSignal },
    ) => {
      permissionRequests.push(params);
      permissionSignals.push(requestOptions?.cancellationSignal);
      if (!answerPermissions) return new Promise(() => {});
      if (permissionDelayMs > 0) {
        await Bun.sleep(permissionDelayMs);
      }
      return {
        outcome: { outcome: "selected" as const, optionId: permissionOptionId },
      };
    },
  } as unknown as AgentContext;
  return { context, updates, permissionRequests, permissionSignals };
}

async function openSession(
  agent: LettaAcpAgent,
  context: AgentContext,
  cwd = "/tmp/letta-acp-test",
) {
  await agent.initialize({ protocolVersion: 1, clientCapabilities: {} });
  return agent.newSession({ cwd, mcpServers: [] }, context);
}

describe("Agent SDK app-server integration", () => {
  test("advertises and maps Agent SDK MCP transports", async () => {
    const servers: McpServer[] = [
      {
        name: "local",
        command: "/usr/bin/node",
        args: ["server.js"],
        env: [{ name: "TOKEN", value: "local-token" }],
      },
      {
        type: "http",
        name: "remote",
        url: "https://example.com/mcp",
        headers: [{ name: "Authorization", value: "Bearer token" }],
      },
      {
        type: "sse",
        name: "legacy",
        url: "https://example.com/sse",
        headers: [],
      },
      {
        type: "acp",
        name: "in-band",
        serverId: "server-1",
      },
    ];

    expect(toSdkMcpServers(servers)).toEqual({
      local: {
        command: "/usr/bin/node",
        args: ["server.js"],
        env: { TOKEN: "local-token" },
      },
      remote: {
        type: "http",
        url: "https://example.com/mcp",
        headers: { Authorization: "Bearer token" },
      },
      legacy: {
        type: "sse",
        url: "https://example.com/sse",
        headers: {},
      },
    });

    const agent = createAgent(new FakeAppServer());
    try {
      const response = await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
      });
      expect(response.agentCapabilities?.mcpCapabilities).toEqual({
        http: true,
        sse: true,
      });
    } finally {
      agent.shutdown();
    }
  });

  test("creates a named, unpinned ACP agent with the configured model", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, { agentId: undefined });
    const { context } = createContext();

    try {
      const session = await openSession(agent, context);

      expect(session.sessionId).toBe("conv-test-2");
      expect(server.createdAgentBodies).toHaveLength(1);
      expect(server.createdAgentBodies[0]).toMatchObject({
        name: "Letta",
        description: "Letta agent driven through the Agent Client Protocol",
        model: "letta/auto",
        memory_blocks: expect.arrayContaining([
          expect.objectContaining({ label: "persona" }),
          expect.objectContaining({ label: "human" }),
        ]),
      });
      expect(server.createdAgentBodies[0]).not.toHaveProperty("hidden", true);
      expect(server.createdAgentPinGlobals).toEqual([false]);
    } finally {
      agent.shutdown();
    }
  });

  test("uses an explicit model when creating an agent", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      agentId: undefined,
      model: "anthropic/claude-sonnet-4",
    });
    const { context } = createContext();

    try {
      await openSession(agent, context);
      expect(server.createdAgentBodies[0]).toMatchObject({
        model: "anthropic/claude-sonnet-4",
      });
      expect(server.createdAgentPinGlobals).toEqual([false]);
    } finally {
      agent.shutdown();
    }
  });

  test("reuses an explicit agent without creating or pinning another", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, { agentId: "agent-existing" });
    const { context } = createContext();

    try {
      await openSession(agent, context);
      expect(server.createdAgentBodies).toEqual([]);
      expect(server.createdAgentPinGlobals).toEqual([]);
      expect(server.updatedModelPayloads).toEqual([]);
    } finally {
      agent.shutdown();
    }
  });

  test("closes a displaced session when the same conversation is loaded again", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context } = createContext();
    const params = {
      sessionId: "conv-existing",
      cwd: "/tmp/letta-acp-test",
      mcpServers: [],
    };

    try {
      await agent.initialize({ protocolVersion: 1, clientCapabilities: {} });
      await agent.loadSession(params, context);
      const closedBeforeReload = server.closedSockets;

      await agent.loadSession(params, context);

      expect(server.closedSockets).toBeGreaterThan(closedBeforeReload);
    } finally {
      agent.shutdown();
    }
  });

  test("bounds session discovery lookups to one registry page", async () => {
    const directory = await mkdtemp(join(tmpdir(), "letta-acp-list-page-"));
    const scope = "pagination-test";
    const registry = new SessionRegistry(directory, scope);
    const cwd = "/tmp/letta-acp-test";
    for (let index = 0; index < 51; index += 1) {
      await registry.record("agent-test", `conv-page-${String(index).padStart(2, "0")}`, cwd);
    }

    const server = new FakeAppServer();
    const agent = createAgent(server, {
      sessionRegistryDir: directory,
      sessionRegistryScope: scope,
    });
    try {
      const first = await agent.listSessions({ cwd });
      expect(first.sessions).toHaveLength(50);
      expect(first.nextCursor).toBeString();
      expect(server.conversationRetrieveIds).toHaveLength(50);

      const second = await agent.listSessions({ cwd, cursor: first.nextCursor });
      expect(second.sessions).toHaveLength(1);
      expect(second.nextCursor).toBeUndefined();
      expect(server.conversationRetrieveIds).toHaveLength(51);
    } finally {
      agent.shutdown();
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("removes archived and foreign sessions but retains unavailable records", async () => {
    const directory = await mkdtemp(join(tmpdir(), "letta-acp-list-stale-"));
    const scope = "stale-test";
    const registry = new SessionRegistry(directory, scope);
    const cwd = "/tmp/letta-acp-test";
    await registry.record("agent-test", "conv-visible", cwd);
    await registry.record("agent-test", "conv-archived", cwd);
    await registry.record("agent-test", "conv-foreign", cwd);
    await registry.record("agent-test", "conv-missing", cwd);

    const server = new FakeAppServer();
    const agent = createAgent(server, {
      sessionRegistryDir: directory,
      sessionRegistryScope: scope,
    });
    try {
      const result = await agent.listSessions({ cwd });
      expect(result.sessions.map((session) => session.sessionId)).toEqual([
        "conv-visible",
      ]);
      expect(
        (await registry.list("agent-test", cwd)).map((record) => record.sessionId).sort(),
      ).toEqual(["conv-missing", "conv-visible"]);
    } finally {
      agent.shutdown();
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("starts repeated sessions with distinct runtime request ids", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context } = createContext();

    try {
      const sessionIds: string[] = [];
      for (let index = 0; index < 3; index += 1) {
        sessionIds.push((await openSession(agent, context)).sessionId);
      }

      expect(sessionIds).toEqual([
        "conv-test-1",
        "conv-test-2",
        "conv-test-3",
      ]);
      expect(new Set(server.runtimeStartRequestIds).size).toBe(3);
    } finally {
      agent.shutdown();
    }
  });

  test("advertises available models in the session configuration", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context } = createContext();

    try {
      const session = await openSession(agent, context);

      expect(session.configOptions).toEqual([
        {
          id: "permissions",
          name: "Permissions",
          description: "Approval behavior for tool calls",
          category: "mode",
          type: "select",
          currentValue: "standard",
          options: [
            expect.objectContaining({ value: "standard", name: "Ask before edits" }),
            expect.objectContaining({ value: "acceptEdits", name: "Accept edits" }),
            expect.objectContaining({
              value: "unrestricted",
              name: "Bypass permissions",
            }),
          ],
        },
        {
          id: "model",
          name: "Model",
          description: "Model used for this Letta session",
          category: "model",
          type: "select",
          currentValue: "test-model",
          options: [
            {
              value: "test-model",
              name: "Test Model",
              description: "Deterministic test model",
            },
            {
              value: "other-model",
              name: "Other Model",
              description: "Alternate deterministic model",
            },
          ],
        },
      ]);
    } finally {
      agent.shutdown();
    }
  });

  test("switches models through session configuration options", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: "model",
        value: "other-model",
      });

      expect(server.updatedModelPayloads).toEqual([{ model_id: "other-model" }]);
      expect(
        result.configOptions.find((option) => option.id === "model"),
      ).toMatchObject({
        id: "model",
        category: "model",
        currentValue: "other-model",
      });
    } finally {
      agent.shutdown();
    }
  });

  test("switches permission enforcement through session config options", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, permissionRequests } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: "permissions",
        value: "unrestricted",
      });
      expect(
        result.configOptions.find((option) => option.id === "permissions"),
      ).toMatchObject({
        category: "mode",
        currentValue: "unrestricted",
      });

      await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );
      const approval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt();

      expect(await approval).toMatchObject({
        decision: { behavior: "allow" },
      });
      expect(permissionRequests).toEqual([]);
    } finally {
      agent.shutdown();
    }
  });

  test("streams an assistant response through a complete prompt turn", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );

      expect(result).toEqual({ stopReason: "end_turn" });
      expect(updates).toContainEqual({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "PROMPT_OK" },
      });
    } finally {
      agent.shutdown();
    }
  });

  test("streams fragmented tool arguments as one tool card", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "fragmented prompt" }],
        },
        context,
      );
      expect(result).toEqual({ stopReason: "end_turn" });

      expect(
        updates.filter((update) => update.toolCallId === "call-fragmented"),
      ).toEqual([
        {
          sessionUpdate: "tool_call",
          toolCallId: "call-fragmented",
          title: "Bash",
          kind: "execute",
          status: "in_progress",
          rawInput: { raw: '{"command":"git status"' },
          locations: [],
        },
        {
          sessionUpdate: "tool_call_update",
          toolCallId: "call-fragmented",
          title: "Show the working tree status",
          kind: "execute",
          status: "in_progress",
          rawInput: {
            command: "git status",
            description: "Show the working tree status",
          },
          locations: [],
        },
        {
          sessionUpdate: "tool_call_update",
          toolCallId: "call-fragmented",
          status: "completed",
          rawOutput: "nothing to commit",
          content: [
            {
              type: "content",
              content: { type: "text", text: "nothing to commit" },
            },
          ],
        },
      ]);
    } finally {
      agent.shutdown();
    }
  });

  test("renders Bash output as a native terminal when the client supports it", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: { _meta: { terminal_output: true } },
      });
      const session = await agent.newSession(
        { cwd: "/tmp/letta-acp-test", mcpServers: [] },
        context,
      );
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "fragmented prompt" }],
        },
        context,
      );

      expect(result).toEqual({ stopReason: "end_turn" });
      expect(
        updates.filter((update) => update.toolCallId === "call-fragmented"),
      ).toEqual([
        {
          sessionUpdate: "tool_call",
          toolCallId: "call-fragmented",
          title: "Bash",
          kind: "execute",
          status: "in_progress",
          rawInput: { raw: '{"command":"git status"' },
          locations: [],
          content: [{ type: "terminal", terminalId: "call-fragmented" }],
          _meta: {
            terminal_info: {
              terminal_id: "call-fragmented",
              cwd: "/tmp/letta-acp-test",
            },
          },
        },
        {
          sessionUpdate: "tool_call_update",
          toolCallId: "call-fragmented",
          title: "Show the working tree status",
          kind: "execute",
          status: "in_progress",
          rawInput: {
            command: "git status",
            description: "Show the working tree status",
          },
          locations: [],
        },
        {
          sessionUpdate: "tool_call_update",
          toolCallId: "call-fragmented",
          _meta: {
            terminal_output: {
              terminal_id: "call-fragmented",
              data: "nothing to commit",
            },
          },
        },
        {
          sessionUpdate: "tool_call_update",
          toolCallId: "call-fragmented",
          status: "completed",
          rawOutput: "nothing to commit",
          content: [
            { type: "terminal", terminalId: "call-fragmented" },
            {
              type: "content",
              content: { type: "text", text: "nothing to commit" },
            },
          ],
          _meta: {
            terminal_exit: {
              terminal_id: "call-fragmented",
              exit_code: 0,
              signal: null,
            },
          },
        },
      ]);
    } finally {
      agent.shutdown();
    }
  });

  test("renders an SDK Edit call as a native ACP diff", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "edit prompt" }],
        },
        context,
      );

      expect(result).toEqual({ stopReason: "end_turn" });
      expect(updates).toContainEqual(
        expect.objectContaining({
          sessionUpdate: "tool_call",
          toolCallId: "call-edit",
          content: [
            {
              type: "diff",
              path: "/tmp/example.ts",
              oldText: "const value = 1;",
              newText: "const value = 2;",
            },
          ],
        }),
      );
      const completed = updates.find(
        (update) =>
          update.sessionUpdate === "tool_call_update" &&
          update.toolCallId === "call-edit" &&
          update.status === "completed",
      );
      expect(completed).toEqual(
        expect.objectContaining({
          rawOutput: {
            message: "Successfully replaced 1 occurrence",
            replacements: 1,
            startLine: 12,
          },
          locations: [{ path: "/tmp/example.ts", line: 12 }],
        }),
      );
      expect(completed).not.toHaveProperty("content");
    } finally {
      agent.shutdown();
    }
  });

  test("routes a permission request after the prompt handler returns", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, permissionRequests, permissionSignals } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );
      expect(result).toEqual({ stopReason: "end_turn" });

      const approval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt();

      expect(await approval).toEqual(
        expect.objectContaining({
          kind: "approval_response",
          request_id: "approval-request-after-prompt",
          decision: expect.objectContaining({ behavior: "allow" }),
        }),
      );
      expect(permissionRequests).toContainEqual(
        expect.objectContaining({
          sessionId: session.sessionId,
          toolCall: expect.objectContaining({
            title: "Show working directory after prompt",
          }),
        }),
      );
      // Out-of-turn requests carry a deadline the client can observe.
      expect(permissionSignals.at(-1)).toBeInstanceOf(AbortSignal);
    } finally {
      agent.shutdown();
    }
  });

  test("auto-allows task bookkeeping in ask-before-edits mode", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, permissionRequests } = createContext();

    try {
      const session = await openSession(agent, context);
      await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );
      const approval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt({
        requestId: "task-update-request",
        toolName: "TaskUpdate",
        toolCallId: "call-task-update",
        input: { taskId: "task_1", status: "completed" },
      });

      expect(await approval).toEqual(
        expect.objectContaining({
          request_id: "task-update-request",
          decision: expect.objectContaining({ behavior: "allow" }),
        }),
      );
      expect(permissionRequests).toEqual([]);
    } finally {
      agent.shutdown();
    }
  });

  test("aligns editor-tool approvals with session modes and cwd", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, permissionRequests } = createContext();
    const root = await mkdtemp(join(tmpdir(), "letta-acp-editor-permissions-"));
    const workspace = join(root, "repo");
    const inside = join(workspace, "src", "inside.ts");
    const outside = join(root, "outside.ts");
    await mkdir(join(workspace, "src"), { recursive: true });
    await writeFile(inside, "inside\n");
    await writeFile(outside, "outside\n");

    try {
      const session = await openSession(agent, context, workspace);
      await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );

      const insideApproval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt({
        requestId: "read-inside",
        toolName: "read_editor_buffer",
        toolCallId: "call-read-inside",
        input: { path: inside },
      });
      expect(await insideApproval).toMatchObject({
        request_id: "read-inside",
        decision: { behavior: "allow" },
      });
      expect(permissionRequests).toEqual([]);

      const outsideApproval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt({
        requestId: "read-outside",
        toolName: "read_editor_buffer",
        toolCallId: "call-read-outside",
        input: { path: outside },
      });
      await outsideApproval;
      expect(permissionRequests.at(-1)).toMatchObject({
        toolCall: {
          toolCallId: "call-read-outside",
          kind: "read",
          rawInput: { path: outside },
        },
      });

      const standardWrite = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt({
        requestId: "write-standard",
        toolName: "write_via_editor",
        toolCallId: "call-write-standard",
        input: { path: inside, content: "changed\n" },
      });
      await standardWrite;
      expect(permissionRequests.at(-1)).toMatchObject({
        toolCall: {
          toolCallId: "call-write-standard",
          kind: "edit",
        },
      });

      await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: "permissions",
        value: "acceptEdits",
      });
      const requestCount = permissionRequests.length;
      const acceptedWrite = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt({
        requestId: "write-accepted",
        toolName: "write_via_editor",
        toolCallId: "call-write-accepted",
        input: { path: inside, content: "accepted\n" },
      });
      expect(await acceptedWrite).toMatchObject({
        request_id: "write-accepted",
        decision: { behavior: "allow" },
      });
      expect(permissionRequests).toHaveLength(requestCount);
    } finally {
      agent.shutdown();
      await rm(root, { recursive: true, force: true });
    }
  });

  test("scopes always-allow editor decisions to one ACP session", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, permissionRequests } = createContext({
      permissionOptionId: "allow_always",
    });
    const root = await mkdtemp(join(tmpdir(), "letta-acp-editor-always-"));
    const firstWorkspace = join(root, "first");
    const secondWorkspace = join(root, "second");
    const outside = join(root, "outside.ts");
    await mkdir(firstWorkspace, { recursive: true });
    await mkdir(secondWorkspace, { recursive: true });
    await writeFile(outside, "outside\n");

    try {
      const first = await openSession(agent, context, firstWorkspace);
      await agent.prompt(
        {
          sessionId: first.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );
      for (const requestId of ["first-read", "first-read-again"]) {
        const approval = server.nextApprovalResponse();
        server.requestPermissionAfterPrompt({
          requestId,
          toolName: "read_editor_buffer",
          toolCallId: `call-${requestId}`,
          input: { path: outside },
        });
        await approval;
      }
      expect(permissionRequests).toHaveLength(1);

      const second = await openSession(agent, context, secondWorkspace);
      await agent.prompt(
        {
          sessionId: second.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );
      const secondApproval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt({
        requestId: "second-read",
        toolName: "read_editor_buffer",
        toolCallId: "call-second-read",
        input: { path: outside },
      });
      await secondApproval;
      expect(permissionRequests).toHaveLength(2);
    } finally {
      agent.shutdown();
      await rm(root, { recursive: true, force: true });
    }
  });

  test("attaches a permission request to the tool call it belongs to", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, permissionRequests } = createContext();

    try {
      const session = await openSession(agent, context);
      // Leaves a streamed Bash call as the most recent one, so name matching
      // cannot find the Write call the approval is actually for.
      await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "fragmented prompt" }],
        },
        context,
      );

      const approval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt({
        requestId: "approval-unrelated-tool",
        toolName: "Write",
        toolCallId: "call-write",
        input: { file_path: "/tmp/example.ts" },
      });
      await approval;

      // A synthetic id here would render a permission card the client can
      // never reconcile with the tool call in the stream.
      expect(permissionRequests.at(-1)).toEqual(
        expect.objectContaining({
          toolCall: expect.objectContaining({
            toolCallId: "call-write",
            title: "Write: /tmp/example.ts",
            kind: "edit",
          }),
        }),
      );
    } finally {
      agent.shutdown();
    }
  });

  test("denies a permission request that lands after cancellation", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, permissionRequests } = createContext();

    try {
      const session = await openSession(agent, context);
      await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );
      await agent.cancel({ sessionId: session.sessionId });

      const approval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt();

      expect(await approval).toEqual(
        expect.objectContaining({
          request_id: "approval-request-after-prompt",
          decision: expect.objectContaining({
            behavior: "deny",
            message: "Prompt turn was cancelled",
          }),
        }),
      );
      // The user already pressed stop; they should not be asked again.
      expect(permissionRequests).toEqual([]);
    } finally {
      agent.shutdown();
    }
  });

  test("denies an out-of-turn permission request the client never answers", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, { outOfTurnPermissionTimeoutMs: 10 });
    const { context, permissionSignals } = createContext({
      answerPermissions: false,
    });

    try {
      const session = await openSession(agent, context);
      await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );

      const approval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt();

      expect(await approval).toEqual(
        expect.objectContaining({
          request_id: "approval-request-after-prompt",
          decision: expect.objectContaining({
            behavior: "deny",
            message:
              "Permission request timed out after 10ms while trying to " +
              "approve Bash outside a prompt turn",
          }),
        }),
      );
      expect(permissionSignals.at(-1)?.aborted).toBe(true);
    } finally {
      agent.shutdown();
    }
  });

  test("ends a contradicted turn when the SDK drops the still-running continuation", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "busy idle prompt" }],
        },
        context,
      );

      expect(result).toEqual({ stopReason: "end_turn" });
      // SDK 0.8 settles every run id listed on the contradicted idle status.
      // The continuation for that run is dropped before this adapter can see
      // it, so the prompt must end instead of waiting for a second idle.
      expect(updates).not.toContainEqual({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "STILL_WORKING" },
      });
    } finally {
      agent.shutdown();
    }
  });

  test("ends the turn when a contradicted completion is never followed up", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, { prematureResultGraceMs: 20 });
    const { context } = createContext();

    try {
      const session = await openSession(agent, context);
      // The status was simply stale: nothing follows it. The grace period has
      // to expire on its own, or the prompt would hang forever.
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "silent idle prompt" }],
        },
        context,
      );

      expect(result).toEqual({ stopReason: "end_turn" });
    } finally {
      agent.shutdown();
    }
  });

  test("recovers a prompt after an approved mutating tool call", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates, permissionRequests } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "approval prompt" }],
        },
        context,
      );

      expect(result).toEqual({ stopReason: "end_turn" });
      expect(permissionRequests).toHaveLength(1);
      expect(permissionRequests[0]).toMatchObject({
        sessionId: session.sessionId,
        toolCall: { title: "Write: /tmp/test.txt" },
      });
      expect(server.approvalResponses).toHaveLength(1);
      expect(server.approvalResponses[0]).toMatchObject({
        kind: "approval_response",
        request_id: "approval-request-1",
        decision: { behavior: "allow" },
      });
      expect(updates).toContainEqual(
        expect.objectContaining({
          sessionUpdate: "tool_call_update",
          toolCallId: "call-approval",
          status: "completed",
          content: [
            {
              type: "content",
              content: { type: "text", text: "write completed" },
            },
          ],
        }),
      );
      expect(updates).toContainEqual({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "APPROVAL_OK" },
      });
    } finally {
      agent.shutdown();
    }
  });

  test("recovers an execute command after an approved tool call", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates, permissionRequests } = createContext({
      permissionDelayMs: 10,
    });

    try {
      const session = await openSession(agent, context);
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "/remember command fact" }],
        },
        context,
      );

      expect(result).toEqual({ stopReason: "end_turn" });
      expect(permissionRequests).toHaveLength(1);
      expect(permissionRequests[0]).toMatchObject({
        sessionId: session.sessionId,
        toolCall: {
          toolCallId: "call-command-approval",
          title: "Write: /tmp/command-test.txt",
        },
      });
      expect(updates).toContainEqual({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "COMMAND_APPROVAL_OK" },
      });
    } finally {
      agent.shutdown();
    }
  });

  test("drains a cancelled turn before starting its replacement", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context);
      const started = server.nextCancelablePrompt();
      const first = agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "cancel active prompt" }],
        },
        context,
      );
      await started;
      await agent.cancel({ sessionId: session.sessionId });
      expect(await first).toEqual({ stopReason: "cancelled" });

      const replacementUpdateStart = updates.length;
      const replacement = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "replacement after cancel" }],
        },
        context,
      );

      expect(replacement).toEqual({ stopReason: "end_turn" });
      expect(updates.slice(replacementUpdateStart)).toContainEqual({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "REPLACEMENT_OK" },
      });
    } finally {
      agent.shutdown();
    }
  });

  test("cancels an execute command before its first turn event", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context } = createContext();

    try {
      const session = await openSession(agent, context);
      const commandStarted = server.nextCommand();
      const prompt = agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "/init" }],
        },
        context,
      );
      await commandStarted;
      await agent.cancel({ sessionId: session.sessionId });

      expect(await prompt).toEqual({ stopReason: "cancelled" });
    } finally {
      agent.shutdown();
    }
  });
});

/**
 * Idle sessions each pin a Letta runtime (an app-server child process and its
 * per-session MCP subprocesses), so a long-lived adapter accumulates them
 * without bound. These cover closing an idle session, reopening the same
 * conversation on demand, and every case where a session must be kept.
 */
describe("idle session eviction", () => {
  const IDLE_MS = 40;

  /** Polls until the condition holds, so tests never hard-code sweep timing. */
  async function waitFor(
    label: string,
    condition: () => boolean,
    timeoutMs = 2_000,
  ): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (!condition()) {
      if (Date.now() > deadline) throw new Error(`Timed out waiting for ${label}`);
      await Bun.sleep(5);
    }
  }

  /** Loads a conversation by id, which is the id eviction later reopens. */
  async function loadConversation(
    agent: LettaAcpAgent,
    context: AgentContext,
    sessionId: string,
    mcpServers: McpServer[] = [],
  ) {
    await agent.initialize({ protocolVersion: 1, clientCapabilities: {} });
    return agent.loadSession(
      { sessionId, cwd: "/tmp/letta-acp-test", mcpServers },
      context,
    );
  }

  function conversationIdsStarted(server: FakeAppServer): unknown[] {
    return server.runtimeStartMessages.map((message) => message.conversation_id);
  }

  test("closes a session that has been idle past the window", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const { context } = createContext();
    try {
      await loadConversation(agent, context, "conv-idle");
      expect(server.runtimeStartMessages).toHaveLength(1);
      const closedBefore = server.closedSockets;

      await waitFor(
        "the idle session to be closed",
        () => server.closedSockets > closedBefore,
      );
      // Eviction releases the runtime without opening anything new.
      expect(server.runtimeStartMessages).toHaveLength(1);
    } finally {
      agent.shutdown();
    }
  });

  test("reopens the same conversation transparently on the next prompt", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const { context, updates } = createContext();
    try {
      await loadConversation(agent, context, "conv-resume");
      const closedBefore = server.closedSockets;
      await waitFor(
        "the idle session to be closed",
        () => server.closedSockets > closedBefore,
      );

      const response = await agent.prompt(
        { sessionId: "conv-resume", prompt: [{ type: "text", text: "hello" }] },
        context,
      );

      expect(response).toEqual({ stopReason: "end_turn" });
      // The reopen resumed the same conversation rather than creating one, so
      // the client keeps its history and the ACP session id stays valid.
      expect(server.runtimeStartMessages).toHaveLength(2);
      expect(conversationIdsStarted(server)).toEqual([
        "conv-resume",
        "conv-resume",
      ]);
      expect(server.runtimeStartMessages[1]?.create_conversation).toBeUndefined();
      expect(server.createdAgentBodies).toHaveLength(0);
      expect(updates).toContainEqual({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "PROMPT_OK" },
      });
    } finally {
      agent.shutdown();
    }
  });

  test("concurrent requests after eviction share a single reopen", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const { context } = createContext();
    try {
      await loadConversation(agent, context, "conv-shared");
      const closedBefore = server.closedSockets;
      await waitFor(
        "the idle session to be closed",
        () => server.closedSockets > closedBefore,
      );

      const [prompt, mode] = await Promise.all([
        agent.prompt(
          { sessionId: "conv-shared", prompt: [{ type: "text", text: "hi" }] },
          context,
        ),
        agent.setSessionMode(
          { sessionId: "conv-shared", modeId: "acceptEdits" },
          context,
        ),
      ]);

      expect(prompt).toEqual({ stopReason: "end_turn" });
      expect(mode).toEqual({});
      // Two concurrent requests must not open two runtimes for one conversation.
      expect(server.runtimeStartMessages).toHaveLength(2);
    } finally {
      agent.shutdown();
    }
  });

  test("reopens only after the evicting close has released the runtime", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const { context } = createContext();
    try {
      await loadConversation(agent, context, "conv-ordered");
      const closedBefore = server.closedSockets;
      await waitFor(
        "the idle session to be closed",
        () => server.closedSockets > closedBefore,
      );
      const closedAfterEviction = server.closedSockets;

      await agent.prompt(
        { sessionId: "conv-ordered", prompt: [{ type: "text", text: "hi" }] },
        context,
      );

      // The reopen's runtime_start was issued only after the evicting close
      // had already torn the previous runtime down.
      expect(server.runtimeStartClosedSockets[1]).toBeGreaterThanOrEqual(
        closedAfterEviction,
      );
    } finally {
      agent.shutdown();
    }
  });

  test("never evicts a session while a prompt turn is in flight", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const { context } = createContext();
    try {
      const session = await openSession(agent, context);
      const promptStarted = server.nextCancelablePrompt();
      const prompt = agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "cancel active prompt" }],
        },
        context,
      );
      await promptStarted;
      const closedBefore = server.closedSockets;

      // Well past the idle window, with no adapter-side activity at all.
      await Bun.sleep(IDLE_MS * 6);
      expect(server.closedSockets).toBe(closedBefore);

      await agent.cancel({ sessionId: session.sessionId });
      expect(await prompt).toEqual({ stopReason: "cancelled" });
    } finally {
      agent.shutdown();
    }
  });

  test("never evicts a session waiting on an out-of-turn approval", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      outOfTurnPermissionTimeoutMs: 10_000,
      sessionRegistryDir: null,
    });
    // A client that never answers the permission request.
    const { context, permissionRequests } = createContext({
      answerPermissions: false,
    });
    try {
      const session = await openSession(agent, context);
      await agent.prompt(
        { sessionId: session.sessionId, prompt: [{ type: "text", text: "hi" }] },
        context,
      );
      server.requestPermissionAfterPrompt();
      await waitFor(
        "the out-of-turn permission request",
        () => permissionRequests.length > 0,
      );
      const closedBefore = server.closedSockets;

      await Bun.sleep(IDLE_MS * 6);

      // The turn is over, so only the pending approval keeps this session.
      expect(server.closedSockets).toBe(closedBefore);
    } finally {
      agent.shutdown();
    }
  });

  test("session/load during an active prompt reuses the live session", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const { context } = createContext();
    try {
      const session = await openSession(agent, context);
      const promptStarted = server.nextCancelablePrompt();
      const prompt = agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "cancel active prompt" }],
        },
        context,
      );
      await promptStarted;
      const startsBefore = server.runtimeStartMessages.length;

      await agent.loadSession(
        {
          sessionId: session.sessionId,
          cwd: "/tmp/letta-acp-test",
          mcpServers: [],
        },
        context,
      );

      // Displacing the prompt's session would orphan its runtime mid-turn.
      expect(server.runtimeStartMessages).toHaveLength(startsBefore);

      await agent.cancel({ sessionId: session.sessionId });
      expect(await prompt).toEqual({ stopReason: "cancelled" });
    } finally {
      agent.shutdown();
    }
  });

  test("runtime work reported by device status defers eviction, silence ends it", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const { context } = createContext();
    try {
      const session = await openSession(agent, context);
      await agent.prompt(
        { sessionId: session.sessionId, prompt: [{ type: "text", text: "hi" }] },
        context,
      );
      const closedBefore = server.closedSockets;

      // A run still working after the turn returned keeps reporting itself.
      for (let i = 0; i < 6; i++) {
        server.pushDeviceStatus({ isProcessing: true });
        await Bun.sleep(IDLE_MS / 2);
      }
      expect(server.closedSockets).toBe(closedBefore);

      // Once the runtime stops reporting, a turn it never terminated must not
      // pin the session forever.
      await waitFor(
        "the silent session to be closed",
        () => server.closedSockets > closedBefore,
      );
    } finally {
      agent.shutdown();
    }
  });

  test("an idle window of 0 disables eviction", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: 0,
      sessionRegistryDir: null,
    });
    const { context } = createContext();
    try {
      await loadConversation(agent, context, "conv-disabled");
      const closedBefore = server.closedSockets;

      await Bun.sleep(IDLE_MS * 6);

      expect(server.closedSockets).toBe(closedBefore);
      expect(server.runtimeStartMessages).toHaveLength(1);
    } finally {
      agent.shutdown();
    }
  });

  test("preserves the selected mode and model across a transparent reopen", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const { context, permissionRequests } = createContext();
    try {
      await loadConversation(agent, context, "conv-context");
      await agent.setSessionMode(
        { sessionId: "conv-context", modeId: "acceptEdits" },
        context,
      );
      await agent.setSessionConfigOption({
        sessionId: "conv-context",
        configId: "model",
        value: "other-model",
      });
      const closedBefore = server.closedSockets;
      await waitFor(
        "the idle session to be closed",
        () => server.closedSockets > closedBefore,
      );

      // "approval prompt" has the runtime ask to use Write, which acceptEdits
      // auto-allows and standard does not.
      const response = await agent.prompt(
        {
          sessionId: "conv-context",
          prompt: [{ type: "text", text: "approval prompt" }],
        },
        context,
      );

      expect(response.stopReason).toBe("end_turn");
      // A reopen that reset the mode would interrupt the user for an approval
      // they had already delegated for this session.
      expect(permissionRequests).toHaveLength(0);
      // The conversation's selected model is reapplied to the new runtime,
      // rather than silently reverting to the adapter default.
      expect(server.updatedModelPayloads).toEqual([
        { model_id: "other-model" },
        { model_handle: "test/other-model" },
      ]);
    } finally {
      agent.shutdown();
    }
  });

  test("a reopen landing during shutdown closes its session and fails", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const { context } = createContext();
    try {
      await loadConversation(agent, context, "conv-shutdown");
      const closedBefore = server.closedSockets;
      await waitFor(
        "the idle session to be closed",
        () => server.closedSockets > closedBefore,
      );

      // Hold the reopen open so shutdown provably lands mid-restore.
      const releaseRuntimeStart = server.blockRuntimeStart();
      const prompt = agent.prompt(
        { sessionId: "conv-shutdown", prompt: [{ type: "text", text: "hi" }] },
        context,
      );
      await waitFor(
        "the reopen to reach the runtime",
        () => server.runtimeStartMessages.length > 1,
      );
      const closedDuringRestore = server.closedSockets;

      agent.shutdown();
      releaseRuntimeStart();

      await expect(prompt).rejects.toThrow(/during shutdown/);
      // The session that registered after shutdown was closed, exactly once.
      await waitFor(
        "the late session to be closed",
        () => server.closedSockets > closedDuringRestore,
      );
    } finally {
      agent.shutdown();
    }
  });

  test("forgets the oldest evicted conversations beyond the retention bound", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const { context } = createContext();
    try {
      await agent.initialize({ protocolVersion: 1, clientCapabilities: {} });
      // One past the retention bound, so the first conversation is dropped.
      for (let i = 0; i <= 1000; i++) {
        await agent.loadSession(
          {
            sessionId: `conv-bound-${i}`,
            cwd: "/tmp/letta-acp-test",
            mcpServers: [],
          },
          context,
        );
      }
      await waitFor(
        "every idle session to be closed",
        () => server.closedSockets >= 1001,
        30_000,
      );

      // The most recent eviction is still reopenable.
      await expect(
        agent.prompt(
          {
            sessionId: "conv-bound-1000",
            prompt: [{ type: "text", text: "hi" }],
          },
          context,
        ),
      ).resolves.toEqual({ stopReason: "end_turn" });
      // The oldest was forgotten rather than retained without bound. A client
      // can still reach it through session/load, which reopens from its id.
      await expect(
        agent.prompt(
          { sessionId: "conv-bound-0", prompt: [{ type: "text", text: "hi" }] },
          context,
        ),
      ).rejects.toThrow(/Unknown session/);
    } finally {
      agent.shutdown();
    }
  });

  test("reopens for session/set_config_option, which carries no context", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const { context } = createContext();
    try {
      await loadConversation(agent, context, "conv-config-only");
      const closedBefore = server.closedSockets;
      await waitFor(
        "the idle session to be closed",
        () => server.closedSockets > closedBefore,
      );

      // The only session is evicted, so the reopen cannot borrow a context
      // from a live one. ACP sends no context with this request.
      const result = await agent.setSessionConfigOption({
        sessionId: "conv-config-only",
        configId: "model",
        value: "other-model",
      });

      expect(
        result.configOptions.find((option) => option.id === "model"),
      ).toMatchObject({ currentValue: "other-model" });
      expect(server.runtimeStartMessages).toHaveLength(2);
    } finally {
      agent.shutdown();
    }
  });

  test("a reopen uses the requesting connection's client context", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const first = createContext();
    const second = createContext();
    try {
      await loadConversation(agent, first.context, "conv-context-swap");
      const closedBefore = server.closedSockets;
      await waitFor(
        "the idle session to be closed",
        () => server.closedSockets > closedBefore,
      );
      const firstUpdatesBefore = first.updates.length;

      await agent.prompt(
        {
          sessionId: "conv-context-swap",
          prompt: [{ type: "text", text: "hi" }],
        },
        second.context,
      );

      // Eviction must not retain the old context: the reply belongs to the
      // connection that asked for it.
      expect(second.updates).toContainEqual({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "PROMPT_OK" },
      });
      expect(first.updates).toHaveLength(firstUpdatesBefore);
    } finally {
      agent.shutdown();
    }
  });

  test("review regression: shutdown retains a pending retirement that later rejects", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const sdk = instrumentSdkSessions(agent);
    const { context } = createContext();
    try {
      await loadConversation(agent, context, "conv-shutdown-pending-close");
      const closedBefore = server.closedSockets;
      const disposal = sdk.deferNextDisposal();
      await waitFor("pending retirement", () => disposal.started);
      agent.shutdown();
      disposal.reject(new Error("disposal rejected after shutdown"));
      await Bun.sleep(50);
      expect(server.runtimeStartMessages).toHaveLength(1);
      // Shutdown must either close this owned runtime or retain its failed
      // handle for reconciliation; clearing the map is not cleanup evidence.
      expect(
        server.closedSockets > closedBefore ||
          agent.unreleasedRuntimes().some(
            (held) => held.sessionId === "conv-shutdown-pending-close",
          ),
      ).toBe(true);
    } finally {
      agent.shutdown();
    }
  });

  test("shutdown keeps a pending retirement, which its own success releases", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const sdk = instrumentSdkSessions(agent);
    const { context } = createContext();
    try {
      await loadConversation(agent, context, "conv-shutdown-pending-ok");
      const closedBefore = server.closedSockets;
      const disposal = sdk.deferNextDisposal();
      await waitFor(
        "the eviction to start disposing the session",
        () => disposal.started,
      );

      // Shutdown lands before the outcome is known, so it cannot yet decide
      // whether this runtime was released or is being held.
      agent.shutdown();
      expect(server.closedSockets).toBe(closedBefore);

      disposal.resolve();

      // The disposal itself released the runtime, so there is nothing for
      // shutdown to close and nothing left held.
      await waitFor(
        "the pending disposal to release the runtime",
        () => server.closedSockets > closedBefore,
      );
      expect(agent.unreleasedRuntimes()).toEqual([]);
      expect(server.runtimeStartMessages).toHaveLength(1);
    } finally {
      agent.shutdown();
    }
  });

  test("a close failure at shutdown stays inspectable, and is not retried", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const sdk = instrumentSdkSessions(agent);
    const { context } = createContext();
    const failure = new Error("injected disposal failure");
    const held = [{ sessionId: "conv-shutdown-close-fails", error: failure }];
    try {
      await loadConversation(agent, context, "conv-shutdown-close-fails");
      const closedBefore = server.closedSockets;
      const disposal = sdk.deferNextDisposal();
      await waitFor(
        "the eviction to start disposing the session",
        () => disposal.started,
      );
      disposal.reject(failure);
      await waitFor(
        "the unreleased runtime to be recorded",
        () => agent.unreleasedRuntimes().length === 1,
      );

      // The last close shutdown can reach fails too, so nothing has proved
      // this runtime dead.
      sdk.failCloseWith(new Error("injected close failure"));
      agent.shutdown();

      expect(server.closedSockets).toBe(closedBefore);
      // Dropping the record here would leave a leaked child process with no
      // trace of it beyond a log line.
      expect(agent.unreleasedRuntimes()).toEqual(held);

      // A repeated shutdown is not a second attempt: closing again proves no
      // more than the first close did, and must not overwrite the disposal
      // failure that is still the reason this runtime is held.
      agent.shutdown();
      expect(server.closedSockets).toBe(closedBefore);
      expect(agent.unreleasedRuntimes()).toEqual(held);
    } finally {
      agent.shutdown();
    }
  });

  test("a reopen waiting on a disposal when shutdown starts opens nothing", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const sdk = instrumentSdkSessions(agent);
    const { context } = createContext();
    try {
      await loadConversation(agent, context, "conv-shutdown-midwait");
      const closedBefore = server.closedSockets;
      const disposal = sdk.deferNextDisposal();
      await waitFor(
        "the eviction to start disposing the session",
        () => disposal.started,
      );

      // The reopen is parked on the unfinished disposal when shutdown lands.
      const prompt = agent.prompt(
        {
          sessionId: "conv-shutdown-midwait",
          prompt: [{ type: "text", text: "hi" }],
        },
        context,
      );
      await Bun.sleep(IDLE_MS);
      expect(server.runtimeStartMessages).toHaveLength(1);

      agent.shutdown();
      disposal.resolve();

      await expect(prompt).rejects.toThrow(/during shutdown/);
      // The old disposal settling is not permission to start a replacement:
      // the connection that would have been served is already gone.
      expect(server.runtimeStartMessages).toHaveLength(1);
      await waitFor(
        "the evicted runtime to be released",
        () => server.closedSockets > closedBefore,
      );
      expect(agent.unreleasedRuntimes()).toEqual([]);
    } finally {
      agent.shutdown();
    }
  });

  test("a reopen waits for a disposal that is still in flight", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const sdk = instrumentSdkSessions(agent);
    const { context } = createContext();
    try {
      await loadConversation(agent, context, "conv-slow-close");
      const closedBefore = server.closedSockets;
      const disposal = sdk.deferNextDisposal();
      await waitFor(
        "the eviction to start disposing the session",
        () => disposal.started,
      );

      // The reopen begins while the previous runtime is provably still open,
      // rather than after its disposal has already finished.
      const prompt = agent.prompt(
        { sessionId: "conv-slow-close", prompt: [{ type: "text", text: "hi" }] },
        context,
      );
      await Bun.sleep(IDLE_MS * 3);
      expect(server.closedSockets).toBe(closedBefore);
      // No replacement runtime while the previous one is unreleased.
      expect(server.runtimeStartMessages).toHaveLength(1);

      disposal.resolve();

      expect(await prompt).toEqual({ stopReason: "end_turn" });
      expect(server.runtimeStartMessages).toHaveLength(2);
      // The replacement was started only once the previous runtime had closed.
      expect(server.runtimeStartClosedSockets[1]).toBeGreaterThan(closedBefore);
    } finally {
      agent.shutdown();
    }
  });

  test("a failed disposal keeps the conversation instead of replacing its runtime", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const sdk = instrumentSdkSessions(agent);
    const { context } = createContext();
    try {
      await loadConversation(agent, context, "conv-close-failure");
      const closedBefore = server.closedSockets;
      const disposal = sdk.deferNextDisposal();
      await waitFor(
        "the eviction to start disposing the session",
        () => disposal.started,
      );

      // The reopen is already waiting on the disposal when it fails.
      const prompt = agent.prompt(
        {
          sessionId: "conv-close-failure",
          prompt: [{ type: "text", text: "hi" }],
        },
        context,
      );
      await Bun.sleep(IDLE_MS * 3);
      expect(server.runtimeStartMessages).toHaveLength(1);

      const failure = new Error("injected disposal failure");
      disposal.reject(failure);

      // That runtime was never released, so no path may open a second one for
      // the same conversation or report success as if it had been retired.
      await expect(prompt).rejects.toThrow(SessionRetirementError);
      await expect(
        agent.prompt(
          {
            sessionId: "conv-close-failure",
            prompt: [{ type: "text", text: "again" }],
          },
          context,
        ),
      ).rejects.toThrow(SessionRetirementError);
      await expect(
        agent.setSessionMode(
          { sessionId: "conv-close-failure", modeId: "acceptEdits" },
          context,
        ),
      ).rejects.toThrow(SessionRetirementError);
      await expect(
        agent.setSessionConfigOption({
          sessionId: "conv-close-failure",
          configId: "model",
          value: "other-model",
        }),
      ).rejects.toThrow(SessionRetirementError);
      await expect(
        agent.loadSession(
          {
            sessionId: "conv-close-failure",
            cwd: "/tmp/letta-acp-test",
            mcpServers: [],
          },
          context,
        ),
      ).rejects.toThrow(SessionRetirementError);
      expect(server.runtimeStartMessages).toHaveLength(1);
      expect(server.closedSockets).toBe(closedBefore);

      // The rejection is held rather than swallowed, so what the adapter is
      // still holding, and why, is inspectable.
      expect(agent.unreleasedRuntimes()).toEqual([
        { sessionId: "conv-close-failure", error: failure },
      ]);

      // Shutdown releases the handle it kept: a dropped one could not be closed.
      agent.shutdown();
      await waitFor(
        "shutdown to release the retained runtime",
        () => server.closedSockets > closedBefore,
      );
      expect(agent.unreleasedRuntimes()).toEqual([]);
    } finally {
      agent.shutdown();
    }
  });

  test("a request during a reopen waits for the restored mode and model", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    const { context } = createContext();
    try {
      await loadConversation(agent, context, "conv-restore-race");
      await agent.setSessionMode(
        { sessionId: "conv-restore-race", modeId: "acceptEdits" },
        context,
      );
      await agent.setSessionConfigOption({
        sessionId: "conv-restore-race",
        configId: "model",
        value: "other-model",
      });
      const closedBefore = server.closedSockets;
      await waitFor(
        "the idle session to be closed",
        () => server.closedSockets > closedBefore,
      );

      // Hold the reopen inside its model restoration.
      const releaseUpdateModel = server.blockUpdateModel();
      const modelCallsBefore = server.updatedModelPayloads.length;
      const prompt = agent.prompt(
        {
          sessionId: "conv-restore-race",
          prompt: [{ type: "text", text: "approval prompt" }],
        },
        context,
      );
      await waitFor(
        "the reopen to reapply the saved model",
        () => server.updatedModelPayloads.length > modelCallsBefore,
      );

      // A second request arrives while the reopen is still restoring.
      const concurrent = agent.setSessionConfigOption({
        sessionId: "conv-restore-race",
        configId: PERMISSION_MODE_CONFIG_ID,
        value: "acceptEdits",
      });
      await Bun.sleep(10);
      releaseUpdateModel();

      const { configOptions } = await concurrent;
      // The concurrent caller shares the reopen instead of being handed a
      // session whose saved model and mode have not been reapplied yet.
      expect(
        configOptions.find((option) => option.id === "model"),
      ).toMatchObject({ currentValue: "other-model" });
      expect(
        configOptions.find((option) => option.id === PERMISSION_MODE_CONFIG_ID),
      ).toMatchObject({ currentValue: "acceptEdits" });
      expect(await prompt).toEqual({ stopReason: "end_turn" });
      // Both requests were served by one reopened runtime.
      expect(server.runtimeStartMessages).toHaveLength(2);
    } finally {
      agent.shutdown();
    }
  });

  test("replays the conversation's MCP servers to the SDK on reopen", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      idleEvictMs: IDLE_MS,
      sessionRegistryDir: null,
    });
    // Records what the adapter asks the SDK for without letting the SDK
    // connect: no MCP subprocess is spawned and no MCP target is contacted.
    const sdk = instrumentSdkSessions(agent, { stripMcpServers: true });
    const { context } = createContext();
    const mcpServers: McpServer[] = [
      {
        name: "notes",
        command: "/usr/bin/notes-mcp",
        args: ["--stdio"],
        env: [{ name: "NOTES_TOKEN", value: "notes-token" }],
      },
      {
        type: "http",
        name: "docs",
        url: "https://docs.invalid/mcp",
        headers: [{ name: "Authorization", value: "Bearer docs-token" }],
      },
    ];
    try {
      await loadConversation(agent, context, "conv-mcp-replay", mcpServers);
      const closedBefore = server.closedSockets;
      await waitFor(
        "the idle session to be closed",
        () => server.closedSockets > closedBefore,
      );

      await agent.prompt(
        { sessionId: "conv-mcp-replay", prompt: [{ type: "text", text: "hi" }] },
        context,
      );

      expect(server.runtimeStartMessages).toHaveLength(2);
      // The reopened session is configured with exactly the servers the client
      // supplied for the original session.
      expect(sdk.mcpServerOptions).toEqual([
        toSdkMcpServers(mcpServers),
        toSdkMcpServers(mcpServers),
      ]);
    } finally {
      agent.shutdown();
    }
  });
});
