import {
  type AgentContext,
  type CancelNotification,
  type ContentBlock,
  type InitializeRequest,
  type InitializeResponse,
  type LoadSessionRequest,
  type LoadSessionResponse,
  type ListSessionsRequest,
  type ListSessionsResponse,
  type McpServer,
  methods,
  type NewSessionRequest,
  type NewSessionResponse,
  PROTOCOL_VERSION,
  type PromptRequest,
  type PromptResponse,
  type RequestPermissionResponse,
  type SessionConfigOption,
  type SetSessionConfigOptionRequest,
  type SetSessionConfigOptionResponse,
  type SetSessionModeRequest,
  type SetSessionModeResponse,
  type StopReason,
} from "@agentclientprotocol/sdk";
import {
  type CanUseToolContext,
  type CanUseToolResponse,
  LettaAgentClient,
  type LettaCodeSession,
  type LettaConversation,
  type McpServers,
  type MessageContentItem,
  type PermissionMode,
  type SDKMessage,
  type SDKResultMessage,
} from "@letta-ai/letta-agent-sdk";
import { authMethodsForClient } from "./auth.js";
import {
  DEFAULT_ACP_MODEL,
  DEFAULT_IDLE_EVICT_MS,
  type LettaAcpConfig,
} from "./config.js";
import {
  createEditorTools,
  type EditorFsCapabilities,
} from "./editor-tools.js";
import { historyToUpdates } from "./history-replay.js";
import { withAcpRequestTimeout } from "./request-timeout.js";
import { SessionRegistry } from "./session-registry.js";
import {
  editorReadAutoAllows,
  isSessionModeId,
  modeAutoAllows,
  PERMISSION_MODE_CONFIG_ID,
  permissionModeConfigOption,
  sessionModeState,
} from "./session-modes.js";
import {
  buildAvailableCommands,
  isExecuteCommand,
} from "./slash-commands.js";
import {
  accumulateToolInput,
  isTerminalOutputTool,
  parseToolOutput,
  type ToolCallInputState,
  toolDiffContent,
  toolKind,
  toolLocations,
  toolOutputLine,
  toolResultContent,
  toolTitle,
} from "./tool-info.js";

interface StreamedToolCall {
  /** First non-placeholder name seen for this tool call. */
  toolName: string;
  input: ToolCallInputState;
  /** Whether the ACP card already contains a native diff. */
  hasDiff: boolean;
  /** Whether terminal_info has created a native terminal for this call. */
  hasTerminal: boolean;
}

interface AcpSessionState {
  session: LettaCodeSession;
  /** Connection-scoped ACP client context used for requests and notifications. */
  clientContext: AgentContext;
  /** Most recent tool_call streamed, to correlate permission requests. */
  lastToolCall: { id: string; name: string } | null;
  /** Tool calls still streaming or running, keyed by tool call id. */
  toolCalls: Map<string, StreamedToolCall>;
  /** Tools the user chose "always allow" for, scoped to this session. */
  alwaysAllowed: Set<string>;
  cancelled: boolean;
  /** True only while a session/prompt handler is running. */
  promptActive: boolean;
  /** ACP session mode; enforced adapter-side in the permission callback. */
  modeId: PermissionMode;
  /** Session working directory (for project skill discovery). */
  cwd: string;
  /** Model handle reported by the runtime, updated after ACP model changes. */
  currentModel: string | undefined;
  /** Client-provided MCP servers, replayed when the session is reopened. */
  mcpServers: readonly McpServer[] | undefined;
  /** Permission requests awaiting a client answer; blocks idle eviction. */
  pendingPermissions: number;
  /** Last observed activity, used only by idle eviction. */
  lastActivityAt: number;
  /** Detaches the runtime device-status listener when the session closes. */
  removeDeviceStatusListener: (() => void) | null;
}

/**
 * What idle eviction must remember to reopen a conversation on demand.
 *
 * Deliberately excludes `clientContext` and every other connection-scoped or
 * per-turn field: the reopen uses the requesting call's context, and retaining
 * the old one would pin a dead ACP connection for the life of the process.
 */
interface EvictedSessionRecord {
  cwd: string;
  modeId: PermissionMode;
  mcpServers: readonly McpServer[] | undefined;
  alwaysAllowed: Set<string>;
  currentModel: string | undefined;
}

/** Max history messages replayed on session/load. */
const LOAD_HISTORY_LIMIT = 200;
/** Number of project sessions returned in one session/list page. */
const SESSION_LIST_PAGE_SIZE = 50;

/**
 * Liveness bound for permission requests raised outside a prompt turn.
 *
 * ACP scopes `session/request_permission` to a turn: the client's only
 * obligation to answer is the `session/cancel` contract, which no longer
 * applies once the prompt handler has returned. A client with no out-of-turn
 * permission UI would leave the Letta-side tool call pending forever, so those
 * requests get a generous-but-finite deadline. In-turn requests stay unbounded.
 */
const DEFAULT_OUT_OF_TURN_PERMISSION_TIMEOUT_MS = 300_000;

/**
 * How long to keep a turn open after the server reported completion while
 * still listing active runs. Long enough to cover the next model step, short
 * enough that a server which never speaks again cannot wedge the prompt.
 */
const DEFAULT_PREMATURE_RESULT_GRACE_MS = 30_000;

/**
 * How often idle sessions are swept. Capped by the idle window itself so a
 * short window (tests, aggressive deployments) is still honoured promptly.
 */
const IDLE_EVICT_SWEEP_MS = 60_000;

/**
 * Conversations remembered for transparent reopen after eviction, oldest
 * dropped first. Each record is small and holds no connection state, but a
 * long-lived adapter serving many conversations still needs a bound.
 */
const EVICTED_SESSION_MEMORY_LIMIT = 1000;

/** Safety cap on stream rounds while waiting for a slash command to finish. */
const EXECUTE_COMMAND_MAX_ROUNDS = 50;

type PumpOutcome =
  | { kind: "result"; result: SDKResultMessage }
  /** Turn reported complete while the server still listed active runs. */
  | { kind: "premature"; result: SDKResultMessage }
  | { kind: "idle" }
  | { kind: "stream_end" };

const IMAGE_MEDIA_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
]);

/**
 * A runtime this adapter asked to release, with the outcome of that attempt.
 *
 * The handle is retained for the whole attempt and, when disposal rejects, for
 * the life of the process: a runtime nobody proved dead is still holding an
 * app-server child process and its MCP subprocesses.
 */
interface RetiringRuntime {
  session: LettaCodeSession;
  /** Resolves when the disposal attempt has finished, successfully or not. */
  settled: Promise<void>;
  /** True once disposal rejected, which leaves the runtime unreconciled. */
  failed: boolean;
  /** The rejection, kept for diagnostics and for the error callers receive. */
  error: unknown;
  /**
   * True once shutdown's final close has been attempted for this runtime, so a
   * repeated shutdown cannot close it twice or overwrite the first outcome.
   */
  finalCloseAttempted: boolean;
}

/**
 * Raised when a conversation's previous runtime could not be released.
 *
 * Eviction exists to stop idle runtimes from accumulating, so a failed
 * disposal must not be answered by opening a second runtime for the same
 * conversation: that doubles the leak and drops the handle that still needs
 * reconciling. Every path that would reopen the conversation fails with this
 * instead, and the retained handle stays inspectable until shutdown.
 */
export class SessionRetirementError extends Error {
  constructor(
    readonly sessionId: string,
    cause: unknown,
  ) {
    super(
      `Cannot reopen session ${sessionId}: releasing its previous Letta runtime failed (${String(cause)})`,
      { cause },
    );
    this.name = "SessionRetirementError";
  }
}

/**
 * Bridges ACP v1 (agent side) onto a Letta agent via the Letta Agent SDK.
 *
 * One process serves one ACP connection. Every ACP session becomes a new
 * Letta conversation on a single underlying Letta agent, so the agent's
 * memory persists across sessions and editors.
 */
export class LettaAcpAgent {
  private readonly config: LettaAcpConfig;
  private readonly client: LettaAgentClient;
  private readonly outOfTurnPermissionTimeoutMs: number;
  private readonly prematureResultGraceMs: number;
  private readonly sessionRegistry: SessionRegistry;
  private readonly sessions = new Map<string, AcpSessionState>();
  private readonly idleEvictMs: number;
  /** Evicted conversations: id -> what is needed to reopen on demand. */
  private readonly evicted = new Map<string, EvictedSessionRecord>();
  /** In-flight reopen per conversation, so concurrent requests share one. */
  private readonly restoring = new Map<string, Promise<AcpSessionState>>();
  /**
   * Runtimes being released, keyed by conversation. A reopen waits for the
   * entry; a failed one stays, so nothing can get behind it.
   */
  private readonly retiring = new Map<string, RetiringRuntime>();
  /** Operations in progress per conversation, held before their first await. */
  private readonly holds = new Map<string, number>();
  private idleSweep: ReturnType<typeof setInterval> | null = null;
  private shuttingDown = false;
  /**
   * Most recent ACP client context. One process serves one ACP connection, so
   * this reaches the same client as any session's own context. Used to reopen
   * a conversation for `session/set_config_option`, which carries none.
   */
  private lastClientContext: AgentContext | null = null;
  private agentIdPromise: Promise<string> | null = null;
  private supportsTerminalOutput = false;
  private clientFsCaps: EditorFsCapabilities = {
    readTextFile: false,
    writeTextFile: false,
  };

  constructor(config: LettaAcpConfig) {
    this.config = config;
    this.client = new LettaAgentClient(config.clientOptions);
    this.sessionRegistry = new SessionRegistry(
      config.sessionRegistryDir ?? null,
      config.sessionRegistryScope ?? config.clientOptions.backend ?? "local",
    );
    this.outOfTurnPermissionTimeoutMs =
      config.outOfTurnPermissionTimeoutMs ??
      DEFAULT_OUT_OF_TURN_PERMISSION_TIMEOUT_MS;
    this.prematureResultGraceMs =
      config.prematureResultGraceMs ?? DEFAULT_PREMATURE_RESULT_GRACE_MS;
    this.idleEvictMs = config.idleEvictMs ?? DEFAULT_IDLE_EVICT_MS;
  }

  async initialize(params: InitializeRequest): Promise<InitializeResponse> {
    const fs = params.clientCapabilities?.fs;
    this.supportsTerminalOutput =
      params.clientCapabilities?._meta?.terminal_output === true;
    this.clientFsCaps = {
      readTextFile: fs?.readTextFile === true,
      writeTextFile: fs?.writeTextFile === true,
    };
    const requested = params.protocolVersion;
    const protocolVersion =
      typeof requested === "number" && requested < PROTOCOL_VERSION
        ? requested
        : PROTOCOL_VERSION;
    return {
      protocolVersion,
      agentCapabilities: {
        loadSession: true,
        sessionCapabilities: { list: {} },
        mcpCapabilities: { http: true, sse: true },
        promptCapabilities: {
          image: true,
          embeddedContext: true,
        },
      },
      authMethods: authMethodsForClient(
        params.clientCapabilities,
        this.config.clientOptions.backend === "local",
      ),
    };
  }

  async newSession(
    params: NewSessionRequest,
    cx: AgentContext,
  ): Promise<NewSessionResponse> {
    const agentId = await this.ensureAgent();
    const { sessionId, state } = await this.openSession({
      cwd: params.cwd,
      resumeId: null,
      agentId,
      clientContext: cx,
      mcpServers: params.mcpServers,
    });
    await this.rememberSession(agentId, sessionId, params.cwd);
    log(`session ${sessionId} -> agent ${agentId} (cwd: ${params.cwd})`);
    this.announceCommands(sessionId, cx);
    return {
      sessionId,
      modes: sessionModeState(state.modeId),
      configOptions: await this.sessionConfigOptions(state),
    };
  }

  async loadSession(
    params: LoadSessionRequest,
    cx: AgentContext,
  ): Promise<LoadSessionResponse> {
    const release = this.holdSession(params.sessionId);
    try {
      return await this.loadSessionHeld(params, cx);
    } finally {
      release();
    }
  }

  private async loadSessionHeld(
    params: LoadSessionRequest,
    cx: AgentContext,
  ): Promise<LoadSessionResponse> {
    if (
      !params.sessionId.startsWith("conv-") &&
      !params.sessionId.startsWith("local-conv-")
    ) {
      throw new Error(
        `Cannot load session ${params.sessionId}: not a Letta conversation id`,
      );
    }
    const { sessionId, state } =
      (await this.liveSessionForLoad(params.sessionId, cx)) ??
      (await this.openSession({
        cwd: params.cwd,
        resumeId: params.sessionId,
        agentId: null,
        clientContext: cx,
        mcpServers: params.mcpServers,
      }));
    await this.rememberSession(state.session.agentId, sessionId, params.cwd);
    const history = await state.session.listMessages({
      order: "desc",
      limit: LOAD_HISTORY_LIMIT,
    });
    if (history.hasMore) {
      log(
        `session ${sessionId} has more than ${LOAD_HISTORY_LIMIT} messages; replaying the most recent ${LOAD_HISTORY_LIMIT}`,
      );
    }
    const updates = historyToUpdates([...history.messages].reverse(), {
      terminalOutput: this.supportsTerminalOutput,
      cwd: state.cwd,
    });
    for (const update of updates) {
      await cx.notify(methods.client.session.update, { sessionId, update });
    }
    log(`loaded session ${sessionId} (${updates.length} replayed updates)`);
    this.announceCommands(sessionId, cx);
    return {
      modes: sessionModeState(state.modeId),
      configOptions: await this.sessionConfigOptions(state),
    };
  }

  async listSessions(params: ListSessionsRequest): Promise<ListSessionsResponse> {
    const agentId = await this.ensureAgent();
    const records = await this.sessionRegistry.list(agentId, params.cwd);
    if (records.length === 0) return { sessions: [] };

    let start = 0;
    if (params.cursor) {
      const previous = records.findIndex((record) => record.sessionId === params.cursor);
      if (previous < 0) throw new Error("Invalid session/list cursor");
      start = previous + 1;
    }

    // Bound each page by registry records rather than successful lookups. Stale
    // or temporarily unavailable conversations must not turn one list request
    // into an unbounded sequence of management calls.
    const page = records.slice(start, start + SESSION_LIST_PAGE_SIZE);
    const resolved = await Promise.all(
      page.map(async (record) => {
        let conversation: LettaConversation;
        try {
          conversation = await this.client.conversations.retrieve(record.sessionId);
        } catch (error) {
          // A deleted conversation is indistinguishable here from a transient
          // management failure. Omit it for this response but retain the local
          // record so an outage cannot erase discoverability permanently.
          log(`cannot resolve listed session ${record.sessionId}: ${String(error)}`);
          return null;
        }
        if (conversation.agent_id !== agentId || conversation.archived === true) {
          await this.sessionRegistry.remove(record.sessionId);
          return null;
        }
        const title = sessionTitle(conversation);
        const updatedAt =
          conversation.last_message_at ??
          conversation.updated_at ??
          conversation.created_at ??
          record.recordedAt;
        return {
          sessionId: record.sessionId,
          cwd: record.cwd,
          ...(title ? { title } : {}),
          ...(updatedAt ? { updatedAt } : {}),
        };
      }),
    );
    const sessions = resolved.filter(
      (session): session is NonNullable<typeof session> => session !== null,
    );
    const last = page.at(-1);

    return {
      sessions,
      ...(last && start + page.length < records.length
        ? { nextCursor: last.sessionId }
        : {}),
    };
  }

  private async rememberSession(
    agentId: string | null | undefined,
    sessionId: string,
    cwd: string,
  ): Promise<void> {
    if (!agentId) {
      log(`cannot record session ${sessionId}: runtime did not report an agent id`);
      return;
    }
    try {
      await this.sessionRegistry.record(agentId, sessionId, cwd);
    } catch (error) {
      // Session creation/loading already succeeded. Do not turn a local metadata
      // write failure into a retry that creates a duplicate conversation.
      log(`failed to record session ${sessionId}: ${String(error)}`);
    }
  }

  private async sessionConfigOptions(
    state: AcpSessionState,
  ): Promise<SessionConfigOption[]> {
    const { entries } = await state.session.listModels();
    const models = [...new Map(entries.map((entry) => [entry.id, entry])).values()];
    const selected = models.find(
      (entry) =>
        entry.id === state.currentModel || entry.handle === state.currentModel,
    );
    let currentValue =
      selected?.id ??
      models.find((entry) => entry.isDefault)?.id ??
      models[0]?.id;
    const options = models.map((entry) => ({
      value: entry.id,
      name: entry.label,
      ...(entry.description ? { description: entry.description } : {}),
    }));

    // A custom model handle may be active even when it is not in the curated
    // catalog. ACP requires currentValue to identify one of the options, so
    // retain that model rather than falsely presenting a different selection.
    if (!selected && state.currentModel) {
      currentValue = state.currentModel;
      options.unshift({ value: state.currentModel, name: state.currentModel });
    }
    const configOptions: SessionConfigOption[] = [
      permissionModeConfigOption(state.modeId),
    ];
    if (currentValue) {
      configOptions.push({
        id: "model",
        name: "Model",
        description: "Model used for this Letta session",
        category: "model",
        type: "select",
        currentValue,
        options,
      });
    }
    return configOptions;
  }

  async setSessionConfigOption(
    params: SetSessionConfigOptionRequest,
  ): Promise<SetSessionConfigOptionResponse> {
    const release = this.holdSession(params.sessionId);
    try {
      return await this.setSessionConfigOptionHeld(params);
    } finally {
      release();
    }
  }

  private async setSessionConfigOptionHeld(
    params: SetSessionConfigOptionRequest,
  ): Promise<SetSessionConfigOptionResponse> {
    const state = await this.restoreSession(params.sessionId, null);
    if (!state) {
      throw new Error(`Unknown session: ${params.sessionId}`);
    }
    if (params.configId === PERMISSION_MODE_CONFIG_ID) {
      if (typeof params.value !== "string" || !isSessionModeId(params.value)) {
        throw new Error(`Unknown permission mode: ${String(params.value)}`);
      }
      state.modeId = params.value;
      log(`session ${params.sessionId} mode -> ${params.value}`);
      return { configOptions: await this.sessionConfigOptions(state) };
    }
    if (params.configId === "model") {
      if (typeof params.value !== "string") {
        throw new Error("The model configuration option requires a model id");
      }
      const result = await state.session.updateModel(params.value);
      state.currentModel = result.modelHandle ?? result.modelId ?? params.value;
      log(`session ${params.sessionId} model -> ${state.currentModel}`);
      return { configOptions: await this.sessionConfigOptions(state) };
    }
    throw new Error(`Unknown session configuration option: ${params.configId}`);
  }

  async setSessionMode(
    params: SetSessionModeRequest,
    cx: AgentContext,
  ): Promise<SetSessionModeResponse> {
    const release = this.holdSession(params.sessionId);
    try {
      return await this.setSessionModeHeld(params, cx);
    } finally {
      release();
    }
  }

  private async setSessionModeHeld(
    params: SetSessionModeRequest,
    cx: AgentContext,
  ): Promise<SetSessionModeResponse> {
    this.lastClientContext = cx;
    const state = await this.restoreSession(params.sessionId, cx);
    if (!state) {
      throw new Error(`Unknown session: ${params.sessionId}`);
    }
    if (!isSessionModeId(params.modeId)) {
      throw new Error(`Unknown mode: ${params.modeId}`);
    }
    state.modeId = params.modeId;
    log(`session ${params.sessionId} mode -> ${params.modeId}`);
    void cx.notify(methods.client.session.update, {
      sessionId: params.sessionId,
      update: {
        sessionUpdate: "current_mode_update",
        currentModeId: params.modeId,
      },
    });
    return {};
  }

  /**
   * Creates or resumes the underlying Letta session and registers state.
   *
   * The ACP session id is the Letta conversation id, which is what makes
   * session/load work across adapter restarts with no local persistence. The
   * id is only known after the runtime initializes, so callbacks close over a
   * mutable ref that is filled in before any of them can fire (they only run
   * during prompts).
   *
   * The harness always runs with permissionMode "standard"; the ACP-selected
   * mode is enforced adapter-side (see session-modes.ts).
   *
   * This is also the adapter's shutdown fence, in both directions: nothing
   * spawns a runtime without passing the admission check below, and nothing
   * becomes reachable without passing the publication check before
   * `sessions.set`. Callers deliberately do not carry their own copies — see
   * {@link shutdownRefusal}.
   */
  private async openSession(options: {
    cwd: string;
    resumeId: string | null;
    agentId: string | null;
    clientContext: AgentContext;
    mcpServers?: readonly McpServer[];
    /** Restores conversation-scoped state before the session is published. */
    prepare?: (state: AcpSessionState) => Promise<void>;
  }): Promise<{ sessionId: string; state: AcpSessionState }> {
    // Admission. Every caller reaches this line having already awaited
    // something since it was admitted — the agent lookup, a reopen in flight,
    // a pending retirement of this very conversation — so being admitted
    // before shutdown says nothing about now.
    if (this.shuttingDown) throw this.shutdownRefusal(options.resumeId);
    this.lastClientContext = options.clientContext;
    const ref = { sessionId: options.resumeId ?? "" };
    const editorTools = createEditorTools(this.clientFsCaps, {
      getSessionId: () => ref.sessionId,
      getClientContext: () =>
        this.sessions.get(ref.sessionId)?.clientContext ?? null,
    });
    const mcpServers = toSdkMcpServers(options.mcpServers);
    const sessionOptions = {
      cwd: options.cwd,
      model: this.config.model,
      permissionMode: "standard" as const,
      canUseTool: (
        toolName: string,
        toolInput: Record<string, unknown>,
        context?: CanUseToolContext,
      ) =>
        this.requestToolPermission(
          ref.sessionId,
          toolName,
          toolInput,
          context,
        ),
      ...(editorTools.length > 0 ? { tools: editorTools } : {}),
      ...(Object.keys(mcpServers).length > 0 ? { mcpServers } : {}),
    };
    const session = options.resumeId
      ? this.client.resumeSession(options.resumeId, sessionOptions)
      : this.client.createSession(options.agentId ?? "", sessionOptions);
    // Force runtime initialization so the conversation id and active model exist.
    const bootstrap = await session.bootstrapState({ limit: 1 });
    const sessionId = options.resumeId ?? session.conversationId;
    if (!sessionId) {
      session.close();
      throw new Error("Letta session did not report a conversation id");
    }
    ref.sessionId = sessionId;
    if (editorTools.length > 0) {
      log(
        `editor fs tools enabled: ${editorTools.map((tool) => tool.name).join(", ")}`,
      );
    }
    const state: AcpSessionState = {
      session,
      clientContext: options.clientContext,
      lastToolCall: null,
      toolCalls: new Map(),
      alwaysAllowed: new Set(),
      cancelled: false,
      promptActive: false,
      modeId: this.config.permissionMode,
      cwd: options.cwd,
      currentModel: bootstrap.model ?? this.config.model,
      mcpServers: options.mcpServers,
      pendingPermissions: 0,
      lastActivityAt: Date.now(),
      removeDeviceStatusListener: null,
    };
    // Runtime-side work that this adapter is not driving — a run still
    // streaming after a turn returned, or an approval the device is waiting on
    // — is only visible through device status. Each update refreshes the idle
    // deadline rather than pinning the session outright: a live run keeps
    // producing updates, while a run the runtime abandoned goes quiet and
    // becomes evictable instead of holding its child process forever.
    state.removeDeviceStatusListener = session.onDeviceStatus((status) => {
      const live = this.sessions.get(sessionId);
      if (live !== state) return;
      if (status.isProcessing || status.pendingControlRequests.length > 0) {
        state.lastActivityAt = Date.now();
      }
    });
    if (options.prepare) {
      try {
        await options.prepare(state);
      } catch (error) {
        // Nothing has seen this session, so it is this call's to clean up.
        state.removeDeviceStatusListener?.();
        await this.discardSession(sessionId, session);
        throw error;
      }
    }
    // A client may load the same conversation again while its prior ACP
    // session is still open. Retire the displaced SDK session before replacing
    // it so its sockets and per-session MCP subprocesses are not orphaned —
    // and if that retirement fails, discard this replacement instead of
    // leaving two runtimes on one conversation.
    const displaced = this.sessions.get(sessionId);
    if (displaced && displaced.session !== session) {
      this.sessions.delete(sessionId);
      displaced.removeDeviceStatusListener?.();
      this.retireSession(sessionId, displaced.session);
      try {
        await this.awaitRetirement(sessionId);
      } catch (error) {
        state.removeDeviceStatusListener?.();
        await this.discardSession(sessionId, session);
        throw error;
      }
    }
    // Publication. Shutdown can land in any of the awaits above — the
    // runtime's own bootstrap, `prepare`, a displaced runtime's retirement —
    // and this is the last moment before the session becomes reachable.
    // Publishing it now would register a runtime after shutdown had already
    // released the ones it could see, and hand it to a connection that is
    // gone. Nothing has seen this runtime, so it is unambiguously this call's
    // to release — through retirement rather than a best-effort discard, so a
    // disposal that rejects stays named by `unreleasedRuntimes()` instead of
    // only in the log. Any retirement for this conversation was awaited to get
    // here, so there is no record to displace.
    if (this.shuttingDown) {
      state.removeDeviceStatusListener?.();
      this.retireSession(sessionId, session);
      throw this.shutdownRefusal(sessionId);
    }
    this.sessions.set(sessionId, state);
    // The conversation is live again, so any eviction record for it is stale.
    this.evicted.delete(sessionId);
    this.startIdleSweep();
    return { sessionId, state };
  }

  private announceCommands(sessionId: string, cx: AgentContext): void {
    const cwd = this.sessions.get(sessionId)?.cwd ?? process.cwd();
    const availableCommands = buildAvailableCommands(cwd);
    log(`advertising ${availableCommands.length} slash commands`);
    void cx.notify(methods.client.session.update, {
      sessionId,
      update: {
        sessionUpdate: "available_commands_update",
        availableCommands,
      },
    });
  }

  async prompt(
    params: PromptRequest,
    cx: AgentContext,
  ): Promise<PromptResponse> {
    const release = this.holdSession(params.sessionId);
    try {
      return await this.promptHeld(params, cx);
    } finally {
      release();
    }
  }

  private async promptHeld(
    params: PromptRequest,
    cx: AgentContext,
  ): Promise<PromptResponse> {
    this.lastClientContext = cx;
    const state = await this.restoreSession(params.sessionId, cx);
    if (!state) {
      throw new Error(`Unknown session: ${params.sessionId}`);
    }
    state.clientContext = cx;
    state.cancelled = false;
    state.promptActive = true;

    try {
      const commandResponse = await this.maybeRunCommand(params, state, cx);
      if (commandResponse) return commandResponse;
      await state.session.send(toLettaContent(params.prompt));
      // The Letta app-server transport completes a turn with a recoverable
      // "approval_conflict" result whenever a tool needs user approval. The
      // approval itself resolves concurrently: the server sends a
      // can_use_tool control request, our canUseTool callback forwards it to
      // the ACP client, and once answered the run resumes and its events keep
      // streaming — but with no second terminal result message. So: pump the
      // first round to its result, then pump recovery rounds (blocking on the
      // stream while approvals resolve) until the run goes idle.
      let mode: "turn" | "recovery" = "turn";
      let graceMs: number | undefined;
      while (true) {
        const outcome = await this.pumpStream(
          params.sessionId,
          state,
          cx,
          mode,
          graceMs,
        );
        if (state.cancelled) return { stopReason: "cancelled" };
        switch (outcome.kind) {
          case "premature": {
            // The SDK ends its stream on any result, so the only way to stay
            // with the run is another round — the same recovery the
            // approval_conflict path uses, but on a deadline in case the
            // server really was done.
            const runIds = outcome.result.runIds?.join(", ") ?? "unknown";
            log(`turn reported complete while runs are active (${runIds})`);
            mode = "recovery";
            graceMs = this.prematureResultGraceMs;
            continue;
          }
          case "result": {
            const result = outcome.result;
            if (!result.success && result.errorCode === "approval_conflict") {
              mode = "recovery";
              graceMs = undefined;
              continue;
            }
            return this.toPromptResponse(state, result);
          }
          case "idle":
          case "stream_end":
            return { stopReason: "end_turn" };
        }
      }
    } catch (error) {
      if (state.cancelled) return { stopReason: "cancelled" };
      throw error;
    } finally {
      // The client context stays for the lifetime of the session; only the
      // in-a-turn marker is cleared, so late tool traffic still reaches the
      // client but waits on a deadline.
      state.promptActive = false;
    }
  }

  /**
   * Handles slash commands that have a non-LLM implementation: /model runs in
   * the adapter, and execute_command-backed commands run in the harness.
   * Returns null for everything else — unknown /names fall through as prompt
   * text, which is how skill invocations reach the model.
   */
  private async maybeRunCommand(
    params: PromptRequest,
    state: AcpSessionState,
    cx: AgentContext,
  ): Promise<PromptResponse | null> {
    const first = params.prompt[0];
    if (params.prompt.length !== 1 || first?.type !== "text") return null;
    const match = first.text.trim().match(/^\/([a-z0-9-]+)(?:\s+([\s\S]*))?$/i);
    if (!match) return null;
    const command = (match[1] ?? "").toLowerCase();
    const argument = match[2]?.trim();

    if (isExecuteCommand(command)) {
      return this.runExecuteCommand(params.sessionId, state, cx, command, argument);
    }
    if (command !== "model") return null;

    const reply = async (text: string) => {
      await cx.notify(methods.client.session.update, {
        sessionId: params.sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text },
        },
      });
    };
    try {
      if (!argument) {
        const result = await state.session.listModels();
        const lines = result.entries.map((model) => {
          const marks = [
            model.isDefault ? "default" : null,
            model.free ? "free" : null,
          ]
            .filter(Boolean)
            .join(", ");
          return `- \`${model.handle}\` — ${model.label}${marks ? ` (${marks})` : ""}`;
        });
        await reply(
          `Available models:\n${lines.join("\n")}\n\nSwitch with \`/model <handle>\`.`,
        );
      } else {
        const result = await state.session.updateModel(argument);
        state.currentModel = result.modelHandle ?? result.modelId ?? argument;
        await reply(`Model switched to \`${state.currentModel}\`.`);
      }
    } catch (error) {
      await reply(
        `/model failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    return { stopReason: "end_turn" };
  }

  /**
   * Dispatches a slash command to the harness via the app-server protocol's
   * execute_command, forwarding any resulting agent-turn events (e.g. /init
   * and /remember run full turns) and finishing on the slash_command_end
   * stream delta that carries the command's output.
   */
  private async runExecuteCommand(
    sessionId: string,
    state: AcpSessionState,
    cx: AgentContext,
    command: string,
    argument: string | undefined,
  ): Promise<PromptResponse> {
    log(`executing /${command} via execute_command`);
    await state.session.sendCommand({
      type: "execute_command",
      command_id: command,
      request_id: `acp-${crypto.randomUUID()}`,
      runtime: {
        agent_id: state.session.agentId,
        conversation_id: state.session.conversationId,
      },
      ...(argument ? { args: argument } : {}),
    });

    for (let round = 0; round < EXECUTE_COMMAND_MAX_ROUNDS; round++) {
      // A cancel can land between stream rounds after a nested command turn
      // produced its terminal result. Do not attach another stream listener
      // once the command has already been aborted.
      if (state.cancelled) return { stopReason: "cancelled" };
      for await (const message of state.session.stream()) {
        if (state.cancelled) return { stopReason: "cancelled" };
        if (message.type === "stream_event") {
          const event = message.event as Record<string, unknown>;
          if (
            event.message_type === "slash_command_end" &&
            event.command_id === command
          ) {
            const output = typeof event.output === "string" ? event.output : "";
            const success = event.success !== false;
            if (output) {
              await cx.notify(methods.client.session.update, {
                sessionId,
                update: {
                  sessionUpdate: "agent_message_chunk",
                  content: {
                    type: "text",
                    text: success ? output : `/${command} failed: ${output}`,
                  },
                },
              });
            }
            return { stopReason: "end_turn" };
          }
          continue;
        }
        if (message.type === "result") {
          // Commands like /init and /remember run a full agent turn. A
          // successful nested result, or a recoverable approval conflict,
          // ends this SDK stream round but not the command: its continuation
          // and slash_command_end arrive through a subsequent stream().
          if (
            message.success ||
            message.errorCode === "approval_conflict"
          ) {
            continue;
          }
          // Do not turn genuine nested-turn failures into 50 opaque recovery
          // rounds followed by "/<command> did not complete".
          return this.toPromptResponse(state, message);
        }
        await this.forwardMessage(sessionId, state, message, cx);
      }
    }
    throw new Error(`/${command} did not complete`);
  }

  /**
   * Iterates one session.stream() round, forwarding events to the client.
   * In "recovery" mode (post-approval continuation) there is no terminal
   * result message, so loop_status transitions decide when the turn is over.
   * Blocking on the stream while an approval is pending is fine — the
   * canUseTool round-trip resolves concurrently over the control channel.
   */
  private async pumpStream(
    sessionId: string,
    state: AcpSessionState,
    cx: AgentContext,
    mode: "turn" | "recovery",
    firstMessageDeadlineMs?: number,
  ): Promise<PumpOutcome> {
    let sawActivity = false;
    let idleStatusCount = 0;
    let activeRunIds: readonly string[] = [];

    const stream = state.session.stream()[Symbol.asyncIterator]();
    let awaitingFirstMessage = true;
    while (true) {
      const pending = stream.next();
      // Only the first message of a grace round is raced: a server that went
      // quiet after contradicting itself must not wedge the prompt.
      const step =
        awaitingFirstMessage && firstMessageDeadlineMs !== undefined
          ? await raceDeadline(pending, firstMessageDeadlineMs)
          : await pending;
      if (!step) {
        log("no further activity during the grace round; ending turn");
        return { kind: "idle" };
      }
      awaitingFirstMessage = false;
      if (step.done) break;
      const message = step.value;

      if (message.type === "result") {
        // An idle loop status that still lists active runs makes the SDK
        // synthesize a successful, stop-reason-less turn result. Taking it at
        // face value ends the prompt mid-run, stranding the tool calls that
        // are still streaming and pushing their approvals outside the turn.
        if (
          message.success &&
          message.stopReason == null &&
          activeRunIds.length > 0
        ) {
          return { kind: "premature", result: message };
        }
        return { kind: "result", result: message };
      }
      if (message.type === "loop_status") {
        activeRunIds = message.activeRunIds;
        // An abort after real turn activity makes the SDK enqueue this idle
        // status followed by the cancelled turn's synthesized result. Drain
        // that result before returning so the next prompt cannot consume it
        // as its own terminal event. An abort before any activity has no
        // result, so the idle status remains its only completion signal.
        if (message.status === "WAITING_ON_INPUT" && state.cancelled) {
          if (sawActivity) continue;
          return { kind: "idle" };
        }
        if (mode === "recovery") {
          if (message.status === "WAITING_ON_INPUT") {
            idleStatusCount += 1;
            // The first idle status can be stale (queued before the resume);
            // trust it once we've seen real activity or it repeats.
            if (sawActivity || idleStatusCount >= 2) {
              return { kind: "idle" };
            }
            // A grace round is the contradicted-completion path. An empty idle
            // there is the server confirming the listed runs will not continue.
            // SDK 0.8 settles those run ids with the premature result, so the
            // continuation is dropped and a second idle never arrives.
            if (
              firstMessageDeadlineMs !== undefined &&
              message.activeRunIds.length === 0
            ) {
              return { kind: "idle" };
            }
          }
          continue;
        }
      }
      const forwarded = await this.forwardMessage(
        sessionId,
        state,
        message,
        cx,
      );
      sawActivity = sawActivity || forwarded;
    }
    return { kind: "stream_end" };
  }

  async cancel(params: CancelNotification): Promise<void> {
    const state = this.sessions.get(params.sessionId);
    if (!state) return;
    state.cancelled = true;
    // Interrupted tool calls never produce a tool_result, so drop their
    // accumulated arguments here instead of leaking them for the session.
    state.toolCalls.clear();
    try {
      await state.session.abort();
    } catch (error) {
      log(`abort failed for ${params.sessionId}: ${String(error)}`);
    }
  }

  /**
   * Marks a conversation busy until the returned release runs.
   *
   * Taken synchronously at the top of every session-scoped ACP method, before
   * its first await, so a sweep that runs while the method is suspended cannot
   * evict the session out from under it. Releasing also counts as activity:
   * finishing a request is the most recent thing that happened to the session.
   */
  private holdSession(sessionId: string): () => void {
    this.holds.set(sessionId, (this.holds.get(sessionId) ?? 0) + 1);
    let released = false;
    return () => {
      if (released) return;
      released = true;
      const left = (this.holds.get(sessionId) ?? 1) - 1;
      if (left > 0) this.holds.set(sessionId, left);
      else this.holds.delete(sessionId);
      const state = this.sessions.get(sessionId);
      if (state) state.lastActivityAt = Date.now();
    };
  }

  /** Work that must never lose its session, regardless of the idle window. */
  private isSessionBusy(sessionId: string, state: AcpSessionState): boolean {
    if ((this.holds.get(sessionId) ?? 0) > 0) return true;
    if (this.restoring.has(sessionId)) return true;
    if (state.promptActive) return true;
    if (state.pendingPermissions > 0) return true;
    return false;
  }

  private startIdleSweep(): void {
    if (this.idleSweep || this.shuttingDown || this.idleEvictMs <= 0) return;
    this.idleSweep = setInterval(
      () => this.evictIdleSessions(),
      Math.min(IDLE_EVICT_SWEEP_MS, this.idleEvictMs),
    );
    // The sweep must not be the reason the process stays alive.
    this.idleSweep.unref?.();
  }

  /**
   * Closes sessions idle past the configured window and remembers how to
   * reopen them. Returns the number evicted (for tests and logging).
   *
   * Each open ACP session holds a Letta runtime — an app-server child process
   * and its per-session MCP subprocesses — for as long as it stays registered,
   * so an adapter serving many conversations otherwise accumulates idle
   * children without bound.
   */
  private evictIdleSessions(now = Date.now()): number {
    if (this.idleEvictMs <= 0 || this.shuttingDown) return 0;
    let evictedCount = 0;
    for (const [sessionId, state] of this.sessions) {
      if (this.isSessionBusy(sessionId, state)) continue;
      if (now - state.lastActivityAt < this.idleEvictMs) continue;
      this.sessions.delete(sessionId);
      state.removeDeviceStatusListener?.();
      // Re-insert so the bound below drops the least recently evicted first.
      this.evicted.delete(sessionId);
      this.evicted.set(sessionId, {
        cwd: state.cwd,
        modeId: state.modeId,
        mcpServers: state.mcpServers,
        alwaysAllowed: state.alwaysAllowed,
        currentModel: state.currentModel,
      });
      while (this.evicted.size > EVICTED_SESSION_MEMORY_LIMIT) {
        const oldest = this.evicted.keys().next();
        if (oldest.done) break;
        this.evicted.delete(oldest.value);
      }
      this.retireSession(sessionId, state.session);
      log(
        `evicted idle session ${sessionId} after ${Math.round(
          (now - state.lastActivityAt) / 1000,
        )}s`,
      );
      evictedCount++;
    }
    return evictedCount;
  }

  /**
   * Releases a session's runtime. Prefers async disposal so per-session MCP
   * subprocesses are torn down before the conversation may be reopened.
   */
  private async disposeSession(session: LettaCodeSession): Promise<void> {
    if (typeof session[Symbol.asyncDispose] === "function") {
      await session[Symbol.asyncDispose]();
    } else {
      session.close();
    }
  }

  /**
   * Best-effort release for a runtime nothing can reopen behind: a session no
   * caller has seen, or one left over at shutdown. There is no later request
   * to fail closed, so the failure is logged rather than retained.
   */
  private async discardSession(
    sessionId: string,
    session: LettaCodeSession,
  ): Promise<void> {
    try {
      await this.disposeSession(session);
    } catch (error) {
      log(`discarding session ${sessionId} failed: ${String(error)}`);
    }
  }

  /**
   * Starts releasing a conversation's runtime and records the outcome.
   *
   * The record is registered before the disposal can settle, so a reopen can
   * never slip between eviction and the close. A successful retirement drops
   * only its own entry; a failed one is kept, because the adapter still owns a
   * runtime it could not release (see {@link SessionRetirementError}). There
   * is deliberately no retry while the adapter is serving requests: a second
   * close proves nothing about the first failure, and treating it as proof is
   * what would authorize the duplicate runtime this guards against.
   */
  private retireSession(
    sessionId: string,
    session: LettaCodeSession,
  ): RetiringRuntime {
    const record: RetiringRuntime = {
      session,
      settled: Promise.resolve(),
      failed: false,
      error: undefined,
      finalCloseAttempted: false,
    };
    record.settled = this.disposeSession(session).catch((error: unknown) => {
      record.failed = true;
      record.error = error;
      log(
        `releasing session ${sessionId} failed; holding its runtime: ${String(error)}`,
      );
    });
    this.retiring.set(sessionId, record);
    void record.settled.then(() => {
      if (!record.failed) {
        if (this.retiring.get(sessionId) === record) {
          this.retiring.delete(sessionId);
        }
        return;
      }
      // A disposal that rejects after shutdown has no later request to fail
      // closed for either, so shutdown's final release has to cover it from
      // here: shutdown itself could only see a record that had not settled yet.
      if (this.shuttingDown) this.releaseHeldRuntime(sessionId, record);
    });
    return record;
  }

  /**
   * Final release attempt for a runtime a failed disposal left this adapter
   * holding, made only once shutdown has given up the ACP connection.
   *
   * This is not a retry offered as proof of retirement. By shutdown there is no
   * request left to fail closed and no reopen that could put a second runtime
   * on the conversation, so the only choice left is to close the handle or keep
   * it: a close that returns released the child process, and its record goes
   * with it; one that throws leaves the record in place, so
   * {@link unreleasedRuntimes} still names a runtime nobody proved dead rather
   * than dropping it. Attempted at most once per runtime, so a repeated
   * shutdown neither closes again nor rewrites the first outcome.
   */
  private releaseHeldRuntime(
    sessionId: string,
    record: RetiringRuntime,
  ): void {
    if (record.finalCloseAttempted) return;
    record.finalCloseAttempted = true;
    try {
      record.session.close();
    } catch (error) {
      log(`closing held runtime for ${sessionId} failed: ${String(error)}`);
      return;
    }
    if (this.retiring.get(sessionId) === record) {
      this.retiring.delete(sessionId);
    }
  }

  /**
   * The refusal both halves of {@link openSession}'s shutdown fence throw.
   *
   * Shutdown gives up the ACP connection and releases every runtime it can
   * see, so a runtime started or published after it serves no client and
   * nothing is left to notice it. There is deliberately one fence rather than a
   * check per entry point: `session/new`, `session/load` and an implicit reopen
   * are three admission paths onto one spawn site, and guarding them
   * individually is what left the explicit load unguarded twice over. Names the
   * conversation when the caller has one; `session/new` has no id until its
   * runtime reports one.
   */
  private shutdownRefusal(sessionId: string | null): Error {
    return new Error(
      sessionId
        ? `Cannot open session ${sessionId} during shutdown`
        : "Cannot open a session during shutdown",
    );
  }

  /**
   * Waits for an in-flight retirement of this conversation's runtime, and
   * refuses to continue when that retirement failed.
   */
  private async awaitRetirement(sessionId: string): Promise<void> {
    const record = this.retiring.get(sessionId);
    if (!record) return;
    await record.settled;
    if (record.failed) {
      throw new SessionRetirementError(sessionId, record.error);
    }
  }

  /**
   * Conversations whose runtime this adapter could not release, with the
   * failure that left each one unreconciled.
   *
   * These are refusing to reopen, so surface them rather than leaving an
   * operator to infer them from the log.
   */
  unreleasedRuntimes(): Array<{ sessionId: string; error: unknown }> {
    const held: Array<{ sessionId: string; error: unknown }> = [];
    for (const [sessionId, record] of this.retiring) {
      if (record.failed) held.push({ sessionId, error: record.error });
    }
    return held;
  }

  /**
   * Live state for a conversation, reopening it if it was evicted.
   *
   * Reopen is transparent: the ACP session id is the Letta conversation id, so
   * resuming it restores the same conversation with its full server-side
   * history. Returns undefined for an id this adapter has never served, which
   * callers still report as an unknown session.
   */
  private async restoreSession(
    sessionId: string,
    cx: AgentContext | null,
  ): Promise<AcpSessionState | undefined> {
    // A reopen in flight is checked first: it publishes its session before it
    // finishes, so a caller that read `sessions` instead could be handed a
    // conversation whose saved mode and model have not been reapplied yet.
    const inFlight = this.restoring.get(sessionId);
    if (inFlight) return inFlight;
    const live = this.sessions.get(sessionId);
    if (live) return live;
    // Reported before the eviction record is consulted, so the refusal keeps
    // naming the unreleased runtime even once that record has aged out.
    const retiring = this.retiring.get(sessionId);
    if (retiring?.failed) {
      throw new SessionRetirementError(sessionId, retiring.error);
    }
    const saved = this.evicted.get(sessionId);
    if (!saved || this.shuttingDown) return undefined;
    // Only reachable before any ACP request has carried a context, which
    // cannot happen for a conversation this adapter has already served.
    const clientContext = cx ?? this.lastClientContext;
    if (!clientContext) return undefined;

    const pending = (async () => {
      // The evicting retirement may still be releasing the previous runtime,
      // and one that failed keeps ownership of the conversation rather than
      // letting this reopen start a second runtime behind it.
      await this.awaitRetirement(sessionId);
      // Waiting for that release can outlast the ACP connection, and so can
      // the reopen itself. Both are refused by openSession's shutdown fence,
      // which every other way of starting a runtime goes through too; adding a
      // check here again would only make the next unguarded path harder to see.
      const { state } = await this.openSession({
        cwd: saved.cwd,
        resumeId: sessionId,
        agentId: null,
        clientContext,
        mcpServers: saved.mcpServers,
        // Applied before the session is published, so no concurrent caller
        // can observe the conversation with the adapter's default mode or the
        // runtime's default model in place of the ones it had chosen.
        prepare: async (restored) => {
          restored.modeId = saved.modeId;
          restored.alwaysAllowed = saved.alwaysAllowed;
          const model = saved.currentModel;
          if (!model || model === restored.currentModel) return;
          try {
            const result = await restored.session.updateModel(model);
            restored.currentModel =
              result.modelHandle ?? result.modelId ?? model;
          } catch (error) {
            log(
              `could not restore model ${model} for ${sessionId}: ${String(error)}`,
            );
          }
        },
      });
      log(`reopened evicted session ${sessionId}`);
      return state;
    })().finally(() => this.restoring.delete(sessionId));

    this.restoring.set(sessionId, pending);
    return pending;
  }

  /**
   * session/load for a conversation that another operation is already using
   * returns that live session instead of displacing it, so a reopen racing a
   * load cannot leave two runtimes for one conversation.
   */
  private async liveSessionForLoad(
    sessionId: string,
    cx: AgentContext,
  ): Promise<{ sessionId: string; state: AcpSessionState } | null> {
    const pending = this.restoring.get(sessionId);
    if (pending) {
      try {
        await pending;
      } catch (error) {
        // An ordinary failed reopen falls through to a plain load, but one
        // that failed because the previous runtime is still unreconciled must
        // not: the load would open exactly the duplicate runtime it refused.
        if (error instanceof SessionRetirementError) throw error;
      }
    }
    await this.awaitRetirement(sessionId);
    this.evicted.delete(sessionId);
    const live = this.sessions.get(sessionId);
    if (!live) return null;
    // More than this load's own hold, or an active turn, means another
    // operation owns the session and must keep it.
    if ((this.holds.get(sessionId) ?? 0) > 1 || live.promptActive) {
      live.clientContext = cx;
      return { sessionId, state: live };
    }
    // Nothing else owns it, so this load replaces it. Retire the runtime it is
    // replacing first, and keep ownership if that fails, rather than letting
    // the load put a second runtime on the conversation.
    this.sessions.delete(sessionId);
    live.removeDeviceStatusListener?.();
    this.retireSession(sessionId, live.session);
    await this.awaitRetirement(sessionId);
    return null;
  }

  shutdown(): void {
    this.shuttingDown = true;
    if (this.idleSweep) {
      clearInterval(this.idleSweep);
      this.idleSweep = null;
    }
    this.evicted.clear();
    for (const state of this.sessions.values()) {
      try {
        state.removeDeviceStatusListener?.();
        state.session.close();
      } catch {
        // best-effort cleanup on connection close
      }
    }
    this.sessions.clear();
    // Runtimes held back by a failed disposal are released here and nowhere
    // else: the ACP connection is gone, so there is no request left to fail
    // closed for, and leaving the handle unclosed would leak the child process
    // for the life of the adapter.
    //
    // A disposal still in flight keeps its record rather than being skipped and
    // discarded. Only its outcome says whether the runtime ended up released or
    // still held, and that outcome arrives after this loop: dropping the record
    // here would leak the handle of every disposal that rejects from now on, and
    // hide it from `unreleasedRuntimes()` as well. `retireSession` releases
    // those as they settle, which is also why nothing clears the map — what
    // remains in it is exactly what this adapter could not release.
    for (const [sessionId, record] of [...this.retiring]) {
      if (record.failed) this.releaseHeldRuntime(sessionId, record);
    }
  }

  /**
   * Streamed Letta SDK message -> ACP session/update notification.
   * Returns true when the message was substantive turn activity.
   */
  private async forwardMessage(
    sessionId: string,
    state: AcpSessionState,
    message: SDKMessage,
    cx: AgentContext,
  ): Promise<boolean> {
    // Any streamed message is observed runtime work on this conversation.
    state.lastActivityAt = Date.now();
    switch (message.type) {
      case "init":
        log(`turn started (agent ${message.agentId}, model ${message.model})`);
        return false;
      case "assistant":
        await cx.notify(methods.client.session.update, {
          sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: message.content },
          },
        });
        return true;
      case "reasoning":
        await cx.notify(methods.client.session.update, {
          sessionId,
          update: {
            sessionUpdate: "agent_thought_chunk",
            content: { type: "text", text: message.content },
          },
        });
        return true;
      case "tool_call": {
        const existing = state.toolCalls.get(message.toolCallId);
        // Continuation deltas carry an argument fragment and often no name, so
        // the SDK reports "?" for them: the first real name wins.
        const toolName =
          knownToolName(message.toolName) ??
          existing?.toolName ??
          message.toolName;
        const input = accumulateToolInput(
          existing?.input,
          message.rawArguments,
          message.toolInput,
        );
        const diffContent = toolDiffContent(toolName, input.input);
        const hasDiff = existing?.hasDiff === true || diffContent.length > 0;
        const nativeTerminal =
          this.supportsTerminalOutput && isTerminalOutputTool(toolName);
        const addTerminal = nativeTerminal && existing?.hasTerminal !== true;
        state.toolCalls.set(message.toolCallId, {
          toolName,
          input,
          hasDiff,
          hasTerminal: existing?.hasTerminal === true || addTerminal,
        });
        state.lastToolCall = { id: message.toolCallId, name: toolName };
        // Partial JSON has no usable title or locations, so hold the card
        // steady until the arguments parse rather than flickering through
        // every fragment.
        const worthSending =
          !existing ||
          (input.complete &&
            (!existing.input.complete ||
              existing.input.rawArguments !== input.rawArguments));
        if (!worthSending) return true;
        await cx.notify(methods.client.session.update, {
          sessionId,
          update: {
            sessionUpdate: existing ? "tool_call_update" : "tool_call",
            toolCallId: message.toolCallId,
            title: toolTitle(toolName, input.input),
            kind: toolKind(toolName),
            status: "in_progress",
            rawInput: input.input,
            locations: toolLocations(input.input),
            ...(diffContent.length > 0
              ? { content: diffContent }
              : addTerminal
                ? {
                    content: [{ type: "terminal" as const, terminalId: message.toolCallId }],
                    _meta: {
                      terminal_info: {
                        terminal_id: message.toolCallId,
                        cwd: state.cwd,
                      },
                    },
                  }
                : {}),
          },
        });
        return true;
      }
      case "tool_result": {
        const toolCall = state.toolCalls.get(message.toolCallId);
        const rawOutput = parseToolOutput(message.content);
        state.toolCalls.delete(message.toolCallId);
        const locations = toolCall
          ? toolLocations(toolCall.input.input, toolOutputLine(rawOutput))
          : [];
        const nativeTerminal = toolCall?.hasTerminal === true;
        if (nativeTerminal) {
          // Zed creates the display-only terminal from terminal_info on the
          // initial call, then consumes output and exit as ordered updates.
          await cx.notify(methods.client.session.update, {
            sessionId,
            update: {
              sessionUpdate: "tool_call_update",
              toolCallId: message.toolCallId,
              _meta: {
                terminal_output: {
                  terminal_id: message.toolCallId,
                  data: message.content,
                },
              },
            },
          });
        }
        const resultContent = toolResultContent(
          message.toolCallId,
          message.content,
          {
            includeTerminal: nativeTerminal,
            hasDiff: toolCall?.hasDiff,
            isError: message.isError,
          },
        );
        await cx.notify(methods.client.session.update, {
          sessionId,
          update: {
            sessionUpdate: "tool_call_update",
            toolCallId: message.toolCallId,
            status: message.isError ? "failed" : "completed",
            rawOutput,
            ...(locations.length > 0 ? { locations } : {}),
            ...(resultContent.length > 0 ? { content: resultContent } : {}),
            ...(nativeTerminal
              ? {
                  _meta: {
                    terminal_exit: {
                      terminal_id: message.toolCallId,
                      exit_code: message.isError ? 1 : 0,
                      signal: null,
                    },
                  },
                }
              : {}),
          },
        });
        return true;
      }
      case "error":
        log(`stream error: ${message.message}`);
        return false;
      default:
        // queue_update, loop_status, stream_event, retry — no ACP equivalent.
        return false;
    }
  }

  private toPromptResponse(
    state: AcpSessionState,
    result: SDKResultMessage,
  ): PromptResponse {
    if (result.success) {
      return { stopReason: "end_turn" };
    }
    let stopReason: StopReason;
    switch (result.errorCode) {
      case "interrupted":
        stopReason = "cancelled";
        break;
      case "max_steps":
        stopReason = "max_turn_requests";
        break;
      default:
        if (state.cancelled) {
          stopReason = "cancelled";
          break;
        }
        throw new Error(
          result.errorDetail ?? result.error ?? "Letta turn failed",
        );
    }
    return { stopReason };
  }

  /** Letta canUseTool callback -> ACP session/request_permission. */
  private async requestToolPermission(
    sessionId: string,
    toolName: string,
    toolInput: Record<string, unknown>,
    context: CanUseToolContext | undefined,
  ): Promise<CanUseToolResponse> {
    const state = this.sessions.get(sessionId);
    if (!state) {
      return {
        behavior: "deny",
        message: "Unknown ACP session",
      };
    }
    if (state.cancelled) {
      // The client context now outlives the turn, so nothing else stops a
      // request that raced session/abort from prompting the user again after
      // they pressed stop.
      log(`denying ${toolName}: prompt turn was cancelled`);
      return {
        behavior: "deny",
        message: "Prompt turn was cancelled",
        interrupt: true,
      };
    }
    // A session waiting on the user must not be evicted, including out of
    // turn, where the only other protection is the request's own deadline.
    state.pendingPermissions += 1;
    try {
      return await this.resolveToolPermission(
        sessionId,
        state,
        state.clientContext,
        toolName,
        toolInput,
        context,
      );
    } finally {
      state.pendingPermissions -= 1;
      state.lastActivityAt = Date.now();
    }
  }

  private async resolveToolPermission(
    sessionId: string,
    state: AcpSessionState,
    cx: AgentContext,
    toolName: string,
    toolInput: Record<string, unknown>,
    context: CanUseToolContext | undefined,
  ): Promise<CanUseToolResponse> {
    const workspaceEditorRead = await editorReadAutoAllows(
      state.modeId,
      toolName,
      toolInput,
      state.cwd,
    );
    if (modeAutoAllows(state.modeId, toolName) || workspaceEditorRead) {
      log(`auto-allowing ${toolName} (mode ${state.modeId})`);
      return { behavior: "allow", updatedInput: toolInput };
    }
    log(
      `permission requested for ${toolName}` +
        (state.promptActive ? "" : " (outside a prompt turn)"),
    );
    if (state.alwaysAllowed.has(toolName)) {
      return { behavior: "allow", updatedInput: toolInput };
    }

    // The approval carries the id of the tool call it belongs to. Without it
    // the client gets a permission card that matches no streamed tool call:
    // guessing by name attaches to the wrong card when several calls are in
    // flight, and the synthetic id orphans the card entirely.
    const toolCallId =
      context?.toolCallId ??
      (state.lastToolCall?.name === toolName
        ? state.lastToolCall.id
        : `${toolName}_${crypto.randomUUID()}`);
    const params = {
      sessionId,
      toolCall: {
        toolCallId,
        title: toolTitle(toolName, toolInput),
        kind: toolKind(toolName),
        status: "pending" as const,
        rawInput: toolInput,
        locations: toolLocations(toolInput),
      },
      options: [
        {
          optionId: "allow_once",
          name: `Allow ${toolName} once`,
          kind: "allow_once" as const,
        },
        {
          optionId: "allow_always",
          name: `Always allow ${toolName} this session`,
          kind: "allow_always" as const,
        },
        {
          optionId: "reject_once",
          name: "Reject",
          kind: "reject_once" as const,
        },
      ],
    };

    let response: RequestPermissionResponse;
    try {
      response = state.promptActive
        ? await cx.request(methods.client.session.requestPermission, params)
        : await withAcpRequestTimeout(
            (signal) =>
              cx.request(methods.client.session.requestPermission, params, {
                cancellationSignal: signal,
              }),
            {
              label: "Permission request",
              description: `approve ${toolName} outside a prompt turn`,
              timeoutMs: this.outOfTurnPermissionTimeoutMs,
            },
          );
    } catch (error) {
      log(`permission request for ${toolName} failed: ${String(error)}`);
      return {
        behavior: "deny",
        message: error instanceof Error ? error.message : String(error),
        interrupt: true,
      };
    }

    if (response.outcome.outcome === "cancelled") {
      return {
        behavior: "deny",
        message: "Prompt turn was cancelled",
        interrupt: true,
      };
    }
    switch (response.outcome.optionId) {
      case "allow_always":
        state.alwaysAllowed.add(toolName);
        return { behavior: "allow", updatedInput: toolInput };
      case "allow_once":
        return { behavior: "allow", updatedInput: toolInput };
      default:
        return { behavior: "deny", message: "User rejected this tool call" };
    }
  }

  private ensureAgent(): Promise<string> {
    if (!this.agentIdPromise) {
      this.agentIdPromise = this.resolveAgent();
      this.agentIdPromise.catch(() => {
        // Allow retry on the next session/new instead of caching the failure.
        this.agentIdPromise = null;
      });
    }
    return this.agentIdPromise;
  }

  private async resolveAgent(): Promise<string> {
    if (this.config.agentId) {
      log(`using existing agent ${this.config.agentId}`);
      return this.config.agentId;
    }
    log("creating a new Letta agent (set LETTA_AGENT_ID to reuse one)...");
    let agentId: string;
    try {
      agentId = await this.client.createAgent({
        // SDK 0.5 made personality presets opt-in. Keep the adapter's established
        // behavior instead of silently creating a zero-memory generic agent.
        personality: "memo",
        name: "Letta",
        description: "Letta agent driven through the Agent Client Protocol",
        model: this.config.model ?? DEFAULT_ACP_MODEL,
      });
    } catch (error) {
      log(
        `agent creation failed: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}`,
      );
      throw error;
    }
    log(
      `created agent ${agentId} — set LETTA_AGENT_ID=${agentId} to keep using it`,
    );
    return agentId;
  }
}

function sessionTitle(conversation: LettaConversation): string | undefined {
  const source = conversation.summary ?? conversation.description;
  if (!source) return undefined;
  const title = source.replace(/[\r\n]+/g, " ").replace(/\s+/g, " ").trim();
  if (!title) return undefined;
  return title.length <= 256 ? title : `${title.slice(0, 255)}…`;
}

/** Awaits `pending`, or resolves null when the deadline expires first. */
async function raceDeadline<T>(
  pending: Promise<T>,
  timeoutMs: number,
): Promise<T | null> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<null>((resolve) => {
    timer = setTimeout(() => resolve(null), timeoutMs);
    timer.unref?.();
  });
  try {
    return await Promise.race([pending, deadline]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

/**
 * The SDK reports "?" when a streamed tool_call chunk omits the function name,
 * which every continuation fragment does. Treat that as "unknown" so the name
 * from the opening chunk survives.
 */
function knownToolName(toolName: string): string | undefined {
  if (!toolName || toolName === "?") return undefined;
  return toolName;
}

/** Convert ACP transport shapes to the Agent SDK's keyed MCP configuration. */
export function toSdkMcpServers(
  servers: readonly McpServer[] | undefined,
): McpServers {
  const configs: Array<[string, McpServers[string]]> = [];
  for (const server of servers ?? []) {
    if (!("type" in server)) {
      configs.push([
        server.name,
        {
          command: server.command,
          args: server.args,
          env: Object.fromEntries(
            server.env.map(({ name, value }) => [name, value]),
          ),
        },
      ]);
      continue;
    }
    if (server.type === "http" || server.type === "sse") {
      configs.push([
        server.name,
        {
          type: server.type,
          url: server.url,
          headers: Object.fromEntries(
            server.headers.map(({ name, value }) => [name, value]),
          ),
        },
      ]);
      continue;
    }
    log(`MCP server "${server.name}" skipped: ACP transport is not supported`);
  }
  return Object.fromEntries(configs);
}

/** ACP prompt content blocks -> Letta multimodal message content. */
export function toLettaContent(blocks: ContentBlock[]): MessageContentItem[] {
  const content: MessageContentItem[] = [];
  for (const block of blocks) {
    switch (block.type) {
      case "text":
        content.push({ type: "text", text: block.text });
        break;
      case "image":
        if (IMAGE_MEDIA_TYPES.has(block.mimeType)) {
          content.push({
            type: "image",
            source: {
              type: "base64",
              media_type: block.mimeType as
                | "image/png"
                | "image/jpeg"
                | "image/gif"
                | "image/webp",
              data: block.data,
            },
          });
        }
        break;
      case "resource_link":
        content.push({ type: "text", text: `[Referenced file: ${block.uri}]` });
        break;
      case "resource": {
        const resource = block.resource;
        if ("text" in resource && typeof resource.text === "string") {
          content.push({
            type: "text",
            text: `<context uri="${resource.uri}">\n${resource.text}\n</context>`,
          });
        }
        break;
      }
      default:
        // audio and future block types are not advertised in promptCapabilities
        break;
    }
  }
  if (content.length === 0) {
    content.push({ type: "text", text: "" });
  }
  return content;
}

function log(message: string): void {
  // stdout carries the ACP JSON-RPC stream; all logging goes to stderr.
  console.error(`[letta-acp] ${message}`);
}
