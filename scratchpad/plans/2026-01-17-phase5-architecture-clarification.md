# Plan: Phase 5 Architecture Clarification

## Problem Statement

The current architecture has a fundamental confusion: Hermes directly instantiates PsycheCore as a library, when it should be a **client** that connects to a Psyche **server**.

### User's Key Insights

1. Clients should only access emotion/memory through MCP (not library imports)
2. Psyche's dreams are private - internal server processing when no clients connected
3. Hermes should be a CLIENT of Psyche server, not a wrapper around PsycheCore
4. Idle (file exploration) is Hermes-specific, distinct from Dreaming
5. Psyche should NOT dream while a client is connected

### Reference: Phase 5 Workplan Vision

From `scratchpad/plans/2026-01-16-psyche-substrate-workplan-v2.md`:

> **Tools and ReAct loop are NOT in Psyche Core. They belong to agents (including Psyche TUI).**

> **Agents provide all tools** (file ops, bash, search, etc)
> **Elpis feels and infers** (emotional state + modulated inference)
> **Mnemosyne stores Elpis's emotional memories** (one per Elpis instance)
> **Psyche coordinates** (working memory + automatic memory management)
> **Standard protocols only** (OpenAI HTTP, standard MCP - no custom APIs)

## How OpenAI's Tool Calling Works

From [OpenAI Function Calling docs](https://platform.openai.com/docs/guides/function-calling):

```
1. Client sends: messages + tools definition
2. Server returns: response with tool_calls (if any)
3. CLIENT executes tools locally
4. Client sends: tool results back to server
5. Server generates: final response
```

**Key insight:** Server returns `tool_calls`, **CLIENT executes them**. Server does NOT execute tools.

This is the pattern Psyche should follow:
- Psyche server returns `tool_calls`
- Hermes (client) executes tools in its workspace
- Psyche server is workspace-agnostic

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     PSYCHE SERVER                           │
│                                                             │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────┐     │
│  │    Elpis     │  │   Mnemosyne   │  │  DreamHandler │     │
│  │  (internal)  │  │   (internal)  │  │   (internal)  │     │
│  └──────────────┘  └───────────────┘  └───────────────┘     │
│                                                             │
│  PsycheCore: Working memory, auto-retrieval, importance     │
│                                                             │
│  External Interface:                                        │
│  ├─ HTTP: /v1/chat/completions (OpenAI-compatible)          │
│  └─ MCP: chat(), recall(), store(), get_emotion()           │
│                                                             │
│  Returns tool_calls in response - does NOT execute them     │
│                                                             │
│  Dreams when NO clients connected (memory palace)           │
└─────────────────────┬───────────────────────────────────────┘
                      │
        HTTP / MCP (standard protocols)
                      │
        ┌─────────────┴─────────────────┐
        │                               │
   ┌────▼─────┐                  ┌──────▼─────┐
   │  HERMES  │                  │   AIDER    │
   │  (TUI)   │                  │  OPENCODE  │
   │          │                  │  CONTINUE  │
   │ + Tools  │                  │            │
   │ + ReAct  │                  │  + Tools   │
   │ + Idle   │                  │  + ReAct   │
   └──────────┘                  └────────────┘

   Executes tools locally        Executes tools locally
   IdleHandler explores          Each has own idle behavior
   workspace when user           (or none)
   is inactive
```

## Three Distinct States

| State | Location | Trigger | Behavior | Tools |
|-------|----------|---------|----------|-------|
| **Awake** | Psyche server | Client connected | Serving requests, returning tool_calls | None (server-side) |
| **Idle** | Hermes client | User inactive, still connected | Workspace exploration | SAFE_IDLE_TOOLS |
| **Dreaming** | Psyche server | No client connected | Memory palace introspection | None (generative) |

### Key Distinctions

**Idle vs Dreaming:**
- **Idle**: CLIENT-side behavior when user is inactive but connection open. Hermes explores workspace.
- **Dreaming**: SERVER-side behavior when NO clients connected. Psyche explores memory palace.

**Tool Execution:**
- **Psyche server**: Returns `tool_calls` in response, does NOT execute
- **Clients (Hermes, etc)**: Execute tools locally, send results back

## What This Means for the Codebase

### IdleHandler stays in Hermes (client-side)
- File exploration is workspace-specific
- Each client can have its own idle behavior (or none)
- IdleHandler uses SAFE_IDLE_TOOLS (filesystem access)
- Only active when TUI is running and user is idle

### DreamHandler lives in Psyche server (new)
- Memory palace is server-state, not client-state
- Dreams happen when server has no active connections
- DreamHandler uses NO tools - purely generative with memory context
- Runs only on Psyche server, not in any client

### ReactHandler refactor
- Currently in `psyche/handlers/` - used by Hermes directly
- Should become: Server returns tool_calls, client runs ReAct loop
- Or: Keep in Hermes, just change how it connects to Psyche

### Memory Operations: Internal vs External

Psyche server needs INTERNAL memory operations (not exposed as tool_calls):

| Operation | When | Purpose |
|-----------|------|---------|
| **Auto-retrieval** | Before generating response | Query Mnemosyne for relevant memories |
| **Auto-storage** | After generating response | Store important exchanges (heuristic) |
| **Dream seeding** | When dreaming | Load memories for memory palace context |

These are internal calls from Psyche server → Mnemosyne (via MnemosyneClient MCP).

**External memory tools** (optional MCP tools exposed to clients):
- `recall_memory(query, n)` - Explicit memory retrieval
- `store_memory(content, importance)` - Explicit memory storage
- `get_emotion()` - Read emotional state
- `update_emotion(event)` - Report emotional event

Clients get automatic memory management by default. MCP tools are for explicit control.

```
Psyche Server
  │
  ├── Internal: Psyche → Mnemosyne (auto-retrieval, auto-storage, dream seeding)
  │             Psyche → Elpis (inference, emotion)
  │
  └── External: Client → Psyche (MCP tools for explicit memory/emotion control)
                        (optional, most clients won't need these)
```

## Implementation Phases

### Phase 5A: Server Infrastructure (2 sessions)
1. Create `src/psyche/server/http.py` - FastAPI with `/v1/chat/completions`
2. Create `src/psyche/server/mcp.py` - MCP server with chat/memory tools
3. Implement `psyche/cli.py` to launch server daemon
4. Server returns `tool_calls`, does NOT execute tools

### Phase 5B: Client Refactor (2 sessions)
1. Implement `RemotePsycheClient` (HTTP/MCP to Psyche server)
2. Refactor `hermes/cli.py` to be thin client launcher
3. Hermes owns IdleHandler and tool execution
4. Hermes runs ReAct loop locally

### Phase 5C: Dream Infrastructure (2 sessions)
1. Create `src/psyche/handlers/dream_handler.py`
2. Implement connection tracking in Psyche server
3. Add dream scheduling when no clients connected
4. Wire wake-on-connect protocol

## Answered Questions

**MCP vs HTTP for Psyche server?**
→ Both. HTTP `/v1/chat/completions` for OpenAI-compat agents. MCP for Psyche-aware clients.

**Who handles tool execution?**
→ **Clients**. Psyche server returns `tool_calls`, clients execute locally. This matches OpenAI's pattern.

**How does workspace work?**
→ Server is workspace-agnostic. Each client (Hermes) has its own workspace for tool execution.

**Should IdleHandler be optional in Hermes?**
→ Yes, configuration flag. Some users want chat only.

## Verification

After implementation:
1. `psyche-server` launches HTTP + MCP interfaces
2. `hermes` connects to Psyche server (remote or local)
3. Hermes executes tools locally in its workspace
4. Hermes IdleHandler explores workspace (client-side)
5. When all clients disconnect, Psyche server may dream
6. External agents (Aider, OpenCode) can connect via `/v1/chat/completions`

## Sources

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [OpenAI Migrate to Responses API](https://platform.openai.com/docs/guides/migrate-to-responses)
- Internal: `scratchpad/plans/2026-01-16-psyche-substrate-workplan-v2.md`

If you need specific details from before exiting plan mode (like exact code snippets, error messages, or content you generated), read the full             
  transcript at: /home/lemoneater/.claude/projects/-home-lemoneater-Projects-Elpis/4a421e12-7a3f-4f6b-a8c0-094f17d2f298.jsonl  