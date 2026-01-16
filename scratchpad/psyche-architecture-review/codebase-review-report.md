# Psyche Codebase Architecture Review Report

**Agent**: Codebase Review Agent
**Date**: 2026-01-16
**Task**: Review current Psyche codebase structure in relation to identified issues

---

## Executive Summary

This report provides a comprehensive analysis of Psyche's current architecture, mapping the codebase structure to the identified issues from the improvement planning document. The codebase shows a functional but prototype-level implementation with several architectural concerns that should be addressed before further development.

**Key Findings:**
- The architecture has good separation between UI (Textual) and logic (MemoryServer)
- Memory system has sound foundations but broken execution workflows
- Tool system works but has poor UX (raw JSON display)
- No interruption support for streaming or tool execution
- Several technical debt items and quick hacks need addressing

---

## 1. Architecture Overview

### 1.1 Component Map

```
Psyche/
src/psyche/
+-- cli.py                    # Entry point, subprocess management
+-- client/
|   +-- app.py               # Textual TUI application
|   +-- repl.py              # Alternative REPL interface (legacy)
|   +-- widgets/
|       +-- chat_view.py     # Streaming message display
|       +-- thought_panel.py # Internal thought display
|       +-- tool_activity.py # Tool execution status
|       +-- sidebar.py       # Emotional state display
|       +-- user_input.py    # Command and message input
+-- memory/
|   +-- server.py            # Main inference loop, context management
|   +-- compaction.py        # Token-based sliding window
+-- mcp/
|   +-- client.py            # Elpis and Mnemosyne MCP clients
+-- tools/
    +-- tool_engine.py       # Async tool orchestrator
    +-- tool_definitions.py  # Tool schema definitions
    +-- implementations/     # Individual tool implementations
```

### 1.2 Data Flow

```
User Input
    |
    v
+-------------+     callbacks      +----------------+
| app.py      | <----------------> | MemoryServer   |
| (Textual)   |   (on_token,       | (server.py)    |
+-------------+    on_thinking)    +----------------+
    |                                     |
    | widgets                             | MCP
    v                                     v
+-------------+                   +----------------+
| ChatView    |                   | Elpis MCP      |
| ThoughtPanel|                   | (inference)    |
| ToolActivity|                   +----------------+
+-------------+                           |
                                          v
                                  +----------------+
                                  | Mnemosyne MCP  |
                                  | (memory store) |
                                  +----------------+
```

---

## 2. Issue Analysis

### 2.1 UI Issues

#### Main chat streams but dream state doesn't

**Location**:
- `server.py` lines 822-967 (`_generate_idle_thought`)
- `app.py` line 120 (`_on_token` callback)

**Current Implementation**:
- Main chat: Tokens streamed via `_on_token` callback -> `chat.append_token()` -> `StreamingMessage` widget
- Dream state: Generates text but stores final result only in `ThoughtPanel` (server.py:946-955)
- No token-by-token streaming for idle thoughts

**What Needs to Change**:
1. Modify `_generate_idle_thought()` to stream tokens similar to user input processing
2. Add "Dream Mode" visual indicator in UI
3. Differentiate dream streaming from chat streaming in UI

**Complexity**: Medium

#### No hints about /quit or slash commands

**Location**:
- `app.py` lines 213-256 (`_process_command`, `_show_help`)
- `user_input.py` - Input widget

**Current Implementation**:
- Help shown only when `/help` typed explicitly
- Input placeholder says "Type your message... (/ for commands)" - subtle
- Short aliases exist (`/q`, `/clear`) but not advertised

**What Needs to Change**:
1. Show help on first startup or add persistent hint
2. Add command completion/suggestions
3. Document all aliases clearly in help

**Complexity**: Low

---

### 2.2 Interruption Issues

#### Cannot interrupt LLM during talking/dreaming or tool use

**Location**:
- `server.py` lines 370-468 (inference loop)
- `server.py` lines 470-574 (user input processing)
- `cli.py` lines 127-131 (keyboard interrupt handling)

**Current Implementation**:
- Main inference uses `asyncio.wait_for()` with timeout but no interrupt during streaming
- Idle thoughts run to completion without interrupt
- Tool execution is sequential with no cancellation
- `Ctrl+C` at TUI level exits app but may leave inference running

**What Needs to Change**:
1. Add interrupt flag/event for streaming
2. Check flag in token poll loop (`mcp/client.py` generate_stream)
3. Add timeout/cancellation token to tool execution
4. Show "interrupted" status and preserve partial responses
5. Proper state cleanup on interrupt

**Complexity**: High

---

### 2.3 Memory Issues

#### Compaction/storage system flawed; agent doesn't store to memory on shutdown

**Location**:
- `server.py` lines 1035-1097 (staging and consolidation)
- `server.py` lines 1153-1330 (shutdown)
- `compaction.py` lines 83-150 (sliding window)

**Current Implementation**:
- **Compaction**: Only sliding window - drops oldest messages
- `summarize_fn` parameter exists but is never used
- `_staged_messages` buffer exists but never gets populated during normal operation
- **Shutdown flow** (`shutdown_with_consolidation`):
  - DOES store staged messages (lines 1291-1303)
  - Stores remaining context (lines 1300-1303)
  - Generates conversation summary (lines 1305-1308)
  - BUT: `_staged_messages` is often empty because compaction drops without staging
- **Mnemosyne connection**: Only established if both client provided AND consolidation enabled

**Critical Bug**: Messages dropped by compaction are not staged for later storage

**What Needs to Change**:
1. Fix staging mechanism - populate `_staged_messages` when compaction occurs
2. Implement actual summarization (not just truncation)
3. Ensure shutdown is always called even on abnormal exit
4. Store important memories incrementally, not just on shutdown
5. Add local fallback for when Mnemosyne disconnected

**Complexity**: High

---

### 2.4 Tool Issues

#### Tool implementations unintuitive; JSON dumps in chat

**Location**:
- `tool_engine.py` lines 264-325 (tool execution)
- `server.py` lines 576-695 (tool call parsing, execution, result formatting)
- `tool_activity.py` (status display)

**Current Implementation**:
- Tool results formatted as: `json.dumps(result, indent=2)` (server.py:665)
- Full JSON embedded in chat context (visible to LLM, not to user)
- Tool activity widget shows only name + status icons, not details
- Logging only at end of execution with duration

**What Needs to Change**:
1. Transform tool results into human-readable summaries
2. Show contextual information (e.g., "Reading 42 lines" not raw JSON)
3. Expand tool activity widget to show results/errors
4. Add structured logging with intent, action, result
5. Surface tool failures clearly to user

**Complexity**: Medium

---

### 2.5 Reasoning Issues

#### No reasoning/thinking step before responses

**Location**:
- `server.py` lines 242-282 (system prompt)
- `server.py` lines 968-1003 (idle reflection - separate path)

**Current Implementation**:
- System prompt instructs "respond conversationally", "respond naturally"
- No explicit reasoning phase before responses
- Idle reflection exists but is separate from user responses
- Dream thoughts don't influence main conversation

**What Needs to Change**:
1. Add optional reasoning step in system prompt
2. Parse reasoning from response (e.g., `<thinking>` tags)
3. Route reasoning to ThoughtPanel
4. Allow toggling reasoning display
5. Consider performance impact for local models

**Complexity**: Medium-High

---

### 2.6 Interoperability Issues

#### External MCP server support unclear; duplicate memory tools

**Location**:
- `mcp/client.py` - Elpis and Mnemosyne clients
- `tools/implementations/memory_tools.py` - Psyche memory tools
- `server.py` lines 174-240 (tool registration)
- `cli.py` lines 91-96 (hardcoded server commands)

**Current Implementation**:
- Only Elpis and Mnemosyne supported (hardcoded in cli.py)
- No dynamic MCP server loading or discovery
- **Memory tools duplication**:
  - Psyche implements `recall_memory` and `store_memory` (server.py)
  - These wrap Mnemosyne MCP client (memory_tools.py)
  - Redundant layer creates confusion
- Tool execution is solid (async, timeout-protected, error handling)
- `SAFE_IDLE_TOOLS` limits tools during reflection (hardcoded list)

**What Needs to Change**:
1. Add mechanism to register external MCP servers dynamically
2. Remove Psyche wrapper layer for memory (or justify it for caching)
3. Unified tool discovery and documentation
4. Better error messages for tool failures
5. Streaming progress for long-running tools

**Complexity**: High

---

## 3. Architectural Patterns Analysis

### 3.1 Good Patterns Found

| Pattern | Location | Description |
|---------|----------|-------------|
| Callback-based UI | `app.py`, `server.py` | Clean separation between logic and UI via callbacks |
| State machine | `server.py` `ServerState` | Clear state transitions with enum |
| Async-first | Throughout | Proper asyncio usage |
| Tool validation | `tool_engine.py` | Pydantic models ensure type-safe arguments |
| Graceful degradation | `server.py` | Optional Mnemosyne doesn't break core functionality |

### 3.2 Anti-Patterns Found

| Anti-Pattern | Location | Issue |
|--------------|----------|-------|
| Blocking stdin in Textual | `cli.py:124-125` | stderr redirected to file as workaround |
| Silent failures | Multiple exception handlers | Catch and log but don't propagate to UI |
| Callback hell | `app.py` | Multiple nested `call_later()` calls |
| Mixed concerns | `server.py` | Single class does inference, tools, memory, compaction |
| Polling for streaming | `mcp/client.py:226-245` | Uses polling instead of true streaming |
| Hardcoded commands | `cli.py:64-65` | "elpis-server" and "mnemosyne-server" hardcoded |
| Inconsistent errors | Multiple | Mix of technical JSON and user-friendly messages |
| Blocking tool execution | `server.py` | Tools run in main inference thread |

---

## 4. Technical Debt Inventory

| Item | Location | Description | Priority |
|------|----------|-------------|----------|
| MCP library patch | `cli.py:26` | Patching "dictionary keys changed during iteration" | Low |
| Exception unwrapping | `server.py:325-347` | Complex exception handling for subprocess issues | Medium |
| Dead code: staged_messages | `server.py:159` | Buffer exists but never populated | High |
| Incomplete summarization | `compaction.py` | `summarize_fn` parameter unused | High |
| Timezone-naive datetime | `mnemosyne/models.py` | Uses `datetime.now()` without timezone | Low |
| Hardcoded result limits | `server.py` | 16000 chars main, 8000 idle | Medium |
| No rate limiting | `server.py` | Consolidation can trigger frequently | Low |

---

## 5. Complexity Assessment Summary

| Issue Area | Complexity | Impact | Priority |
|------------|-----------|--------|----------|
| UI Streaming (dream state) | Medium | User experience | Medium |
| Command hints/help | Low | Usability | Low |
| Interruption support | High | User control | High |
| Memory storage fix | High | Data persistence | Critical |
| Tool display | Medium | User clarity | Medium |
| Reasoning workflow | Medium-High | Response quality | Medium |
| MCP interoperability | High | Extensibility | Medium |

---

## 6. File-by-File Summary

| File | Lines | Role | Health |
|------|-------|------|--------|
| `server.py` | ~1400 | Core logic, too many responsibilities | Needs refactor |
| `app.py` | ~300 | TUI app, reasonable | Good |
| `compaction.py` | ~150 | Context management | Incomplete |
| `tool_engine.py` | ~350 | Tool orchestration | Good |
| `mcp/client.py` | ~250 | MCP communication | Good |
| `cli.py` | ~140 | Entry point, subprocess management | Has workarounds |

---

## 7. Recommendations

### Immediate (Critical)

1. **Fix memory storage bug** - Ensure `_staged_messages` is populated during compaction
2. **Add shutdown handlers** - Catch signals to ensure `shutdown_with_consolidation` is called

### Short-term

3. **Implement clean tool display** - Transform JSON to human-readable summaries
4. **Add interruption support** - At minimum, interrupt streaming generation
5. **Show help on startup** - Improve discoverability of commands

### Medium-term

6. **Refactor MemoryServer** - Split into separate concerns (inference, tools, memory)
7. **Add reasoning workflow** - Implement `<thinking>` tag parsing
8. **Stream dream state** - Unify streaming for all generation modes

### Long-term

9. **MCP plugin system** - Dynamic server registration
10. **Provider abstraction** - Support multiple LLM backends

---

## 8. Conclusion

Psyche's codebase shows a working prototype with good foundational patterns (async, callbacks, state machines) but several execution issues that need addressing. The most critical issue is the broken memory storage workflow - conversations are not being persisted correctly. The second priority is user control through interruption support.

The architecture is sound enough that these issues can be fixed incrementally without a full rewrite, but `server.py` should be refactored to separate concerns before adding more features.

---

**Report Status**: Complete
**Files Reviewed**: 15 Python files in src/psyche/
**Agent**: Codebase Review Agent
**Date**: 2026-01-16
