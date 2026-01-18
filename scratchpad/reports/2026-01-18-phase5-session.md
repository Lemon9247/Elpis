# Phase 5 Session Report: Psyche as Substrate

**Date:** 2026-01-18
**Session:** Phase 5 Implementation + Tool Use Investigation

## Summary

Completed Phase 5 implementation (Psyche as Substrate), fixed multiple bugs during testing, and investigated tool use issues in remote mode.

---

## Phase 5 Implementation

### 5A: Server Infrastructure
- `src/psyche/server/http.py` - FastAPI HTTP server with OpenAI-compatible `/v1/chat/completions`
- `src/psyche/server/daemon.py` - Server lifecycle, MCP client management, connection tracking
- `src/psyche/server/mcp.py` - MCP server for Psyche-aware clients
- `src/psyche/handlers/dream_handler.py` - Dream state when no clients connected
- `src/psyche/cli.py` - Updated CLI with startup/shutdown messaging

### 5B: Client Refactor
- `src/psyche/handlers/psyche_client.py` - RemotePsycheClient HTTP implementation
- `src/hermes/cli.py` - Added `--server` option for remote mode

### 5C: Dream Infrastructure
- `src/psyche/core/server.py` - Added `generate_dream()` and `retrieve_random_memories()`

---

## Bugs Fixed

### 1. MCP Servers Not Starting
**Problem:** Daemon created client objects but never connected.
**Fix:** Use nested async context managers in `daemon.py:start()`.

### 2. No Response in Remote Mode
**Problem:** `_process_user_input` returned early when `react_handler=None`.
**Fix:** Added `_process_via_client()` fallback path.

### 3. ChatView Method Names
**Problem:** Used wrong methods (`start_assistant_message` vs `start_stream`).
**Fix:** Corrected to use `start_stream()`, `append_token()`, `end_stream()`.

### 4. Event Loop Closed Error
**Problem:** aiohttp ClientSession created in wrong event loop.
**Fix:** Use urllib for sync health check, defer aiohttp to Textual's loop.

### 5. Context Growing with Duplicates
**Problem:** HTTP server added messages without clearing, causing duplicates.
**Fix:** Clear context before rebuilding from request messages (stateless API pattern).

### 6. Shutdown Failing - "Not Connected to Elpis"
**Problem:** Shutdown ran AFTER MCP context managers exited.
**Fix:** Move shutdown inside try/finally within context managers:
```python
async with self._connect_elpis() as elpis:
    async with self._connect_mnemosyne() as mnemosyne:
        try:
            await self._run_server()
        finally:
            await self.shutdown()  # While still connected!
```

### 7. User Rejected /shutdown Endpoint
**Request:** "I don't want clients to be able to shut down the server."
**Action:** Reverted the endpoint, improved CLI messaging instead.

---

## Memory Verification

Verified Psyche's memories in ChromaDB (`data/memory/`):
- 95 short-term memories
- 0 long-term memories
- Memories from Jan 14-18 including:
  - Willow as creator/mother
  - Tripartite being (Elpis/Mnemosyne/Psyche)
  - Sheep conversation (today, saved on shutdown)

Cleaned up empty `data/chromadb/` folder (was created by test/old config).

---

## Tool Use Investigation

### Problem Identified
In remote mode, Psyche generates tool_call blocks but tools are never executed.

### Root Cause
Remote mode tool loop is incomplete:
1. Server parses tool_calls and returns them in SSE finish chunk
2. Hermes streams tokens but ignores finish chunk
3. No tool execution, no results sent back

### Current Architecture
- **Local mode:** ReactHandler parses text, executes tools, loops
- **Remote mode:** Just streams - no tool handling

### Files Involved
| File | Issue |
|------|-------|
| `hermes/app.py:394-418` | `_process_via_client()` doesn't check for tools |
| `psyche/handlers/psyche_client.py:515-561` | `generate_stream()` ignores finish chunk |
| `hermes/cli.py:405` | Creates ToolEngine but doesn't pass to app |

### Additional Discovery
Memory tools (`recall_memory`, `store_memory`) are only registered in `hermes/cli.py`, not in the base ToolEngine. This means they're unavailable in HTTP server mode.

---

## Commits Made

1. **Phase 5A:** Server infrastructure (http.py, daemon.py, mcp.py, dream_handler.py)
2. **Phase 5B:** Client refactor (RemotePsycheClient, --server option)
3. **Phase 5C:** Dream infrastructure (generate_dream, retrieve_random_memories)
4. **Fix:** Context duplication in HTTP server
5. **Fix:** Run shutdown BEFORE MCP context managers exit
6. Various incremental fixes for remote mode routing

---

## Remaining Work

### Immediate: Fix Remote Mode Tools
1. Update `generate_stream()` to capture tool_calls from finish chunk
2. Update `_process_via_client()` to implement tool execution loop
3. Pass ToolEngine to Hermes in remote mode

### Tool Organization
Need to investigate which tools should be:
- In Psyche (server-side)
- In Hermes (client-side)
- Shared between both

Current tool locations:
- `psyche/tools/` - Core tools (read_file, create_file, edit_file, execute_bash, search_codebase, list_directory)
- `psyche/tools/implementations/memory_tools.py` - Memory tools (but only registered in hermes/cli.py)

---

## Architecture Notes

### Why Text-Based Tool Parsing?
Elpis uses llama.cpp/transformers for local inference. These don't support structured tool calling like OpenAI/Anthropic APIs. Text parsing is the only option for local models.

### OpenAI-Compatible Design
The HTTP server follows OpenAI's pattern:
- Server returns `tool_calls` in response
- Client executes tools locally
- Client sends results back
- This is correct for security (server doesn't need workspace access)

### The Missing Piece
Hermes in remote mode never completes this loop - it just displays the response including the tool_call block text.

---

## Session Duration
Extended session covering implementation, testing, debugging, and research.
