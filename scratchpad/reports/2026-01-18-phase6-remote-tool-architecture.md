# Phase 6: Remote Mode Tool Architecture - Session Report

**Date:** 2026-01-18
**Branch:** phase4/architecture-refactor
**Commit:** 2550886

## Summary

Implemented the remote mode tool architecture so that when Hermes connects to a Psyche server, tools are properly executed. Also fixed a critical context window mismatch bug and added context synchronization.

## Changes Made

### 1. Client Tool Call Capture (`psyche_client.py`)

Added tracking for tool calls in `RemotePsycheClient`:

- New instance variables `_last_tool_calls` and `_last_finish_reason`
- Modified `generate_stream()` to capture tool calls from the finish chunk
- Added `get_pending_tool_calls()` method to retrieve tool calls after streaming

### 2. Hermes Tool Execution Loop (`app.py`, `cli.py`)

- Added `tool_engine` parameter to Hermes `__init__`
- Rewrote `_process_via_client()` with a tool execution loop:
  - Streams response from server
  - Checks for pending tool calls
  - Executes tools locally via ToolEngine
  - Sends results back to server
  - Loops until no more tool calls (max 10 iterations)
- CLI passes `tool_engine` to Hermes in remote mode

### 3. Server-Side Memory Tool Execution (`http.py`)

Memory tools (`recall_memory`, `store_memory`) execute server-side because:
- Server has the Mnemosyne connection
- Memory is part of Psyche's "self"

Implementation:
- Added `MEMORY_TOOLS` constant
- Added `_execute_memory_tool()` helper
- Added `_separate_tool_calls()` to split memory vs client tools
- Rewrote `_generate_response()` and `_stream_response()` with memory tool loop
- Added error handling for context overflow

### 4. Context Window Synchronization (`elpis/server.py`, `mcp/client.py`, `daemon.py`)

**The Bug:** PsycheCore was configured with `max_context_tokens=24000` but Elpis's LLM had `context_length=4096`. After adding memory tool results to context, generation failed with "Requested tokens exceed context window".

**The Fix:**
- Added `get_capabilities` MCP tool to Elpis (returns context_length, max_tokens, backend, etc.)
- Added `get_capabilities()` method to ElpisClient
- PsycheDaemon now queries Elpis capabilities after connecting
- Configures context: `max_tokens = context_length * 0.75`, `reserve = context_length * 0.20`
- Changed default fallbacks to conservative 3000/800

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PSYCHE SERVER                           │
│  - Memory tools execute INTERNALLY                          │
│  - Queries Elpis for context_length on startup              │
│  - Returns file/bash/search tool_calls to client            │
└─────────────────────────────────────────────────────────────┘
                            ↕ HTTP API
┌─────────────────────────────────────────────────────────────┐
│                     HERMES CLIENT                           │
│  - Receives tool_calls from server                          │
│  - Executes file/bash/search tools locally                  │
│  - Sends results back to server                             │
└─────────────────────────────────────────────────────────────┘
```

## Known Issues / Investigation Needed

### Memory Retrieval in Remote Mode

User reports Psyche is struggling to recall memories. Potential causes to investigate:

1. **Context clearing on each request:** In `http.py:_process_messages()`, context is cleared at the start of each request (OpenAI API pattern). This is correct for the stateless API, but need to verify memory retrieval still happens properly.

2. **Memory result size:** Added `MAX_MEMORY_RESULT_CHARS = 2000` constant but never implemented truncation. Large memory results could still overflow context.

3. **Tool result format:** Memory results are added via `core.add_tool_result()` as JSON. Need to verify the format is what Psyche expects.

4. **Timing:** Memory retrieval happens in `add_user_message()` for the last user message. If the memory tool loop adds more results, those go through `add_tool_result()` which doesn't trigger compaction handling (it's synchronous).

### Code to check:

```python
# http.py:_process_messages - Does this interfere with memory?
self.core.clear_context()  # Clears everything including memories?

# http.py:_execute_memory_tool - Is result format correct?
memories = await self.core.retrieve_memories(query, n_results)
return json.dumps({"memories": memories})  # Is this format right?

# server.py:add_tool_result - Doesn't handle compaction
def add_tool_result(self, tool_name: str, result: str) -> None:
    self._context.add_message("user", f"[Tool result for {tool_name}]:\n{result}")
    # Note: Doesn't check/handle CompactionResult like add_user_message does
```

## TODO

- [ ] Investigate memory retrieval issue in remote mode
- [ ] Implement memory result truncation (MAX_MEMORY_RESULT_CHARS)
- [ ] Make `add_tool_result` async and handle compaction
- [ ] Refactor Hermes local mode to query Elpis capabilities after connecting
- [ ] Consider whether `clear_context()` should preserve retrieved memories

## Files Modified

| File | Lines Changed |
|------|---------------|
| `src/elpis/server.py` | +26 |
| `src/hermes/app.py` | +75, -8 |
| `src/hermes/cli.py` | +8, -4 |
| `src/psyche/cli.py` | +6, -4 |
| `src/psyche/handlers/psyche_client.py` | +39, -12 |
| `src/psyche/mcp/client.py` | +9 |
| `src/psyche/server/daemon.py` | +21 |
| `src/psyche/server/http.py` | +332, -99 |

## Testing

- All 561 existing tests pass
- Manual testing needed for remote mode tool execution
- Memory retrieval needs investigation
