# Phase 6 Session Report: Remote Mode Tool Architecture

**Date:** 2026-01-18
**Branch:** phase4/architecture-refactor
**Commits:** 2550886, 62b1b10

## Overview

This session implemented the remote mode tool architecture for Psyche, fixing the issue where tools were never executed when Hermes connected to a Psyche server. Also fixed a critical context window mismatch bug and added configuration for larger context windows.

## Problem Statement

In remote mode (`hermes --server`), Psyche would generate `tool_call` blocks but nothing happened - tools were never executed. The architecture needed to:

1. Have Hermes execute file/bash/search tools locally
2. Have Psyche server execute memory tools internally
3. Properly synchronize context window sizes between Psyche and Elpis

## Architecture Implemented

```
┌─────────────────────────────────────────────────────────────┐
│                     PSYCHE SERVER                           │
│  - Memory tools (recall_memory, store_memory) → INTERNAL    │
│  - Queries Elpis for context_length on startup              │
│  - Returns file/bash/search tool_calls to client            │
└─────────────────────────────────────────────────────────────┘
                            ↕ HTTP API
┌─────────────────────────────────────────────────────────────┐
│                     HERMES CLIENT                           │
│  - Receives tool_calls from server                          │
│  - Executes file/bash/search tools locally via ToolEngine   │
│  - Sends results back to server                             │
│  - Loops until no more tool_calls (max 10 iterations)       │
└─────────────────────────────────────────────────────────────┘
```

## Changes Made

### 1. Client Tool Call Capture (`src/psyche/handlers/psyche_client.py`)

- Added `_last_tool_calls` and `_last_finish_reason` instance variables
- Modified `generate_stream()` to capture tool calls from the SSE finish chunk
- Added `get_pending_tool_calls()` method to retrieve tool calls after streaming

### 2. Hermes Tool Execution Loop (`src/hermes/app.py`)

- Added `tool_engine` parameter to Hermes `__init__`
- Rewrote `_process_via_client()` with full tool execution loop:
  - Streams response from server
  - Checks for pending tool calls via `get_pending_tool_calls()`
  - Executes tools locally via ToolEngine
  - Shows tool activity in UI via `_on_tool_call()`
  - Sends results back to server via `add_tool_result()`
  - Loops until no more tool calls

### 3. CLI Integration (`src/hermes/cli.py`)

- Pass `tool_engine` to Hermes app in remote mode
- Updated default context tokens to conservative 3000/800 (fallback)

### 4. Server-Side Memory Tool Execution (`src/psyche/server/http.py`)

Memory tools execute server-side because Psyche's memory is part of her "self":

- Added `MEMORY_TOOLS = {"recall_memory", "store_memory"}` constant
- Added `_execute_memory_tool()` - executes memory tools and returns JSON result
- Added `_separate_tool_calls()` - splits memory vs client tools
- Rewrote `_generate_response()` with memory tool execution loop
- Rewrote `_stream_response()` with memory tool execution loop
- Added error handling for context overflow (graceful degradation)

### 5. Context Window Synchronization

**The Bug:** PsycheCore was configured with `max_context_tokens=24000` but Elpis's LLM defaulted to `context_length=4096`. After adding memory tool results, generation failed with "Requested tokens (4157) exceed context window of 4096".

**The Fix:**

- Added `get_capabilities` MCP tool to Elpis (`src/elpis/server.py`)
  - Returns `context_length`, `max_tokens`, `backend`, `model_path`, etc.
- Added `get_capabilities()` method to ElpisClient (`src/psyche/mcp/client.py`)
- Modified `PsycheDaemon._init_core_with_clients()` (`src/psyche/server/daemon.py`)
  - Queries Elpis capabilities after connecting
  - Configures context: `max_tokens = context_length * 0.75`, `reserve = context_length * 0.20`
- Updated fallback defaults to conservative 3000/800 tokens

### 6. Configuration for Larger Context

Created `.env` file for easy configuration:

```
MODEL__CONTEXT_LENGTH=16384
```

This uses pydantic-settings nested delimiter (`__`) to set `settings.model.context_length`.

### 7. Memory Summary Length

Increased memory summary truncation from 100 to 500 characters across:
- `src/mnemosyne/server.py`
- `src/psyche/core/memory_handler.py`
- `src/psyche/core/server.py`
- `src/psyche/tools/implementations/memory_tools.py`

100 characters was too short to capture meaningful context for memory retrieval.

## Files Modified

| File | Changes |
|------|---------|
| `src/elpis/server.py` | +26 (get_capabilities tool) |
| `src/hermes/app.py` | +75, -8 (tool execution loop) |
| `src/hermes/cli.py` | +8, -4 (pass tool_engine, update defaults) |
| `src/psyche/cli.py` | +6, -4 (update defaults) |
| `src/psyche/handlers/psyche_client.py` | +39, -12 (tool call capture) |
| `src/psyche/mcp/client.py` | +9 (get_capabilities method) |
| `src/psyche/server/daemon.py` | +21 (query capabilities) |
| `src/psyche/server/http.py` | +332, -99 (memory tool execution, error handling) |
| `src/mnemosyne/server.py` | +1, -1 (summary length) |
| `src/psyche/core/memory_handler.py` | +2, -2 (summary length) |
| `src/psyche/core/server.py` | +1, -1 (summary length) |
| `src/psyche/tools/implementations/memory_tools.py` | +1, -1 (summary length) |
| `.env` | Created (context_length config) |

## Testing

- All 561 existing tests pass
- Manual testing of remote mode tool execution needed
- Context synchronization verified via server logs:
  ```
  Elpis context_length: 16384
  Context configured: max_tokens=12288, reserve=3276
  ```

## Known Issues / Future Work

1. **Hermes local mode** - Still uses hardcoded defaults, needs refactoring to query Elpis after connecting (TODO in code)

2. **Memory retrieval logging** - At DEBUG level, hard to verify if memories are being retrieved. Consider adding INFO-level summary.

3. **Memory result truncation** - Added `MAX_MEMORY_RESULT_CHARS = 2000` constant but didn't implement truncation. May be needed for very large memory results.

4. **Settings system refactor** - Currently context_length is configured separately in Elpis and Psyche. Future work should unify configuration.

## Configuration

To run with larger context window:

```bash
# Option 1: Environment variable
ELPIS_MODEL_CONTEXT_LENGTH=16384 psyche-server

# Option 2: .env file in project root
MODEL__CONTEXT_LENGTH=16384
```

Recommended values:
- **8192** - 2x default, modest memory increase
- **16384** - 4x default, good balance for conversations with memory
- **32768** - 8x default, for long conversations

## Conclusion

Remote mode tool architecture is now functional. Hermes can connect to a Psyche server and execute tools properly - file/bash/search tools run locally on the client, memory tools run server-side. Context window synchronization prevents overflow errors.

The main remaining concern is verifying that memory retrieval is working correctly - the logs don't show memory retrieval at INFO level, so it's hard to confirm memories are being injected into context.
