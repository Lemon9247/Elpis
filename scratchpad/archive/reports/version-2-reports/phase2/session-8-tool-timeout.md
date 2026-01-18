# Session 8: C2.2 - Tool Execution Timeout

**Date:** 2026-01-16
**Phase:** Phase 2 - UX Improvements
**Task:** C2.2 - Tool Execution Timeout

## Summary

Implemented configurable timeout support for tool execution and enhanced interrupt handling in the ReAct loop to allow Ctrl+C to properly interrupt tool execution.

## Changes Made

### 1. Tool Engine Timeout Support (`src/psyche/tools/tool_engine.py`)

Added timeout support to the `execute_tool_call` method:

- **New setting**: Added `tool_timeout: float = 60.0` to `ToolSettings` dataclass (line 38)
  - This provides a configurable default timeout of 60 seconds for tool execution

- **Modified `execute_tool_call` method** (lines 265-340):
  - Added optional `timeout` parameter that overrides the default from settings
  - Wrapped tool handler execution in `asyncio.timeout()` context manager
  - Returns structured error result on timeout: `{"success": False, "error": f"Timed out after {timeout}s"}`
  - Logs warning when timeout occurs

### 2. ReAct Loop Interrupt Handling (`src/psyche/memory/server.py`)

Added interrupt event checks in the ReAct loop for cleaner interruption:

- **Check at loop iteration start** (lines 624-630):
  - Before each ReAct iteration, checks if `_interrupt_event` is set
  - If set, clears the event, logs the interruption, notifies callback with "[Interrupted]", and returns

- **Check before tool execution** (lines 706-712):
  - After parsing a tool call but before executing it, checks for interrupt
  - If interrupted, responds with the partial output plus "[Interrupted before tool execution]"
  - This prevents tools from starting if the user has requested cancellation

## Files Modified

1. `/home/lemoneater/Projects/Personal/Elpis/src/psyche/tools/tool_engine.py`
   - Added `tool_timeout` setting to `ToolSettings`
   - Enhanced `execute_tool_call` with timeout support via `asyncio.timeout()`

2. `/home/lemoneater/Projects/Personal/Elpis/src/psyche/memory/server.py`
   - Added interrupt check at the start of ReAct loop iterations
   - Added interrupt check before tool execution

## Test Results

All 205 tests pass:

```
tests/psyche/ - 205 passed in 2.45s
```

Key test files verified:
- `tests/psyche/unit/test_tool_engine.py` - 19 tests pass (includes existing tool execution tests)
- `tests/psyche/integration/test_memory_server.py` - All integration tests pass

## Technical Notes

### Timeout Implementation

The timeout uses Python 3.11+'s `asyncio.timeout()` context manager:

```python
try:
    async with asyncio.timeout(effective_timeout):
        result = await tool_def.handler(**validated_args.model_dump())
except asyncio.TimeoutError:
    # Return structured error
    return {
        "success": False,
        "result": {"success": False, "error": f"Timed out after {effective_timeout}s"},
        ...
    }
```

### Interrupt Flow

The interrupt mechanism works with Session 7's `_interrupt_event` asyncio.Event:

1. User presses Ctrl+C -> `interrupt()` method is called -> `_interrupt_event.set()`
2. ReAct loop checks `_interrupt_event.is_set()` at:
   - Start of each iteration (before generation)
   - Before tool execution (after generation, before tool runs)
3. If set, the event is cleared and the loop returns with appropriate response

### Configuration

The tool timeout can be configured via `ToolSettings`:

```python
settings = ToolSettings(tool_timeout=30.0)  # 30 second timeout
engine = ToolEngine(workspace_dir, settings)
```

Or overridden per-call:

```python
result = await engine.execute_tool_call(tool_call, timeout=10.0)
```

## Integration with Session 7

This session builds on Session 7's interrupt mechanism:
- Session 7 added `_interrupt_event` asyncio.Event to `MemoryServer`
- Session 7 added interrupt checking during streaming generation
- Session 8 extends this to cover the tool processing phase of the ReAct loop

## Issues Encountered

None - implementation was straightforward and all tests passed on first run.
