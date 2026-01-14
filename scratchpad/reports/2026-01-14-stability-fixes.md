# Session Report: Stability Fixes - 2026-01-14

## Issues Addressed

### 1. StoreMemoryInput Validation Error (FIXED)
**Problem**: When Psyche tried to store a memory, the LLM would include `emotional_context` in the arguments (with extra fields like `quadrant`, `salience`), but the `StoreMemoryInput` Pydantic model rejected these extra fields.

**Error**:
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for StoreMemoryInput
emotional_context
  Extra inputs are not permitted
```

**Fix**: Added optional `emotional_context` field to `StoreMemoryInput` in `src/psyche/tools/tool_definitions.py`. The implementation already auto-fetches emotional state, so this field is accepted but ignored.

### 2. Connection Drop During Idle Thinking (DIAGNOSED + LOGGING)
**Problem**: After some time running, the Elpis server would silently die, causing "Connection closed" errors during idle thinking. Subsequent idle thoughts would fail with empty error messages.

**Analysis**:
- The "Connection closed" error comes from the MCP library when the read stream closes
- This happens when the Elpis server process exits
- The Elpis logs showed no errors - server just stops logging
- Most likely cause: either the MCP server exits when stdin closes, or llama-cpp crashes silently

**Fixes**:
1. **Better error logging** in `_generate_idle_thought()`:
   - Now logs exception type when message is empty
   - Detects "Connection closed" errors specifically
   - Sets `_connection_lost` flag for future reconnection support

2. **Signal handling** in Elpis server:
   - Added SIGTERM handler to log when signal is received
   - Added logging when server exits normally
   - Added logging in `run_server()` for MCP lifecycle events

3. **Diagnostic logging** in `run_server()`:
   - Logs when stdio streams connect
   - Logs when MCP server.run() completes
   - Logs on shutdown

## Files Modified

1. **`src/psyche/tools/tool_definitions.py`**
   - Added `emotional_context: Optional[Dict[str, Any]]` field to `StoreMemoryInput`

2. **`src/psyche/memory/server.py`**
   - Enhanced error logging in `_generate_idle_thought()` (lines 854-861)
   - Added `_connection_lost` attribute initialization (line 155)

3. **`src/elpis/server.py`**
   - Added signal import (line 6)
   - Added SIGTERM signal handler in `main()` (lines 636-643)
   - Enhanced `run_server()` with try/except/finally and logging (lines 617-631)
   - Added "Server exited normally" log in `main()` (line 648)

## Root Cause Analysis (Connection Drop)

The connection drop issue is likely caused by one of:

1. **MCP stdio server exit**: The MCP library's `stdio_server()` exits when stdin is closed. This could happen if:
   - The parent process (Psyche) closes the pipe
   - An error in Psyche causes the pipe to be garbage collected

2. **llama-cpp crash**: A native library crash would bypass Python's exception handling entirely. Common causes:
   - Out of GPU memory
   - Invalid model state
   - Buffer overflow in C code

3. **Process timeout**: The subprocess might be getting killed by the OS for resource reasons.

## Testing

- All 33 tool definition tests pass
- All 19 memory tools tests pass
- Manual verification of imports succeeds

## Next Steps

1. **Monitor logs**: With the new diagnostic logging, the next time the connection drops we'll have more information
2. **Consider reconnection**: Implement automatic reconnection when `_connection_lost` is set
3. **Check GPU memory**: If the issue persists, add GPU memory monitoring to Elpis
4. **Consider watchdog**: Add a health-check mechanism to detect and recover from server death
