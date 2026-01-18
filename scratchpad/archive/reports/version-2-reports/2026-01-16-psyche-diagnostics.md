# Psyche Diagnostic Report - 2026-01-16

## Summary

Investigation of Psyche shutdown issues and memory recall failures based on logs in `~/.psyche` and `~/.elpis`.

## Issues Found and Fixes

### 1. Missing `recall_memory` Method (FIXED)

**Problem**: `src/psyche/memory/server.py:562` called `self.mnemosyne_client.recall_memory()` but `MnemosyneClient` only has `search_memories()`.

**Symptom**:
```
WARNING | Failed to retrieve memories: 'MnemosyneClient' object has no attribute 'recall_memory'
```

**Fix**: Changed `recall_memory` to `search_memories` at line 562.

### 2. Memory Store Failures During Shutdown (FIXED)

**Problem**: `_store_messages_to_mnemosyne()` called `self.client.get_emotion()` unconditionally, which fails when Elpis is disconnected during shutdown.

**Symptom**:
```
ERROR | Failed to store message to Mnemosyne:
```
(empty error message from connection failure)

**Fix**: Added connection check before calling `get_emotion()`:
```python
emotional_context = None
if self.client and self.client.is_connected:
    try:
        emotion = await self.client.get_emotion()
        emotional_context = {...}
    except Exception:
        pass  # Store without emotional context if unavailable
```

Also improved error logging to include exception type for better debugging.

### 3. MCP Session Shutdown Race Condition (UPSTREAM BUG)

**Problem**: `mcp/shared/session.py:448` iterates over `self._response_streams.items()` without copying, causing `RuntimeError: dictionary changed size during iteration` when another task modifies it during shutdown cleanup.

**Symptom** (in stderr.log):
```
RuntimeError: dictionary changed size during iteration
```

**Status**: This is an upstream bug in the MCP library (v1.25.0). The fix should be:
```python
# Current (buggy):
for id, stream in self._response_streams.items():

# Fixed:
for id, stream in list(self._response_streams.items()):
```

**Recommendation**: File issue with MCP library maintainers. Current workaround is that this error is non-fatal and cleanup proceeds via fallback paths.

### 4. Widget Access During Shutdown (ALREADY HANDLED)

**Problem**: When server fails to connect, error handler tries to access `#chat` widget which may be destroyed during shutdown.

**Status**: Current code at `app.py:127-131` properly handles this with try/except. The errors in stderr.log are from an older version of the file.

## Log Analysis

### Key Error Patterns

1. **Connection closed during generation**: Elpis server connections drop mid-operation, likely due to model timeouts or resource exhaustion
2. **Idle thought generation failures**: `ClosedResourceError` after connection loss
3. **Fallback storage working correctly**: When Mnemosyne is unavailable, messages are properly saved to `~/.psyche/fallback_memories/`

### Healthy Shutdown Sequence

The logs show the fallback mechanism working:
```
WARNING | Mnemosyne unavailable, saving 8 messages to local fallback
INFO    | Saved 8 messages to local fallback: /home/lemoneater/.psyche/fallback_memories/...
```

## Files Modified

- `src/psyche/memory/server.py`:
  - Line 562: `recall_memory` -> `search_memories`
  - Lines 1423-1448: Added Elpis connection check before `get_emotion()` call

## Testing

All unit tests pass after fixes:
```
tests/psyche/unit/test_memory_tools.py: 19 passed
```

## Recommendations

1. **Short-term**: Monitor for continued "Generation failed: Connection closed" errors - may indicate Elpis server stability issues
2. **Medium-term**: File upstream issue for MCP library race condition
3. **Long-term**: Consider implementing connection health checks and automatic reconnection in the MCP clients
