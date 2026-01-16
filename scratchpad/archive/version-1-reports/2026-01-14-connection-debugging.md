# Connection Debugging Session - 2026-01-14

## Problem
Psyche intermittently fails to connect to Elpis with error:
```
RuntimeError: generator didn't yield
```

And when connections do work, interrupting idle thinking causes:
```
RuntimeError: dictionary keys changed during iteration
```

## Root Causes Found

### 1. MCP Library Bug (FIXED)
**File**: `venv/.../mcp/shared/session.py:448`
```python
for id, stream in self._response_streams.items():  # BUG: iterates without copy
```

**Fix**: Created `src/psyche/mcp_patch.py` with `SafeIterDict` class that returns copies from `items()`, `keys()`, `values()`. Applied via patching `BaseSession.__init__`.

### 2. TUI Callback Race Condition (FIXED)
**File**: `src/psyche/client/app.py`

Callbacks (`_on_token`, `_on_thought`, etc.) were directly modifying Textual widgets from the inference loop, causing race conditions.

**Fix**: Wrapped all widget updates in `self.call_later(update)` to schedule safely on Textual's event loop.

### 3. Missing Timeouts (FIXED)
**File**: `src/psyche/memory/server.py`

- `_generate_idle_thought()` had no timeout on `client.generate()` call
- `_process_user_input()` had no timeout on streaming generation

**Fix**: Added `generation_timeout` config (default 120s) and wrapped calls with `asyncio.wait_for()` / `asyncio.timeout()`.

### 4. Connection Validation (FIXED)
Added `is_connected` checks before MCP calls in:
- `_generate_idle_thought()`
- `_process_user_input()`

## Current Issue: Intermittent "generator didn't yield"

**Observation**: Connection works perfectly in isolation:
```python
async with client.connect() as c:
    result = await c.get_emotion()  # Works!
```

But fails intermittently when run inside Textual app.

**Hypothesis**: Textual's event loop or signal handling may be interfering with subprocess spawning.

**Log pattern**:
```
Starting memory server...
# Sometimes succeeds:
Connected to Elpis inference server
# Sometimes fails:
Server connection error: generator didn't yield
Memory server stopped
```

**Debug logging added** to `ElpisClient.connect()` to trace exactly where failure occurs.

## Files Modified This Session

1. `src/psyche/mcp_patch.py` - NEW: MCP library monkey-patch
2. `src/psyche/cli.py` - Apply patch before imports
3. `src/psyche/client/app.py` - Safe widget updates, better error handling
4. `src/psyche/memory/server.py` - Timeouts, connection checks, diagnostics
5. `src/psyche/mcp/client.py` - Debug logging for connection
6. `tests/psyche/integration/test_memory_server.py` - Added `is_connected` to mock

## Commits This Session

1. `53bcf62` - Add timeout and connection guards
2. `3c646d9` - Improve ExceptionGroup unwrapping
3. `e21850a` - Fix TUI callback race condition
4. `a2f5cd7` - Add traceback logging for dictionary errors
5. `e27ffc0` - Add MCP library patch (broken version)
6. `10ec9a4` - Fix MCP patch to use SafeIterDict
7. `adf5ab8` - Add debug logging to ElpisClient.connect()

## Next Steps to Investigate

1. **Check Textual event loop interaction**: The subprocess spawning via `stdio_client` may conflict with Textual's event loop setup

2. **Check if `on_mount` timing matters**: Server task is started in `on_mount` - widgets may not be ready

3. **Try delaying server start**: Add a small delay after mount before starting server

4. **Check anyio backend**: Textual uses trio-compatible async, MCP uses anyio - verify compatibility

5. **Check signal handlers**: Textual may install signal handlers that interfere with subprocess

## Test Command
```bash
source venv/bin/activate && python -c "
import asyncio
from psyche.mcp_patch import apply_mcp_patch
apply_mcp_patch()
from psyche.mcp.client import ElpisClient

async def test():
    client = ElpisClient(server_command='elpis-server')
    async with client.connect() as c:
        result = await c.get_emotion()
        print(f'Success: {result}')

asyncio.run(test())
"
```
This works reliably - issue is specific to Textual integration.
