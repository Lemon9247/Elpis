# Psyche Agent Bug Investigation Report

**Date**: 2026-01-14
**Agent**: Psyche Agent
**Scope**: `src/psyche/` - TUI client, MCP clients, memory server, tools

## Summary

After thorough investigation of the Psyche codebase, I identified several bugs and potential issues ranging from critical async task handling problems to medium-severity resource management concerns. The most critical issues relate to exception handling in async task groups and missing connection state validation.

---

## Critical Issues

### 1. Server Task Exception Silently Ignored

**File**: `/home/lemoneater/Projects/Elpis/src/psyche/client/app.py`
**Lines**: 87-93

```python
async def _run_server(self) -> None:
    """Run the memory server as a background task."""
    try:
        await self.memory_server.start()
    except Exception as e:
        chat = self.query_one("#chat", ChatView)
        chat.add_system_message(f"Server error: {e}")
```

**Issue**: When `_run_server` is launched as a background task (line 82), any exception is caught and displayed but the task silently terminates. The `_server_task` is never checked for completion elsewhere, meaning:
- The app continues running with a dead server
- No automatic reconnection attempt
- User may not notice the server died if error scrolls off screen

**Severity**: Critical

**Suggested Fix**: Add a watcher for server task completion, implement reconnection logic, or terminate app on server failure.

---

### 2. Unchecked Connection State Before MCP Operations

**File**: `/home/lemoneater/Projects/Elpis/src/psyche/client/app.py`
**Lines**: 125-143

```python
async def _update_emotional_display(self) -> None:
    """Periodically update emotional state display."""
    try:
        emotion = await self.memory_server.client.get_emotion()
        # ...
    except Exception:
        pass  # Ignore errors during status updates
```

**Issue**: The periodic update calls `self.memory_server.client.get_emotion()` without checking if the client is connected. While the exception is caught, this creates silent failures. If the connection drops, this will silently fail every second until app closure.

**Severity**: High

**Suggested Fix**: Check `client.is_connected` before making calls, or implement a backoff strategy.

---

### 3. Task Group in Inference Loop Missing Proper Cancellation Handling

**File**: `/home/lemoneater/Projects/Elpis/src/psyche/memory/server.py`
**Lines**: 343-406

```python
while self._running:
    try:
        if idle_task is not None and not idle_task.done():
            # ...
            done, pending = await asyncio.wait(
                [input_task, idle_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # ...
    except asyncio.CancelledError:
        if idle_task:
            idle_task.cancel()
            try:
                await idle_task
            except asyncio.CancelledError:
                pass
        break
```

**Issue**: When processing the `done` set, if `idle_task` raised an exception, calling `idle_task.result()` will re-raise it. The code catches this (lines 374-377), but if the exception is not `asyncio.CancelledError`, it only logs but doesn't clean up properly. Additionally, if both tasks are in `done` (race condition), only one is processed.

**Severity**: High

**Suggested Fix**: Handle all tasks in `done` set, and ensure exceptions from `idle_task` don't leave dangling resources.

---

### 4. Mnemosyne Connection Not Released on Partial Failure

**File**: `/home/lemoneater/Projects/Elpis/src/psyche/memory/server.py`
**Lines**: 293-316

```python
async def start(self) -> None:
    # ...
    async with self.client.connect():
        # ...
        if self.mnemosyne_client and self.config.enable_consolidation:
            async with self.mnemosyne_client.connect():
                # ...
                await self._inference_loop()
        else:
            # ...
            await self._inference_loop()
```

**Issue**: The Mnemosyne connection is nested inside the Elpis connection context. If `_inference_loop()` raises during the Mnemosyne branch, the exception propagates through both context managers. However, if an exception occurs BEFORE entering `mnemosyne_client.connect()` but AFTER `client.connect()`, and the code logic is modified, there's potential for connection leak. Current code seems safe but fragile to refactoring.

**Severity**: Medium

---

## High Severity Issues

### 5. Memory Tools Use Client Without Connection Check

**File**: `/home/lemoneater/Projects/Elpis/src/psyche/tools/implementations/memory_tools.py`
**Lines**: 43-76

```python
async def recall_memory(self, query: str, n_results: int = 5) -> Dict[str, Any]:
    try:
        memories = await self.client.search_memories(
            query=query,
            n_results=n_results,
        )
        # ...
    except Exception as e:
        # ...
```

**Issue**: `self.client.search_memories()` is called without checking `self.client.is_connected`. The exception is caught, but the error message will be generic and won't indicate it's a connection issue.

**Severity**: High

**Suggested Fix**: Check `is_connected` first and return descriptive error if disconnected.

---

### 6. Potential Race Condition in Staged Messages

**File**: `/home/lemoneater/Projects/Elpis/src/psyche/memory/server.py`
**Lines**: 1131-1157

```python
async def _handle_compaction_result(self, result: CompactionResult) -> None:
    # Store previously staged messages to Mnemosyne
    if self._staged_messages and self.mnemosyne_client:
        # ...
        success = await self._store_messages_to_mnemosyne(self._staged_messages)
        if success:
            self._staged_messages = []  # Clear on success
        # ...
    # Stage newly dropped messages
    if result.dropped_messages:
        self._staged_messages.extend(result.dropped_messages)
```

**Issue**: If `_handle_compaction_result` is called while another call is still processing (unlikely but possible if compaction is triggered from multiple paths), there could be a race condition where `_staged_messages` is modified concurrently. Since Python's GIL protects list operations and this is single-threaded async, this is low risk but should have explicit protection.

**Severity**: Medium

---

### 7. Callback Invocation on Non-Main Thread

**File**: `/home/lemoneater/Projects/Elpis/src/psyche/client/app.py`
**Lines**: 95-123

```python
def _on_token(self, token: str) -> None:
    """Handle streaming token callback."""
    chat = self.query_one("#chat", ChatView)
    # ...

def _on_thought(self, thought: ThoughtEvent) -> None:
    """Handle internal thought callback."""
    thoughts = self.query_one("#thoughts", ThoughtPanel)
    # ...
```

**Issue**: These callbacks directly modify Textual widgets. If the callbacks are invoked from an async context that's not on the main event loop thread, this could cause issues. While the current code structure keeps everything on the same event loop, this pattern is fragile.

**Severity**: Medium

**Suggested Fix**: Use `self.call_from_thread()` or `self.post_message()` for thread-safe widget updates.

---

## Medium Severity Issues

### 8. Shutdown Consolidation Can Fail Silently

**File**: `/home/lemoneater/Projects/Elpis/src/psyche/memory/server.py`
**Lines**: 1159-1216

```python
async def shutdown_with_consolidation(self) -> None:
    # ...
    try:
        # ...
    except Exception as e:
        logger.error(f"Shutdown consolidation failed: {e}")
```

**Issue**: All exceptions during shutdown consolidation are logged but swallowed. This is intentional (shutdown should proceed), but if the Mnemosyne client disconnected mid-operation, staged messages are lost permanently.

**Severity**: Medium

---

### 9. File Read Opens File Twice

**File**: `/home/lemoneater/Projects/Elpis/src/psyche/tools/implementations/file_tools.py`
**Lines**: 109-117

```python
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    lines = []
    for i, line in enumerate(f):
        if i >= max_lines:
            break
        lines.append(line)

content = ''.join(lines)
total_lines_in_file = sum(1 for _ in open(path, 'r', encoding='utf-8', errors='replace'))
```

**Issue**: The file is opened twice - once to read content and once to count total lines. This is inefficient and could cause issues if the file is modified between reads.

**Severity**: Low (performance/correctness)

**Suggested Fix**: Count lines during the first read or use a single pass.

---

### 10. Missing `await` Candidates (False Positives Confirmed)

After checking all function calls, no obvious missing `await` keywords were found. The codebase properly uses `await` for all async operations.

---

### 11. Type Annotation Issue

**File**: `/home/lemoneater/Projects/Elpis/src/psyche/tools/implementations/memory_tools.py`
**Line**: 16

```python
get_emotion_fn: Optional[callable] = None,
```

**Issue**: `callable` should be `Callable` (from typing module). While this works at runtime, it's incorrect typing.

**Severity**: Low

---

## Low Severity Issues

### 12. Hardcoded Stream Poll Interval

**File**: `/home/lemoneater/Projects/Elpis/src/psyche/mcp/client.py`
**Lines**: 175-235

The `poll_interval` defaults to 0.05 seconds. Under high load, this creates significant polling overhead. No exponential backoff is implemented.

**Severity**: Low

---

### 13. Tool Result Dict Access Without Default

**File**: `/home/lemoneater/Projects/Elpis/src/psyche/memory/server.py`
**Lines**: 589-593

```python
if result.get("success", True):
    await self.client.update_emotion("success", intensity=0.3)
else:
    await self.client.update_emotion("failure", intensity=0.5)
```

**Issue**: If `result` is somehow not a dict (shouldn't happen, but defensive coding), this would raise AttributeError.

**Severity**: Low

---

### 14. Dangerous Commands List Incomplete

**File**: `/home/lemoneater/Projects/Elpis/src/psyche/tools/implementations/bash_tool.py`
**Lines**: 20-32

The `DANGEROUS_PATTERNS` list blocks some dangerous commands but misses others like:
- `> /etc/passwd`
- `chmod 777`
- `chown`
- `sudo`

**Severity**: Low (security enhancement opportunity)

---

## Resource Cleanup Issues

### 15. StreamingMessage Buffer Not Cleared on Error

**File**: `/home/lemoneater/Projects/Elpis/src/psyche/client/widgets/chat_view.py`

If streaming starts but the server crashes mid-stream, `end_stream()` may never be called, leaving the `_buffer` with partial content. The next stream would need to call `start()` which does clear it, so this is self-healing but could show stale content briefly.

**Severity**: Low

---

## Summary Table

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | Critical | app.py:87-93 | Server task exception silently ignored |
| 2 | High | app.py:125-143 | Unchecked connection state |
| 3 | High | server.py:343-406 | Task group cancellation handling |
| 4 | Medium | server.py:293-316 | Mnemosyne connection handling fragile |
| 5 | High | memory_tools.py:43-76 | Client connection not checked |
| 6 | Medium | server.py:1131-1157 | Potential staged messages race |
| 7 | Medium | app.py:95-123 | Callbacks modify widgets directly |
| 8 | Medium | server.py:1159-1216 | Shutdown consolidation fails silently |
| 9 | Low | file_tools.py:109-117 | File opened twice |
| 10 | - | - | No missing awaits found |
| 11 | Low | memory_tools.py:16 | Type annotation error |
| 12 | Low | client.py:175-235 | Hardcoded poll interval |
| 13 | Low | server.py:589-593 | Dict access without defensive check |
| 14 | Low | bash_tool.py:20-32 | Incomplete dangerous commands list |
| 15 | Low | chat_view.py | Streaming buffer not cleared on error |

---

## Recommendations

1. **Implement server health monitoring** - Add periodic health checks and automatic reconnection for the MCP connections.

2. **Add connection guards** - Wrap all client calls with `is_connected` checks or use a decorator pattern.

3. **Use thread-safe widget updates** - Replace direct widget modifications in callbacks with Textual's message posting system.

4. **Add graceful degradation** - When Mnemosyne is unavailable, continue with reduced functionality instead of failing.

5. **Improve error visibility** - Critical errors should be more prominently displayed to users, not just logged.

---

*Report generated by Psyche Agent*
