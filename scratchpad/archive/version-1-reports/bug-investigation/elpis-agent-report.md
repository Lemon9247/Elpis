# Elpis Inference Server Bug Investigation Report

**Date**: 2026-01-14
**Investigator**: Elpis Agent
**Scope**: `src/elpis/` - Main MCP server, LLM backends, emotional state management

---

## Executive Summary

This report documents critical bugs and potential issues found in the Elpis inference MCP server codebase. The investigation focused on async code patterns, exception handling, resource management, and potential silent failures.

**Critical Issues Found**: 2
**High Severity Issues Found**: 4
**Medium Severity Issues Found**: 5
**Low Severity Issues Found**: 3

---

## Critical Issues

### 1. Fire-and-Forget Task in Stream Producer (Task Group Error Risk)

**File**: `/home/lemoneater/Projects/Elpis/src/elpis/server.py`
**Lines**: 399-419
**Severity**: Critical

**Description**:
The `_handle_generate_stream_start` function creates a background task using `asyncio.create_task(stream_producer())` but does not store a reference to the task. This is a classic "fire-and-forget" anti-pattern that can cause:

1. **Unhandled exceptions being silently lost** - If the task raises an exception outside the try/except block (e.g., during creation of the async iterator), it will be logged as "Task exception was never retrieved"
2. **Task cancellation during shutdown** - The task may be cancelled without cleanup during server shutdown
3. **No way to cancel the task** - The stream cancel function cannot actually stop the background task

```python
# Line 419 - Task reference is not stored
asyncio.create_task(stream_producer())
```

**Suggested Fix**:
Store task references in `StreamState` and implement proper cancellation:
```python
stream_state.task = asyncio.create_task(stream_producer())
```

---

### 2. Race Condition in Stream State Access

**File**: `/home/lemoneater/Projects/Elpis/src/elpis/server.py`
**Lines**: 456-459 and 476-479
**Severity**: Critical

**Description**:
The stream read and cancel functions delete the stream from `active_streams` after accessing it, but the background producer task may still be running and writing to `stream_state.buffer`. This creates a race condition where:

1. Client calls `generate_stream_read` when stream is complete
2. Stream is deleted from `active_streams` on line 459
3. Background producer (still running) continues to append to `stream_state.buffer`
4. Producer finishes and tries to call `regulator.process_response()` on line 417

The reference to `stream_state` survives (due to closure), but any subsequent reads will fail with "Unknown stream_id" even though the producer hasn't finished.

```python
# Line 459 - Deletes stream while producer may still be writing
del ctx.active_streams[stream_id]
```

**Suggested Fix**:
Use a lock or wait for producer task completion before cleanup. Add task reference to StreamState and await it.

---

## High Severity Issues

### 3. Thread Join Timeout with No Error Handling

**File**: `/home/lemoneater/Projects/Elpis/src/elpis/llm/backends/llama_cpp/inference.py`
**Lines**: 285
**File**: `/home/lemoneater/Projects/Elpis/src/elpis/llm/backends/transformers/inference.py`
**Lines**: 310, 348

**Severity**: High

**Description**:
The streaming implementations use `thread.join(timeout=1.0)` but do not check if the thread is still alive after the timeout. If the thread doesn't finish within 1 second, it continues running in the background (daemon thread), potentially causing resource leaks and concurrent access issues.

```python
# Line 285 (llama_cpp/inference.py)
finally:
    thread.join(timeout=1.0)  # No check if thread actually terminated
```

**Suggested Fix**:
Check `thread.is_alive()` after join and log a warning if the thread didn't terminate:
```python
thread.join(timeout=1.0)
if thread.is_alive():
    logger.warning("Producer thread did not terminate within timeout")
```

---

### 4. Missing Await on Potentially Async Cleanup

**File**: `/home/lemoneater/Projects/Elpis/src/elpis/llm/backends/transformers/inference.py`
**Lines**: 443-446

**Severity**: High

**Description**:
The `__del__` method calls `self.steering.remove_hook()` during garbage collection. However:
1. `__del__` is not guaranteed to run
2. It runs in an unpredictable context where the event loop may not be available
3. If steering hook cleanup requires async operations in the future, this will fail silently

```python
def __del__(self):
    """Cleanup on deletion."""
    if hasattr(self, "steering"):
        self.steering.remove_hook()
```

**Suggested Fix**:
Implement an explicit async cleanup method and ensure it's called during server shutdown:
```python
async def cleanup(self):
    if hasattr(self, "steering"):
        self.steering.remove_hook()
```

---

### 5. Steering Hook Not Removed on Exception in Streaming

**File**: `/home/lemoneater/Projects/Elpis/src/elpis/llm/backends/transformers/inference.py`
**Lines**: 279-310

**Severity**: High

**Description**:
In `_chat_completion_stream_sync`, if an exception occurs in the `generate()` function after the hook is applied but before `model.generate()` starts (or if the thread is killed), the steering hook removal in the `finally` block may not execute if the thread is terminated externally.

Additionally, if the main thread raises an exception while iterating `streamer`, the hook will remain attached to the model, affecting subsequent requests.

```python
# Lines 287-302 - Hook cleanup is in nested thread's finally block
def generate():
    try:
        with torch.no_grad():
            self.model.generate(...)
    finally:
        if steering_vector is not None:
            self.steering.remove_hook()  # May never execute

thread = threading.Thread(target=generate, daemon=True)
thread.start()

for token in streamer:  # If exception here, hook remains
    yield token
```

**Suggested Fix**:
Add hook cleanup in the outer scope as well, using try/finally around the yield loop.

---

### 6. Type Mismatch in hardware_backend Validation

**File**: `/home/lemoneater/Projects/Elpis/src/elpis/config/settings.py`
**Lines**: 36-38
**File**: `/home/lemoneater/Projects/Elpis/src/elpis/llm/backends/llama_cpp/config.py`
**Line**: 66

**Severity**: High

**Description**:
`ModelSettings.hardware_backend` is typed as `str` but `LlamaCppConfig.hardware_backend` uses a `Literal["auto", "cuda", "rocm", "cpu"]` type. When `to_llama_cpp_config()` copies the value, Pydantic validation will fail if ModelSettings receives an invalid backend string from environment variables.

```python
# settings.py line 36-38
hardware_backend: str = Field(
    default="auto", description="Hardware backend: auto, cuda, rocm, cpu"
)

# LlamaCppConfig line 66
hardware_backend: Literal["auto", "cuda", "rocm", "cpu"] = Field(...)
```

The validation error will be cryptic and won't point to the actual configuration issue.

**Suggested Fix**:
Use consistent typing across all config classes:
```python
hardware_backend: Literal["auto", "cuda", "rocm", "cpu"] = Field(...)
```

---

## Medium Severity Issues

### 7. Silent Failure When Emotion Vectors Directory Missing

**File**: `/home/lemoneater/Projects/Elpis/src/elpis/llm/backends/transformers/steering.py`
**Lines**: 63-66

**Severity**: Medium

**Description**:
When the emotion vectors directory doesn't exist, a warning is logged but no exception is raised. The server continues to run without steering capability, which may not be the desired behavior if steering was explicitly configured.

```python
if not path.exists():
    logger.warning(f"Emotion vectors directory not found: {path}")
    return  # Silent failure - vectors dict remains empty
```

**Suggested Fix**:
Consider raising an exception or adding a `strict` mode that fails fast when configured directories are missing.

---

### 8. Potential KeyError in Stream Deletion During Concurrent Access

**File**: `/home/lemoneater/Projects/Elpis/src/elpis/server.py`
**Lines**: 459, 479

**Severity**: Medium

**Description**:
The `del ctx.active_streams[stream_id]` statements are not protected against concurrent access. If two coroutines simultaneously try to read/cancel the same stream, one will succeed and the other will raise a KeyError.

```python
del ctx.active_streams[stream_id]  # KeyError if already deleted
```

**Suggested Fix**:
Use `ctx.active_streams.pop(stream_id, None)` for safe deletion.

---

### 9. regulator.process_response() Called Without Error Checking

**File**: `/home/lemoneater/Projects/Elpis/src/elpis/server.py`
**Lines**: 308, 416-417

**Severity**: Medium

**Description**:
`ctx.regulator.process_response(content)` is called without checking if `content` is valid (non-empty string). While the implementation handles empty strings gracefully, this is an implicit contract that could break if the regulator's implementation changes.

```python
# Line 308
ctx.regulator.process_response(content)

# Lines 416-417
if stream_state.buffer:
    full_content = "".join(stream_state.buffer)
    regulator.process_response(full_content)  # buffer could contain only empty strings
```

**Suggested Fix**:
Add explicit validation:
```python
if content and content.strip():
    ctx.regulator.process_response(content)
```

---

### 10. torch.load with weights_only=True May Fail on Older Vectors

**File**: `/home/lemoneater/Projects/Elpis/src/elpis/llm/backends/transformers/steering.py`
**Lines**: 71-72

**Severity**: Medium

**Description**:
The use of `weights_only=True` is a security best practice but may fail when loading older steering vectors that were saved with pickle serialization containing non-tensor objects.

```python
vector = torch.load(
    vector_file, map_location=self.device, weights_only=True
)
```

**Suggested Fix**:
Add fallback with warning:
```python
try:
    vector = torch.load(vector_file, map_location=self.device, weights_only=True)
except Exception:
    logger.warning(f"Loading {vector_file} with weights_only=False (legacy format)")
    vector = torch.load(vector_file, map_location=self.device, weights_only=False)
```

---

### 11. Missing Type Annotation for Return Value

**File**: `/home/lemoneater/Projects/Elpis/src/elpis/llm/base.py`
**Lines**: 71-93

**Severity**: Medium

**Description**:
The `chat_completion_stream` abstract method has a return type annotation of `AsyncIterator[str]`, but the actual implementations use `async def` with `yield`. This is semantically correct but the `yield` makes the return type an `AsyncGenerator[str, None]`. Static type checkers may flag this inconsistency.

```python
@abstractmethod
async def chat_completion_stream(...) -> AsyncIterator[str]:
    pass
```

**Suggested Fix**:
Update type hints to use `AsyncGenerator[str, None]` or import from `collections.abc` for consistency.

---

## Low Severity Issues

### 12. Redundant try/except in Main Entry Points

**File**: `/home/lemoneater/Projects/Elpis/src/elpis/cli.py`
**Lines**: 9-18
**File**: `/home/lemoneater/Projects/Elpis/src/elpis/server.py`
**Lines**: 605-614

**Severity**: Low

**Description**:
Both `cli.py` and `server.py` have `main()` functions with identical exception handling. The `cli.py` version imports and calls the server functions, creating redundant exception handling.

**Suggested Fix**:
Remove the duplicate `main()` function from `server.py` or consolidate entry points.

---

### 13. Hardcoded Polling Interval in Async Streaming

**File**: `/home/lemoneater/Projects/Elpis/src/elpis/llm/backends/llama_cpp/inference.py`
**Line**: 278
**File**: `/home/lemoneater/Projects/Elpis/src/elpis/llm/backends/transformers/inference.py`
**Line**: 341

**Severity**: Low

**Description**:
The `await asyncio.sleep(0.01)` polling interval is hardcoded. This may be too fast (wasting CPU) or too slow (adding latency) depending on the model and hardware.

```python
while token_queue.empty():
    await asyncio.sleep(0.01)  # Hardcoded 10ms polling
```

**Suggested Fix**:
Make the polling interval configurable or use `asyncio.Event` for proper signaling.

---

### 14. Unused Import in Type Checking Block

**File**: `/home/lemoneater/Projects/Elpis/src/elpis/llm/backends/transformers/inference.py`
**Lines**: 29-31

**Severity**: Low

**Description**:
The `if TYPE_CHECKING:` block re-imports `torch` which is already imported in the runtime block. This is redundant.

```python
if TYPE_CHECKING:
    import torch  # Already imported above
```

**Suggested Fix**:
Remove the redundant import in the TYPE_CHECKING block.

---

## Summary Table

| # | Severity | File | Line(s) | Issue Summary |
|---|----------|------|---------|---------------|
| 1 | Critical | server.py | 419 | Fire-and-forget asyncio task |
| 2 | Critical | server.py | 456-479 | Race condition in stream cleanup |
| 3 | High | llama_cpp/inference.py, transformers/inference.py | 285, 310, 348 | Thread join timeout without alive check |
| 4 | High | transformers/inference.py | 443-446 | __del__ cleanup unreliable |
| 5 | High | transformers/inference.py | 279-310 | Steering hook leak on exception |
| 6 | High | settings.py, llama_cpp/config.py | 36-38, 66 | Type mismatch in hardware_backend |
| 7 | Medium | steering.py | 63-66 | Silent failure on missing vectors dir |
| 8 | Medium | server.py | 459, 479 | Potential KeyError on concurrent access |
| 9 | Medium | server.py | 308, 416-417 | process_response without validation |
| 10 | Medium | steering.py | 71-72 | torch.load weights_only compatibility |
| 11 | Medium | base.py | 71-93 | AsyncIterator vs AsyncGenerator typing |
| 12 | Low | cli.py, server.py | 9-18, 605-614 | Redundant exception handling |
| 13 | Low | llama_cpp/inference.py, transformers/inference.py | 278, 341 | Hardcoded polling interval |
| 14 | Low | transformers/inference.py | 29-31 | Unused TYPE_CHECKING import |

---

## Recommendations

1. **Immediate Priority**: Fix the critical fire-and-forget task and race condition issues in streaming (Issues #1, #2)
2. **Short Term**: Address high severity issues related to thread management and cleanup (#3, #4, #5)
3. **Medium Term**: Improve type consistency and error handling (#6, #7, #8, #9)
4. **Long Term**: Refactor streaming to use proper async patterns (e.g., `asyncio.Event`, task groups with structured concurrency)

---

*End of Report*
