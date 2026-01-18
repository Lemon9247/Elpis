# Plan: Fix Elpis Streaming Stability

## Problem

Elpis crashes with SIGSEGV during streaming inference because:
1. `llama_context` is NOT thread-safe (confirmed by llama.cpp maintainers)
2. Model loads on main thread, but streaming runs in a worker thread
3. CUDA contexts are thread-local - accessing from wrong thread causes crashes

## Root Cause

```python
# Current broken implementation in inference.py
async def _stream_in_thread(...):
    thread = threading.Thread(target=producer)  # <-- PROBLEM: new thread
    thread.start()
    # Model was loaded on main thread, but producer() runs on worker thread
```

## Solution Options

### Option A: Remove Threading (Recommended - Simpler)

**Insight**: Elpis is ALREADY a subprocess of Psyche. The threading inside Elpis is unnecessary.

**Change**: Iterate sync generator directly with cooperative yielding:

```python
async def chat_completion_stream(...) -> AsyncIterator[str]:
    gen = self._chat_completion_stream_sync(...)
    for token in gen:
        yield token
        await asyncio.sleep(0)  # Yield to event loop
```

**Pros**: Simple, minimal changes, no IPC complexity
**Cons**: Token generation (~50-500ms) blocks event loop between yields

### Option B: True Multiprocessing (More Complex)

**Change**: Move model loading AND inference to a dedicated worker process:

```python
# Worker process owns model, communicates via Queue
def inference_worker(request_queue, response_queue):
    model = load_model()  # Loaded in worker process
    while True:
        request = request_queue.get()
        for token in model.stream(request):
            response_queue.put(token)
```

**Pros**: Complete isolation, handles any edge cases
**Cons**: Complex IPC, serialization overhead, more code to maintain

## Recommended Approach: Option A

Option A is recommended because:
1. The MCP polling mechanism already handles async communication
2. Token generation latency (50-500ms) is acceptable for a TUI
3. Much simpler to implement and maintain
4. Can escalate to Option B if needed later

## Implementation Plan

### Phase 1: Update llama-cpp backend

**File**: `src/elpis/llm/backends/llama_cpp/inference.py`

1. Remove `_stream_in_thread()` method entirely
2. Modify `chat_completion_stream()`:

```python
async def chat_completion_stream(
    self,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    emotion_coefficients: Optional[Dict[str, float]] = None,
) -> AsyncIterator[str]:
    """Stream tokens without threading - all ops on main thread."""
    if emotion_coefficients:
        logger.debug("Emotion coefficients ignored by llama-cpp backend")

    try:
        for token in self._chat_completion_stream_sync(
            messages, max_tokens, temperature, top_p
        ):
            yield token
            await asyncio.sleep(0)  # Cooperative yield to event loop
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        raise
```

3. Remove unused imports: `threading`, `queue`

### Phase 2: Update transformers backend

**File**: `src/elpis/llm/backends/transformers/inference.py`

Same pattern - remove threading, iterate sync generator directly.

Note: Transformers uses `TextIteratorStreamer` with a thread. May need to:
- Fall back to non-streaming for transformers, OR
- Test if PyTorch handles threading differently

### Phase 3: Clean up environment hacks

**File**: `src/elpis/cli.py`

Remove workarounds that are no longer needed:
- `GGML_CUDA_DISABLE_GRAPHS=1` (was for threading issues)
- Keep `OMP_NUM_THREADS=1` (still useful for CPU ops)

### Phase 4: Test

1. Basic streaming test
2. Idle thinking test (the scenario that was crashing)
3. GPU offloading test
4. Long generation test

## Files to Modify

| File | Change |
|------|--------|
| `src/elpis/llm/backends/llama_cpp/inference.py` | Remove threading, direct iteration |
| `src/elpis/llm/backends/transformers/inference.py` | Same pattern (if compatible) |
| `src/elpis/cli.py` | Remove unnecessary env vars |
| `scratchpad/reports/2026-01-14-stability-fixes.md` | Update with final fix |

## No Changes Needed

- `src/elpis/server.py` - MCP polling already works
- `src/psyche/mcp/client.py` - Client already handles polling

## Risks

| Risk | Mitigation |
|------|------------|
| Event loop blocking during token gen | Acceptable for TUI; polling handles latency |
| Transformers incompatibility | Fall back to non-streaming |
| Edge cases in generator | Proper exception handling |

## Rollback

If Option A proves insufficient, escalate to Option B (multiprocessing).
