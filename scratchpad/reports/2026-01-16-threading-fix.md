# Session Report: Streaming Threading Fix - 2026-01-16

## Summary

Implemented the streaming stability plan to fix SIGSEGV crashes during streaming inference. Removed unnecessary threading bridges that were causing race conditions with `llama_context` (which is not thread-safe).

## Changes Made

### 1. llama_cpp Backend

**File**: `src/elpis/llm/backends/llama_cpp/inference.py`

- Removed `_stream_in_thread()` method entirely
- Modified `chat_completion_stream()` to iterate sync generator directly with `await asyncio.sleep(0)` for cooperative yielding
- Removed unused imports: `threading`, `queue`
- Updated comments to reflect new threading model

**Before**: Model loaded on main thread, streaming ran in worker thread (thread-unsafe)
**After**: All operations run on main thread (thread-safe)

### 2. transformers Backend

**File**: `src/elpis/llm/backends/transformers/inference.py`

- Removed redundant outer `_stream_in_thread()` wrapper
- Modified `chat_completion_stream()` to iterate sync generator directly
- Kept internal threading in `_chat_completion_stream_sync()` (required by `TextIteratorStreamer`)
- Removed unused import: `queue`

**Before**: Double-threading (TextIteratorStreamer thread + outer producer thread)
**After**: Single internal thread (required by HuggingFace API, PyTorch handles this safely)

## Environment Variables (Kept as Safety Net)

The following env vars remain in `cli.py` and `inference.py`:
- `OMP_NUM_THREADS=1` - Useful for CPU operations
- `GGML_CUDA_DISABLE_GRAPHS=1` - Safety net until fix is proven stable

## Testing Plans (DEFERRED)

Testing will be performed on a separate machine with capable CPU + GPU. Test scenarios:

### Test 1: Basic Streaming
```bash
# Start elpis-server and send a streaming chat request
# Verify tokens are yielded correctly without crashes
```

### Test 2: Idle Thinking (Dream State)
```bash
# This was the scenario causing crashes
# Run psyche and let it enter idle thinking mode
# Monitor for SIGSEGV crashes during extended generation
```

### Test 3: GPU Offloading
```bash
# Verify hybrid CPU/GPU execution still works
# Check: 20 GPU layers + 12 CPU layers configuration
# Monitor nvidia-smi for VRAM usage
```

### Test 4: Long Generation
```bash
# Generate 1000+ tokens in a single response
# Verify no crashes or memory leaks
```

### Test 5: Rapid Streaming Requests
```bash
# Send multiple streaming requests in quick succession
# Verify proper cleanup between requests
```

## Diagnostic Commands

If crashes occur during testing:
```bash
# Check for core dumps
coredumpctl list

# Get crash details
coredumpctl info <pid>

# Monitor GPU
nvidia-smi -l 1

# Check system logs
journalctl --user -f
```

## Rollback Plan

If crashes persist:
1. Revert threading changes
2. Escalate to Option B: True multiprocessing with worker process owning model

## Files Modified

| File | Lines Changed |
|------|---------------|
| `src/elpis/llm/backends/llama_cpp/inference.py` | ~50 lines removed |
| `src/elpis/llm/backends/transformers/inference.py` | ~45 lines removed |

## Related Documents

- Plan: `scratchpad/plans/2026-01-14-streaming-stability-plan.md`
- Previous report: `scratchpad/reports/2026-01-14-stability-fixes.md`
- Implementation plan: `~/.claude/plans/golden-yawning-cerf.md`

## Commits

Changes ready for commit (not yet committed).
