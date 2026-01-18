# Session Report: Stability Fixes - 2026-01-14

## Summary

Fixed multiple stability issues causing Psyche/Elpis crashes during idle thinking (dream state). Root cause was SIGSEGV in llama.cpp's multi-threaded CPU code, compounded by missing GPU offloading.

## Issues Found & Fixed

### 1. StoreMemoryInput Validation Error
**Problem**: LLM included `emotional_context` with extra fields when storing memories, but Pydantic rejected them.

**Fix**: Added optional `emotional_context` field to `StoreMemoryInput` in `src/psyche/tools/tool_definitions.py`.

### 2. SIGSEGV in ggml Multi-threading (ROOT CAUSE)
**Problem**: Server crashed with SIGSEGV in `ggml_compute_forward_mul_mat` during idle thinking. Coredump analysis revealed race condition in llama.cpp's OpenMP worker threads.

```
Signal: 11 (SEGV)
#1  0x00007fb168ecce59 ggml_compute_forward_mul_mat (libggml-cpu.so)
#3  0x00007fb16abcd737 gomp_thread_start (libgomp.so.1)
```

**Fix**: Multiple steps required:
1. Set `n_threads=1` in llama-cpp config
2. Set environment variables to disable OpenMP/BLAS multi-threading:
   ```python
   os.environ.setdefault("OMP_NUM_THREADS", "1")
   os.environ.setdefault("MKL_NUM_THREADS", "1")
   os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
   ```
3. **Disable CUDA graph optimization** (critical for streaming):
   ```python
   os.environ.setdefault("GGML_CUDA_DISABLE_GRAPHS", "1")
   ```

The environment variables are critical because:
- `n_threads=1` alone doesn't fully disable OpenMP
- **`llama_context` is NOT thread-safe** - streaming uses a separate Python thread
- CUDA graph optimization causes race conditions when the context is accessed from multiple threads
- See: https://github.com/ggml-org/llama.cpp/issues/11804

### 3. No GPU Offloading
**Problem**: Despite `gpu_layers=35` config, `llama_supports_gpu_offload()` returned `False`. All 32 layers were running on CPU because llama-cpp-python was built without CUDA support.

**Fix**: Rebuilt llama-cpp-python with CUDA:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 4. VRAM Out of Memory
**Problem**: With CUDA enabled, 25 GPU layers + 32768 context exceeded 6GB VRAM.

**Fix**: Reduced to fit 6GB VRAM:
- `gpu_layers=20` (~2.9GB model buffer)
- `context_length=4096` (~320MB KV cache + 670MB compute)
- Total: ~3.9GB VRAM usage

## Final Configuration

```toml
# For 6GB VRAM (RTX 3060 Laptop)
context_length = 4096
gpu_layers = 20
n_threads = 1
```

- **20 layers on GPU**: Fast inference via CUDA
- **12 layers on CPU**: Single-threaded to avoid ggml race condition
- **4096 context**: Fits in remaining VRAM with KV cache

## Files Modified

1. `src/psyche/tools/tool_definitions.py` - Added emotional_context field
2. `src/psyche/memory/server.py` - Better error logging, connection tracking
3. `src/elpis/server.py` - Signal handlers, lifecycle logging
4. `src/elpis/config/settings.py` - Updated defaults for 6GB VRAM
5. `configs/config.default.toml` - Updated defaults for 6GB VRAM
6. `src/elpis/cli.py` - Set OMP_NUM_THREADS=1 at startup to fully disable OpenMP
7. `src/elpis/llm/backends/llama_cpp/inference.py` - Set OMP/MKL/OPENBLAS threads to 1

## Diagnostic Tools Used

- `coredumpctl list` / `coredumpctl info <pid>` - Crash analysis
- `journalctl --user` - System logs
- `nvidia-smi` - GPU memory monitoring
- `llama_supports_gpu_offload()` - Check CUDA support
- Model load with `verbose=True` - Layer assignment debugging

## Commits

1. `4ffc43b` - Fix store_memory validation and improve connection diagnostics
2. `78a5738` - Reduce default n_threads to 4 (initial attempt)
3. `954c99f` - Set n_threads=1 to avoid ggml SIGSEGV race condition
4. `079f8b3` - Enable CUDA GPU offloading with 25 layers
5. `00e06a3` - Reduce to 20 GPU layers + 4096 context to fit 6GB VRAM

## Lessons Learned

1. llama-cpp-python pip wheels are often built WITHOUT CUDA - must rebuild with `CMAKE_ARGS="-DGGML_CUDA=on"`
2. Even with GPU offloading, some ops run on CPU - threading bugs still affect hybrid execution
3. VRAM budget must account for: model layers + KV cache + compute buffers
4. `coredumpctl` is invaluable for diagnosing native library crashes that bypass Python exception handling

---

## Update: Root Cause Analysis (Late Session)

After extensive debugging (crashes continued even with all env var fixes), we identified the **true root cause**:

### The Real Problem

The threading bridge in `_stream_in_thread()` is fundamentally broken:
- Model loads on main thread
- Streaming runs in worker thread via `threading.Thread`
- `llama_context` is NOT thread-safe (confirmed by llama.cpp maintainers)
- CUDA contexts are thread-local

### Why Our Fixes Didn't Work

| Fix Attempted | Why It Failed |
|--------------|---------------|
| `n_threads=1` | Only affects llama.cpp's internal thread pool, not our Python thread |
| `OMP_NUM_THREADS=1` | Disabled OpenMP, but our Python thread still accesses CUDA context |
| `GGML_CUDA_DISABLE_GRAPHS=1` | Disabled graph optimization, but core thread-safety issue remains |

### Planned Solution

See: `scratchpad/plans/2026-01-14-streaming-stability-plan.md`

**Option A (Recommended)**: Remove threading entirely - iterate sync generator directly with `await asyncio.sleep(0)` between tokens for cooperative yielding.

**Option B (Fallback)**: True multiprocessing if Option A is insufficient.

### Key Insight

Elpis is ALREADY a subprocess of Psyche (process isolation exists). The problem is threading INSIDE Elpis. The MCP polling mechanism can handle async communication without internal threading.
