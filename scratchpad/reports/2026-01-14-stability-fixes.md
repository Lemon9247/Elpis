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

**Fix**: Set `n_threads=1` to disable CPU multi-threading entirely.

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
