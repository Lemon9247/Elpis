# Backend Plugin Architecture Refactor - Final Report

## Summary

Successfully restructured elpis LLM backends into a plugin architecture with dependency injection, providing clear demarcation between llama-cpp and transformers backends.

## Changes Made

### New Directory Structure

```
src/elpis/llm/
├── __init__.py             # Updated exports, backward compat imports
├── base.py                 # Enhanced with capability flags
├── prompts.py              # Unchanged
├── inference.py            # SHIM: re-export with deprecation warning
├── transformers_inference.py  # SHIM: re-export with deprecation warning
└── backends/
    ├── __init__.py         # Backend registry and factory
    ├── llama_cpp/
    │   ├── __init__.py     # Package exports
    │   ├── config.py       # LlamaCppConfig class
    │   └── inference.py    # LlamaInference class
    └── transformers/
        ├── __init__.py     # Package exports with AVAILABLE flag
        ├── config.py       # TransformersConfig class
        ├── inference.py    # TransformersInference class
        └── steering.py     # SteeringManager class (extracted)
```

### Key Features Implemented

1. **Backend-Specific Config Classes**
   - `LlamaCppConfig`: Fields for llama-cpp specific settings (gpu_layers, n_threads, chat_format)
   - `TransformersConfig`: Fields for transformers specific settings (torch_dtype, steering_layer, emotion_vectors_dir)

2. **Backend Registry & Factory**
   - `register_backend()`: Dynamically register backends
   - `get_available_backends()`: Query available backends
   - `is_backend_available()`: Check if specific backend is available
   - `create_backend()`: Factory function using ModelSettings

3. **Capability Flags on InferenceEngine**
   - `SUPPORTS_STEERING`: Boolean flag indicating steering support
   - `MODULATION_TYPE`: Literal type ("none", "sampling", "steering")
   - LlamaInference: `SUPPORTS_STEERING=False`, `MODULATION_TYPE="sampling"`
   - TransformersInference: `SUPPORTS_STEERING=True`, `MODULATION_TYPE="steering"`

4. **SteeringManager Class (Extracted)**
   - `load_vectors()`: Load emotion vectors from directory
   - `compute_blended_vector()`: Blend emotion vectors by coefficients
   - `apply_hook()`: Register forward hook for steering injection
   - `remove_hook()`: Clean up hook on completion

5. **ServerContext for Dependency Injection**
   - `ServerContext` dataclass: Holds llm, emotion_state, regulator, settings, active_streams
   - `get_context()`: Access current server context
   - All handlers updated to use context instead of globals

6. **Backward Compatibility Shims**
   - `llm/inference.py`: Re-exports from backends/llama_cpp with deprecation warning
   - `llm/transformers_inference.py`: Re-exports from backends/transformers with deprecation warning

### Files Created (9)

| File | Description |
|------|-------------|
| `backends/__init__.py` | Backend registry and factory functions |
| `backends/llama_cpp/__init__.py` | Package exports |
| `backends/llama_cpp/config.py` | LlamaCppConfig class |
| `backends/llama_cpp/inference.py` | LlamaInference class |
| `backends/transformers/__init__.py` | Package exports with AVAILABLE flag |
| `backends/transformers/config.py` | TransformersConfig class |
| `backends/transformers/inference.py` | TransformersInference class |
| `backends/transformers/steering.py` | SteeringManager class |
| `scratchpad/backend-refactor/` | Coordination files and reports |

### Files Modified (6)

| File | Changes |
|------|---------|
| `config/settings.py` | Added `to_llama_cpp_config()` and `to_transformers_config()` methods |
| `llm/__init__.py` | Updated exports, added factory imports, backward compat imports |
| `llm/base.py` | Added SUPPORTS_STEERING and MODULATION_TYPE class attributes |
| `llm/inference.py` | Converted to deprecation shim |
| `llm/transformers_inference.py` | Converted to deprecation shim |
| `server.py` | Added ServerContext, get_context(), updated all handlers |

### Test Updates

- Updated `tests/elpis/unit/test_llm_inference.py`: New import paths for llama_cpp backend
- Updated `tests/elpis/unit/test_transformers_inference.py`: New import paths, fixed mocks
- Updated `tests/elpis/integration/test_mcp_server.py`: Updated to use ServerContext pattern

## Test Results

```
================== 110 passed, 1 skipped, 2 warnings in 10.11s ==================
```

- 91 unit tests passing
- 19 integration tests passing
- 1 test skipped (import guard test only runs when transformers unavailable)
- 2 warnings (expected deprecation warnings from backward compat shims)

## Architecture Benefits

1. **Clear Separation**: Each backend is now a self-contained package with its own config, inference, and supporting classes
2. **Runtime Discovery**: Backends register themselves and can be queried for availability
3. **Capability Introspection**: Code can query backend capabilities at runtime
4. **Dependency Injection**: ServerContext enables better testability and cleaner handler code
5. **Backward Compatibility**: Old import paths still work with deprecation warnings
6. **Extensibility**: New backends can be added by implementing InferenceEngine and registering

## Migration Guide

### Old Import (Deprecated)
```python
from elpis.llm.inference import LlamaInference
from elpis.llm.transformers_inference import TransformersInference
```

### New Import (Recommended)
```python
from elpis.llm.backends.llama_cpp import LlamaInference, LlamaCppConfig
from elpis.llm.backends.transformers import TransformersInference, TransformersConfig

# Or use the factory
from elpis.llm import create_backend
llm = create_backend(settings.model)
```

### Using ServerContext
```python
from elpis.server import get_context

ctx = get_context()
response = await ctx.llm.chat_completion(messages)
current_emotion = ctx.emotion_state.quadrant
```

## Subagent Coordination

Three subagents worked in parallel:
- **LlamaCpp Agent**: Created llama_cpp backend package
- **Transformers Agent**: Created transformers backend package with extracted SteeringManager
- **Server Agent**: Created registry, shims, and ServerContext

Coordination was managed via `scratchpad/backend-refactor/hive-mind-backend-refactor.md`.
