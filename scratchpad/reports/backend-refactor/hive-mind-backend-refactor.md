# Backend Refactor Hive Mind

Coordination file for the backend plugin architecture refactor.

## Overview

Restructuring elpis LLM backends into a plugin architecture with:
- Separate backend packages under `src/elpis/llm/backends/`
- Backend-specific config classes
- ServerContext for dependency injection
- Backend registry and factory

## Shared Context

### Already Created
- `src/elpis/llm/backends/llama_cpp/config.py` - LlamaCppConfig class
- `src/elpis/llm/backends/transformers/config.py` - TransformersConfig class

### Target Structure
```
src/elpis/llm/
├── __init__.py             # Updated exports, factory
├── base.py                 # Enhanced with capability flags
├── prompts.py              # Unchanged
├── inference.py            # SHIM: re-export with deprecation
├── transformers_inference.py  # SHIM: re-export with deprecation
└── backends/
    ├── __init__.py         # Registry, factory
    ├── llama_cpp/
    │   ├── __init__.py
    │   ├── config.py       # DONE
    │   └── inference.py    # LlamaInference class
    └── transformers/
        ├── __init__.py
        ├── config.py       # DONE
        ├── inference.py    # TransformersInference class
        └── steering.py     # SteeringManager class
```

## Agent Assignments

### LlamaCpp Agent
- Create `backends/llama_cpp/__init__.py`
- Create `backends/llama_cpp/inference.py` (move from `llm/inference.py`)
- Update to use `LlamaCppConfig` instead of `ModelSettings`
- Add capability flags: `SUPPORTS_STEERING = False`, `MODULATION_TYPE = "sampling"`

### Transformers Agent
- Create `backends/transformers/__init__.py`
- Create `backends/transformers/steering.py` (extract from transformers_inference.py)
- Create `backends/transformers/inference.py` (move from `llm/transformers_inference.py`)
- Update to use `TransformersConfig` and `SteeringManager`
- Add capability flags: `SUPPORTS_STEERING = True`, `MODULATION_TYPE = "steering"`

### Server Agent
- Create `backends/__init__.py` with registry and factory
- Update `config/settings.py` with `to_llama_cpp_config()` and `to_transformers_config()` methods
- Update `llm/__init__.py` with new exports
- Update `llm/base.py` with capability documentation
- Create backward compat shims in `llm/inference.py` and `llm/transformers_inference.py`
- Create `ServerContext` in `server.py`
- Update all server handlers to use `get_context()`

## Agent Status

### LlamaCpp Agent: COMPLETE
- Created `backends/llama_cpp/__init__.py` - Package exports with docstring
- Created `backends/llama_cpp/inference.py` - LlamaInference class refactored to use:
  - `LlamaCppConfig` instead of `ModelSettings` (aliased to `self.settings` for compatibility)
  - Uses `self.settings.chat_format` from config instead of hardcoded "llama-3"
  - Class attributes: `SUPPORTS_STEERING = False`, `MODULATION_TYPE = "sampling"`
  - Comprehensive module and class docstrings explaining the sampling-based modulation approach
  - All methods preserved with identical logic, just updated imports

### Transformers Agent: COMPLETE
- Created `backends/transformers/__init__.py` - Package exports with AVAILABLE flag
- Created `backends/transformers/steering.py` - SteeringManager class extracted from transformers_inference.py
- Created `backends/transformers/inference.py` - TransformersInference class refactored to use:
  - `TransformersConfig` instead of `ModelSettings`
  - `SteeringManager` instance (`self.steering`) for all steering operations
  - Class attributes: `SUPPORTS_STEERING = True`, `MODULATION_TYPE = "steering"`

### Server Agent: COMPLETE
- Created `backends/__init__.py` with:
  - Backend registry (`_BACKENDS` dict)
  - `register_backend()` - Register backends dynamically
  - `get_available_backends()` - Query backend availability
  - `is_backend_available()` - Check specific backend
  - `create_backend()` - Factory function using settings
  - Auto-registration of llama-cpp and transformers backends with try/except for optional deps
- Updated `llm/base.py` with:
  - `SUPPORTS_STEERING` class attribute (bool)
  - `MODULATION_TYPE` class attribute (Literal["none", "sampling", "steering"])
  - Detailed docstring explaining backend capabilities
- Updated `config/settings.py` with:
  - `to_llama_cpp_config()` method on ModelSettings
  - `to_transformers_config()` method on ModelSettings
  - TYPE_CHECKING imports for config types
- Updated `llm/__init__.py` with:
  - Exports for `create_backend`, `get_available_backends`, `is_backend_available`, `register_backend`
  - Backward-compatible imports of LlamaInference and TransformersInference
- Created backward compatibility shims:
  - `llm/inference.py` - Re-exports from backends/llama_cpp with deprecation warning
  - `llm/transformers_inference.py` - Re-exports from backends/transformers with deprecation warning
- Updated `server.py` with:
  - `ServerContext` dataclass holding llm, emotion_state, regulator, settings, active_streams
  - `get_context()` function for dependency injection
  - Updated `initialize()` to use `create_backend()` and return ServerContext
  - Updated ALL handlers to accept `ctx: ServerContext` parameter
  - Uses ctx.llm, ctx.emotion_state, ctx.regulator instead of globals

## Questions / Coordination Notes

(Agents: write questions here if you need input from other agents)
