# Server Agent Report: Backend Refactor

**Date:** 2026-01-14
**Agent:** Server Agent
**Task:** Backend registry, settings conversion, shims, and ServerContext implementation

## Summary

Successfully implemented the core infrastructure for the modular backend architecture, including the backend registry/factory system, configuration conversion methods, backward compatibility shims, and the ServerContext dependency injection pattern.

## Files Created/Modified

### 1. Created: `src/elpis/llm/backends/__init__.py`

The central registry and factory module for LLM backends.

**Key Components:**
- `_BACKENDS`: Internal registry dict mapping backend names to (loader, availability) tuples
- `register_backend(name, loader, available)`: Register new backends dynamically
- `get_available_backends()`: Query which backends are installed and available
- `is_backend_available(name)`: Check if a specific backend can be used
- `create_backend(settings)`: Factory function that creates appropriate InferenceEngine based on ModelSettings

**Backend Registration:**
```python
# Registers llama-cpp with try/except for optional deps
try:
    from elpis.llm.backends.llama_cpp import LlamaInference
    register_backend("llama-cpp", _load_llama_cpp, available=True)
except ImportError:
    register_backend("llama-cpp", _load_llama_cpp, available=False)
```

The loader functions (`_load_llama_cpp`, `_load_transformers`) use the settings conversion methods to translate ModelSettings to backend-specific configs.

### 2. Modified: `src/elpis/llm/base.py`

Added capability flags to the `InferenceEngine` ABC:

```python
class InferenceEngine(ABC):
    # Capability flags - subclasses should override these
    SUPPORTS_STEERING: bool = False
    MODULATION_TYPE: Literal["none", "sampling", "steering"] = "none"
```

Added comprehensive docstring explaining:
- What each capability flag means
- How backends should declare their capabilities
- Example usage

### 3. Modified: `src/elpis/config/settings.py`

Added conversion methods to `ModelSettings`:

```python
def to_llama_cpp_config(self) -> "LlamaCppConfig":
    """Convert to llama-cpp backend config."""
    from elpis.llm.backends.llama_cpp.config import LlamaCppConfig
    return LlamaCppConfig(
        path=self.path,
        context_length=self.context_length,
        gpu_layers=self.gpu_layers,
        # ... etc
    )

def to_transformers_config(self) -> "TransformersConfig":
    """Convert to transformers backend config."""
    from elpis.llm.backends.transformers.config import TransformersConfig
    return TransformersConfig(
        path=self.path,
        torch_dtype=self.torch_dtype,
        steering_layer=self.steering_layer,
        # ... etc
    )
```

Also added TYPE_CHECKING imports to avoid circular imports.

### 4. Modified: `src/elpis/llm/__init__.py`

Updated to export the new factory API:
- `create_backend`: Main factory function
- `get_available_backends`: Query available backends
- `is_backend_available`: Check specific backend
- `register_backend`: For custom backends

Maintains backward compatibility by still exposing `LLAMA_CPP_AVAILABLE`, `TRANSFORMERS_AVAILABLE`, and the backend classes themselves.

### 5. Created: `src/elpis/llm/inference.py` (Shim)

Converted to a backward compatibility shim that:
- Re-exports `LlamaInference` and `LlamaCppConfig` from `backends.llama_cpp`
- Emits a `DeprecationWarning` on import

```python
warnings.warn(
    "Importing from elpis.llm.inference is deprecated. "
    "Use elpis.llm.backends.llama_cpp or elpis.llm.create_backend() instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

### 6. Created: `src/elpis/llm/transformers_inference.py` (Shim)

Same pattern as above:
- Re-exports `TransformersInference`, `TransformersConfig`, `SteeringManager` from `backends.transformers`
- Emits deprecation warning

### 7. Modified: `src/elpis/server.py`

This was the most substantial change. Implemented the ServerContext dependency injection pattern:

**ServerContext Dataclass:**
```python
@dataclass
class ServerContext:
    """Container for all server dependencies."""
    llm: InferenceEngine
    emotion_state: EmotionalState
    regulator: HomeostasisRegulator
    settings: Settings
    active_streams: Dict[str, StreamState] = field(default_factory=dict)
```

**get_context() Function:**
```python
def get_context() -> ServerContext:
    """Get the current server context."""
    if _context is None:
        raise RuntimeError("Server not initialized. Call initialize() first.")
    return _context
```

**Updated initialize():**
- Now uses `create_backend(settings.model)` instead of manual if/else
- Logs backend capabilities (SUPPORTS_STEERING, MODULATION_TYPE)
- Returns the ServerContext

**Updated ALL Handlers:**
Every handler function now:
1. Receives `ctx: ServerContext` as first parameter
2. Uses `ctx.llm`, `ctx.emotion_state`, `ctx.regulator` instead of globals
3. Uses `ctx.active_streams` for stream management

Example:
```python
async def _handle_generate(ctx: ServerContext, args: Dict[str, Any]) -> Dict[str, Any]:
    # ... uses ctx.llm.chat_completion(), ctx.emotion_state.get_modulated_params(), etc.
```

## Design Decisions

### 1. Lazy Imports in Factory

The backend loaders use lazy imports to avoid loading heavy dependencies (torch, llama-cpp-python) until actually needed:
```python
def _load_llama_cpp(settings):
    from elpis.llm.backends.llama_cpp import LlamaInference
    config = settings.to_llama_cpp_config()
    return LlamaInference(config)
```

### 2. Global Context with Accessor

Chose to keep a global `_context` variable with a `get_context()` accessor rather than passing context through decorators because:
- MCP server decorators (`@server.call_tool()`) don't support custom parameters
- Accessor pattern provides clear initialization check
- Still encapsulates state better than multiple globals

### 3. Deprecation Warnings in Shims

Used `DeprecationWarning` with `stacklevel=2` so the warning points to the import site in user code, not the shim module itself.

### 4. Settings Conversion Methods

Put the conversion methods on `ModelSettings` rather than the backend configs because:
- ModelSettings knows about all possible fields
- Avoids circular import issues
- Natural place for "outbound" conversion

## Testing Notes

The changes should be verified by:
1. Running existing tests to ensure backward compatibility
2. Testing the factory: `create_backend(settings)` should return appropriate engine
3. Testing deprecation warnings trigger when using old import paths
4. Testing ServerContext is properly initialized and accessible via `get_context()`

## Integration Points

This implementation integrates with the work of other agents:
- **LlamaCpp Agent**: Expects `backends/llama_cpp/inference.py` to export `LlamaInference` using `LlamaCppConfig`
- **Transformers Agent**: Expects `backends/transformers/inference.py` to export `TransformersInference` using `TransformersConfig` and `SteeringManager`

Both backend __init__.py files were verified to export the expected classes.

## Future Considerations

1. **Custom Backend Registration**: Users can register custom backends via `register_backend()`
2. **Backend Discovery**: Could add plugin-based discovery for backends in external packages
3. **Testing Utilities**: May want to add mock backend for testing without loading real models
4. **Context Cleanup**: Could add a `shutdown()` function to clean up the context
