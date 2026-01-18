# Elpis Backend Plugin Architecture Refactor

## Summary

Restructure elpis LLM backends into a plugin architecture with dependency injection, providing clear demarcation between llama-cpp and transformers backends.

## Target Structure

```
src/elpis/
├── server.py                   # Uses ServerContext for DI
├── config/
│   └── settings.py             # Root Settings + conversion methods
└── llm/
    ├── __init__.py             # Backend registry and factory exports
    ├── base.py                 # InferenceEngine ABC with capability docs
    ├── prompts.py              # Shared prompt templates
    ├── inference.py            # DEPRECATED: backward compat shim
    ├── transformers_inference.py  # DEPRECATED: backward compat shim
    └── backends/
        ├── __init__.py         # Registry, factory function
        ├── llama_cpp/
        │   ├── __init__.py
        │   ├── config.py       # LlamaCppConfig
        │   └── inference.py    # LlamaInference
        └── transformers/
            ├── __init__.py
            ├── config.py       # TransformersConfig
            ├── inference.py    # TransformersInference
            └── steering.py     # SteeringManager (extracted)
```

---

## Implementation Phases

### Phase 1: Backend Config Classes

**Create new files:**

1. `src/elpis/llm/backends/llama_cpp/config.py`
   - `LlamaCppConfig(BaseSettings)` with llama-cpp specific fields
   - Fields: `path`, `context_length`, `gpu_layers`, `n_threads`, `temperature`, `top_p`, `max_tokens`, `hardware_backend`

2. `src/elpis/llm/backends/transformers/config.py`
   - `TransformersConfig(BaseSettings)` with transformers specific fields
   - Fields: `path`, `context_length`, `temperature`, `top_p`, `max_tokens`, `torch_dtype`, `steering_layer`, `emotion_vectors_dir`

3. **Modify** `src/elpis/config/settings.py`
   - Add `to_llama_cpp_config()` and `to_transformers_config()` methods to `ModelSettings`

### Phase 2: Backend Package Structure

**Create directory structure and __init__.py files:**

1. `src/elpis/llm/backends/__init__.py` - Empty initially
2. `src/elpis/llm/backends/llama_cpp/__init__.py` - Exports `LlamaInference`, `LlamaCppConfig`
3. `src/elpis/llm/backends/transformers/__init__.py` - Exports `TransformersInference`, `TransformersConfig`, `SteeringManager`, `AVAILABLE`

### Phase 3: Extract Steering Logic

**Create** `src/elpis/llm/backends/transformers/steering.py`
- Extract from `transformers_inference.py`:
  - `_load_emotion_vectors()` → `SteeringManager.load_vectors()`
  - `_compute_blended_steering()` → `SteeringManager.compute_blended_vector()`
  - Hook registration/removal → `SteeringManager.apply_hook()`, `remove_hook()`

### Phase 4: Move Inference Implementations

1. **Create** `src/elpis/llm/backends/llama_cpp/inference.py`
   - Move `LlamaInference` class from `src/elpis/llm/inference.py`
   - Update to accept `LlamaCppConfig` instead of `ModelSettings`
   - Add class attributes: `SUPPORTS_STEERING = False`, `MODULATION_TYPE = "sampling"`

2. **Create** `src/elpis/llm/backends/transformers/inference.py`
   - Move `TransformersInference` class from `src/elpis/llm/transformers_inference.py`
   - Refactor to use `SteeringManager`
   - Update to accept `TransformersConfig`
   - Add class attributes: `SUPPORTS_STEERING = True`, `MODULATION_TYPE = "steering"`

### Phase 5: Backend Registry

**Update** `src/elpis/llm/backends/__init__.py`
- Implement `register_backend()`, `get_available_backends()`, `create_backend()`
- Register both backends with availability checks (try/except ImportError)

### Phase 6: Backward Compatibility Shims

**Convert to shims (deprecation warnings):**

1. `src/elpis/llm/inference.py`
   - Re-export `LlamaInference` from `backends.llama_cpp`
   - Emit `DeprecationWarning`

2. `src/elpis/llm/transformers_inference.py`
   - Re-export `TransformersInference` from `backends.transformers`
   - Emit `DeprecationWarning`

### Phase 7: Update Main llm/__init__.py

**Modify** `src/elpis/llm/__init__.py`
- Export `InferenceEngine`, `create_backend`, `get_available_backends`
- Add backward compat exports for `LlamaInference`, `TransformersInference`
- Add `LLAMA_CPP_AVAILABLE`, `TRANSFORMERS_AVAILABLE` flags

### Phase 8: Enhance InferenceEngine ABC

**Modify** `src/elpis/llm/base.py`
- Add class attributes: `SUPPORTS_STEERING`, `MODULATION_TYPE`
- Add comprehensive docstrings explaining modulation strategies
- Document `emotion_coefficients` parameter usage

### Phase 9: ServerContext for Dependency Injection

**Modify** `src/elpis/server.py`

1. Create `ServerContext` dataclass:
   ```python
   @dataclass
   class ServerContext:
       llm: InferenceEngine
       emotion_state: EmotionalState
       regulator: HomeostasisRegulator
       settings: Settings
       active_streams: Dict[str, StreamState]
   ```

2. Add module-level `_context: Optional[ServerContext] = None`

3. Add `get_context() -> ServerContext` function

4. Update `initialize()` to:
   - Use `create_backend()` factory
   - Return `ServerContext`
   - Set module-level `_context`

5. Update all handler functions to use `ctx = get_context()` instead of globals

### Phase 10: Test Updates

**Create new test files:**
- `tests/elpis/unit/llm/__init__.py`
- `tests/elpis/unit/llm/test_backends.py` - Registry/factory tests
- `tests/elpis/unit/llm/test_llama_cpp.py` - Move from `test_llm_inference.py`
- `tests/elpis/unit/llm/test_transformers.py` - Move from `test_transformers_inference.py`
- `tests/elpis/unit/test_server_context.py` - DI tests

**Update existing tests:**
- Update imports in `tests/conftest.py` if needed

---

## Files Summary

### New Files (13)
| File | Description |
|------|-------------|
| `src/elpis/llm/backends/__init__.py` | Backend registry and factory |
| `src/elpis/llm/backends/llama_cpp/__init__.py` | Package exports |
| `src/elpis/llm/backends/llama_cpp/config.py` | LlamaCppConfig class |
| `src/elpis/llm/backends/llama_cpp/inference.py` | LlamaInference class |
| `src/elpis/llm/backends/transformers/__init__.py` | Package exports |
| `src/elpis/llm/backends/transformers/config.py` | TransformersConfig class |
| `src/elpis/llm/backends/transformers/inference.py` | TransformersInference class |
| `src/elpis/llm/backends/transformers/steering.py` | SteeringManager class |
| `tests/elpis/unit/llm/__init__.py` | Test package |
| `tests/elpis/unit/llm/test_backends.py` | Registry tests |
| `tests/elpis/unit/llm/test_llama_cpp.py` | Backend-specific tests |
| `tests/elpis/unit/llm/test_transformers.py` | Backend-specific tests |
| `tests/elpis/unit/test_server_context.py` | DI tests |

### Files to Modify (6)
| File | Changes |
|------|---------|
| `src/elpis/config/settings.py` | Add `to_*_config()` methods |
| `src/elpis/llm/__init__.py` | Update exports, add factory |
| `src/elpis/llm/base.py` | Add capability docs and flags |
| `src/elpis/llm/inference.py` | Convert to deprecation shim |
| `src/elpis/llm/transformers_inference.py` | Convert to deprecation shim |
| `src/elpis/server.py` | Add ServerContext, refactor handlers |

---

## Implementation Order

1. Create backend config classes (no deps)
2. Update ModelSettings with conversion methods
3. Create backend package __init__.py files
4. Extract SteeringManager to steering.py
5. Move LlamaInference to backends/llama_cpp/
6. Move TransformersInference to backends/transformers/
7. Implement backend registry in backends/__init__.py
8. Create backward compat shims
9. Update llm/__init__.py exports
10. Enhance base.py with capability docs
11. Create ServerContext in server.py
12. Update server handlers to use get_context()
13. Create new test files
14. Run tests and fix any issues
