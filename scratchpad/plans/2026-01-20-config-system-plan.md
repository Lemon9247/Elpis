# Configuration System Unification Plan

## Overview

Create unified Pydantic settings for Mnemosyne, Psyche, and Hermes following Elpis's established pattern at `src/elpis/config/settings.py`. Environment variables only (no YAML/TOML).

**Estimated: ~4-5 sessions** (can run in parallel with Phase 7)

---

## Environment Variable Naming Convention

| Package | Prefix | Example |
|---------|--------|---------|
| Elpis | `ELPIS_` | `ELPIS_MODEL__CONTEXT_LENGTH=4096` (already exists) |
| Mnemosyne | `MNEMOSYNE_` | `MNEMOSYNE_STORAGE__PERSIST_DIRECTORY=./data/memory` |
| Psyche | `PSYCHE_` | `PSYCHE_CONTEXT__MAX_TOKENS=24000` |
| Hermes | `HERMES_` | `HERMES_CONNECTION__SERVER_URL=http://localhost:8741` |

Nested settings use double underscore delimiter (`__`).

---

## Implementation Phases

### Phase 1: Shared Constants (~0.5 session)

**Create:** `src/psyche/shared/constants.py`

```python
# Memory processing
MEMORY_SUMMARY_LENGTH = 500
MEMORY_CONTENT_TRUNCATE_LENGTH = 300

# Auto-storage thresholds
AUTO_STORAGE_THRESHOLD = 0.6
CONSOLIDATION_IMPORTANCE_THRESHOLD = 0.6
CONSOLIDATION_SIMILARITY_THRESHOLD = 0.85
```

**Update imports in:**
- `src/mnemosyne/server.py`
- `src/psyche/core/memory_handler.py`
- `src/psyche/core/server.py`
- `src/psyche/tools/implementations/memory_tools.py`

---

### Phase 2: Mnemosyne Settings (~1 session)

**Create:** `src/mnemosyne/config/settings.py`

```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class StorageSettings(BaseSettings):
    persist_directory: str = "./data/memory"
    embedding_model: str = "all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(env_prefix="MNEMOSYNE_STORAGE_")

class LoggingSettings(BaseSettings):
    level: str = "INFO"
    quiet: bool = False  # Suppress stderr (set by Psyche when subprocess)

    model_config = SettingsConfigDict(env_prefix="MNEMOSYNE_LOGGING_")

class Settings(BaseSettings):
    storage: StorageSettings = Field(default_factory=StorageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False
    )
```

**Modify:**
- `src/mnemosyne/storage/chroma_store.py` - Accept settings in `__init__`
- `src/mnemosyne/server.py` - Load settings, pass to store
- `src/mnemosyne/cli.py` - Load settings on startup

---

### Phase 3: Psyche Settings (~1.5 sessions)

**Create:** `src/psyche/config/settings.py`

Key settings classes:
- `ContextSettings` - max_context_tokens, reserve_tokens, checkpoint_interval
- `MemorySettings` - auto_retrieval, auto_storage, summary_length
- `IdleSettings` - delays, cooldowns, tool iteration limits
- `ServerSettings` - http_host, http_port, elpis/mnemosyne commands
- `ToolSettings` - bash_timeout, max_file_size

**Critical method:**
```python
class ContextSettings(BaseSettings):
    max_context_tokens: int = 24000
    reserve_tokens: int = 4000

    @classmethod
    def from_elpis_capabilities(cls, context_length: int) -> "ContextSettings":
        """Create settings based on Elpis context window."""
        return cls(
            max_context_tokens=int(context_length * 0.75),
            reserve_tokens=int(context_length * 0.20)
        )
```

**Migrate from dataclasses:**
- `src/psyche/core/context_manager.py:ContextConfig` → `ContextSettings`
- `src/psyche/core/server.py:CoreConfig` → use settings
- `src/psyche/core/memory_handler.py:MemoryHandlerConfig` → `MemorySettings`
- `src/psyche/server/daemon.py:ServerConfig` → `ServerSettings`
- `src/psyche/handlers/idle_handler.py:IdleConfig` → `IdleSettings`

---

### Phase 4: Hermes Settings + Context Fix (~1.5 sessions)

**Create:** `src/hermes/config/settings.py`

```python
class ConnectionSettings(BaseSettings):
    server_url: Optional[str] = None  # None = local mode
    elpis_command: str = "elpis-server"
    mnemosyne_command: str = "mnemosyne-server"

    model_config = SettingsConfigDict(env_prefix="HERMES_CONNECTION_")

class WorkspaceSettings(BaseSettings):
    path: str = "."

    model_config = SettingsConfigDict(env_prefix="HERMES_WORKSPACE_")
```

**Fix TODO at `src/hermes/cli.py:239`:**

Current code:
```python
# TODO: Query Elpis capabilities after connecting to get actual context_length
core_config = CoreConfig(
    context=ContextConfig(
        max_context_tokens=3000,  # Hardcoded fallback
        reserve_tokens=800,
    ),
    ...
)
```

Fixed approach:
```python
async def _configure_context_from_elpis(elpis_client: ElpisClient) -> ContextSettings:
    """Query Elpis capabilities and create properly sized context settings."""
    try:
        capabilities = await elpis_client.get_capabilities()
        context_length = capabilities.get("context_length", 4096)
        settings = ContextSettings.from_elpis_capabilities(context_length)
        logger.info(f"Configured from Elpis: context_length={context_length}, "
                   f"max_tokens={settings.max_context_tokens}")
        return settings
    except Exception as e:
        logger.warning(f"Failed to query Elpis: {e}, using defaults")
        return ContextSettings()
```

---

### Phase 5: Testing & Documentation (~0.5 session)

1. Update existing tests for new settings imports
2. Add tests for settings loading from env vars
3. Update `configs/` templates with new variable names
4. Update README configuration section

---

## Files Summary

### New Files
| File | Description |
|------|-------------|
| `src/psyche/shared/__init__.py` | Shared module |
| `src/psyche/shared/constants.py` | Magic numbers |
| `src/mnemosyne/config/__init__.py` | Settings export |
| `src/mnemosyne/config/settings.py` | Mnemosyne settings |
| `src/psyche/config/__init__.py` | Settings export |
| `src/psyche/config/settings.py` | Psyche settings |
| `src/hermes/config/__init__.py` | Settings export |
| `src/hermes/config/settings.py` | Hermes settings |

### Modified Files
| File | Changes |
|------|---------|
| `src/mnemosyne/storage/chroma_store.py` | Accept StorageSettings |
| `src/mnemosyne/server.py` | Load and use settings |
| `src/mnemosyne/cli.py` | Initialize settings |
| `src/psyche/core/context_manager.py` | Use ContextSettings |
| `src/psyche/core/server.py` | Use settings, remove hardcoded constants |
| `src/psyche/core/memory_handler.py` | Use MemorySettings + shared constants |
| `src/psyche/server/daemon.py` | Use ServerSettings |
| `src/psyche/handlers/idle_handler.py` | Use IdleSettings |
| `src/psyche/tools/implementations/memory_tools.py` | Use shared constants |
| `src/hermes/cli.py` | Use settings, **fix TODO line 239** |

---

## Verification

### 1. Environment Variables Work
```bash
# Test Mnemosyne storage path override
MNEMOSYNE_STORAGE__PERSIST_DIRECTORY=/tmp/test_memory python -c \
  "from mnemosyne.config import Settings; print(Settings().storage.persist_directory)"
# Expected: /tmp/test_memory
```

### 2. Context Sync in Local Mode
```bash
hermes --debug
# Check logs for: "Configured from Elpis: context_length=X"
# NOT: "using defaults" or hardcoded 3000
```

### 3. All Tests Pass
```bash
pytest tests/
```

### 4. .env File Loading
```bash
echo "PSYCHE_CONTEXT__MAX_TOKENS=16000" > .env
python -c "from psyche.config import Settings; print(Settings().context.max_context_tokens)"
# Expected: 16000
```

---

## Backward Compatibility

- CLI arguments remain functional (Click options override env vars)
- Default values match current hardcoded values
- Existing dataclass constructors can remain temporarily for compatibility
- No breaking changes to public APIs
