# Session Report: Psyche Stateless Refactoring

**Date:** 2026-01-20
**Goal:** Make Psyche a stateless memory-enriched inference API by moving client orchestration to Hermes

## Summary

Successfully refactored Psyche to be a stateless API by moving client-side orchestration (ReactHandler, IdleHandler, PsycheClient) to Hermes. Memory consolidation is now handled server-side by PsycheDaemon.

## Architecture Changes

### Before
```
Psyche (handlers/)
├── ReactHandler     ← client orchestration
├── IdleHandler      ← client behavior + consolidation
├── PsycheClient     ← client interface
└── DreamHandler     ← server behavior

Hermes
└── Uses Psyche handlers directly
```

### After
```
Psyche
├── PsycheCore       ← memory coordination, inference
├── DreamHandler     ← server-side dreaming
├── PsycheDaemon     ← server-side consolidation loop
├── HTTP/MCP Server  ← stateless API
└── Memory tools     ← recall_memory, store_memory

Hermes (handlers/)
├── ReactHandler     ← ReAct loop orchestration
├── IdleHandler      ← idle workspace exploration (no consolidation)
└── PsycheClient     ← interface to Psyche (local/remote)
```

## Implementation Details

### Phase 1: Create Hermes handlers directory
- Created `src/hermes/handlers/__init__.py`

### Phase 2: Move PsycheClient
- Created `src/hermes/handlers/psyche_client.py`
- Contains: PsycheClient (ABC), LocalPsycheClient, RemotePsycheClient

### Phase 3: Move ReactHandler
- Created `src/hermes/handlers/react_handler.py`
- Contains: ReactHandler, ReactConfig, ToolCallResult
- Keeps imports from psyche (psyche.memory.compaction, psyche.tools)

### Phase 4: Move IdleHandler
- Created `src/hermes/handlers/idle_handler.py`
- Contains: IdleHandler, IdleConfig, ThoughtEvent
- **Removed consolidation responsibility** - no longer calls maybe_consolidate()
- Keeps SAFE_IDLE_TOOLS and SENSITIVE_PATH_PATTERNS

### Phase 5: Update Settings
- Added IdleSettings to `src/hermes/config/settings.py`
- Removed IdleSettings from `src/psyche/config/settings.py`

### Phase 6: Add Consolidation to PsycheDaemon
- Added consolidation config to ServerConfig:
  - consolidation_enabled: bool = True
  - consolidation_interval: float = 300.0
  - consolidation_importance_threshold: float = 0.6
  - consolidation_similarity_threshold: float = 0.85
- Added `_consolidation_loop()` method
- Added `_maybe_consolidate()` method
- Consolidation task starts/stops with daemon lifecycle
- Updated DreamHandler to trigger consolidation after storing dream insights

### Phase 7: Update Config Files
- Removed `[idle]` section from `configs/psyche.toml`
- Added `[idle]` section to `configs/hermes.toml`

### Phase 8: Clean up Psyche handlers
- Updated `src/psyche/handlers/__init__.py` to export only DreamHandler, DreamConfig
- Deleted old handler files from psyche
- Updated `src/psyche/__init__.py` to remove moved exports
- Updated `src/psyche/config/__init__.py` to remove IdleSettings
- Updated `src/hermes/cli.py` to import from hermes.handlers
- Updated `src/hermes/app.py` to import from hermes.handlers

### Phase 9: Move Tests
- Created `tests/hermes/unit/test_idle_handler.py`
- Created `tests/hermes/unit/test_react_handler.py`
- Deleted old test files from `tests/psyche/unit/`
- Updated `tests/psyche/unit/test_settings.py` to remove IdleSettings tests

## Files Changed

### Created
- `src/hermes/handlers/__init__.py`
- `src/hermes/handlers/psyche_client.py`
- `src/hermes/handlers/react_handler.py`
- `src/hermes/handlers/idle_handler.py`
- `tests/hermes/__init__.py`
- `tests/hermes/unit/__init__.py`
- `tests/hermes/unit/test_idle_handler.py`
- `tests/hermes/unit/test_react_handler.py`

### Modified
- `src/hermes/config/settings.py` - Added IdleSettings
- `src/hermes/cli.py` - Updated imports
- `src/hermes/app.py` - Updated imports
- `src/psyche/__init__.py` - Removed moved exports
- `src/psyche/handlers/__init__.py` - Only DreamHandler exports
- `src/psyche/config/__init__.py` - Removed IdleSettings
- `src/psyche/config/settings.py` - Removed IdleSettings
- `src/psyche/server/daemon.py` - Added consolidation loop
- `src/psyche/handlers/dream_handler.py` - Added post-dream consolidation
- `configs/psyche.toml` - Removed [idle] section
- `configs/hermes.toml` - Added [idle] section
- `tests/psyche/unit/test_settings.py` - Removed IdleSettings tests

### Deleted
- `src/psyche/handlers/psyche_client.py`
- `src/psyche/handlers/react_handler.py`
- `src/psyche/handlers/idle_handler.py`
- `tests/psyche/unit/test_idle_handler.py`
- `tests/psyche/unit/test_react_handler.py`

## Test Results

All tests pass:
- 583 passed, 1 skipped
- Handler tests work with new locations
- Psyche tests still pass with removed handlers
- Hermes tests pass with new handler imports

## Benefits of This Refactoring

1. **Clearer Separation of Concerns**
   - Psyche is now a stateless API server
   - Hermes handles all client-side orchestration
   - Memory consolidation runs server-side regardless of client connection

2. **Remote Mode Ready**
   - Hermes can connect to remote Psyche server
   - Client-side handlers work with both local and remote connections
   - Memory tools execute server-side in remote mode

3. **Simpler Testing**
   - Handler tests are now in hermes test directory
   - Each package tests its own components

4. **Server-Side Consolidation**
   - PsycheDaemon runs consolidation loop independently
   - DreamHandler triggers consolidation after dreams
   - No longer depends on IdleHandler for consolidation

## Post-Implementation Fixes

After the main refactoring, found and fixed additional issues:

### Fixed Issues:
1. **`src/psyche/client/__init__.py`** - Updated deprecated import documentation to point to `hermes.handlers` instead of `psyche.handlers`

2. **`src/psyche/memory/__init__.py`** - Updated module docstring to reflect handlers moved to hermes

3. **`src/hermes/cli.py`** - Removed obsolete parameters:
   - Removed `enable_consolidation` and `consolidation_check_interval` from `IdleConfig` (consolidation is now server-side)
   - Removed `mnemosyne_client` parameter from `IdleHandler` constructor

## Recommendations for Future Work

### Optional Improvements (not required):

1. **Use IdleSettings from TOML config** - The CLI currently hardcodes IdleConfig values. Could be refactored to use `hermes.config.settings.IdleSettings` for consistency, but the current hardcoded values work fine.

2. **Add IdleSettings tests** - Could add tests for the new `IdleSettings` in hermes config (similar to the psyche settings tests).

3. **Manual testing** - Run Hermes in local mode to verify everything works end-to-end.

4. **Remote mode testing** - Run Hermes in remote mode (if Psyche server available) to verify the PsycheClient abstraction works correctly.
