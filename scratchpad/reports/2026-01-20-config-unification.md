# Session Report: Configuration System Unification

**Date:** 2026-01-20
**Branch:** `feature/unified-config-settings`
**Commit:** 2229e39

## Summary

Created unified Pydantic settings for Mnemosyne, Psyche, and Hermes following Elpis's established pattern. This addresses the "wonky" configuration mentioned in the README roadmap.

## Changes Made

### New Files Created

| File | Description |
|------|-------------|
| `src/psyche/shared/constants.py` | Shared magic numbers (MEMORY_SUMMARY_LENGTH=500, AUTO_STORAGE_THRESHOLD=0.6, etc.) |
| `src/psyche/shared/__init__.py` | Exports for shared constants |
| `src/mnemosyne/config/settings.py` | StorageSettings, ConsolidationSettings, LoggingSettings |
| `src/mnemosyne/config/__init__.py` | Exports for Mnemosyne settings |
| `src/psyche/config/settings.py` | ContextSettings, MemorySettings, IdleSettings, ServerSettings, ToolSettings |
| `src/psyche/config/__init__.py` | Exports for Psyche settings |
| `src/hermes/config/settings.py` | ConnectionSettings, WorkspaceSettings, LoggingSettings |
| `src/hermes/config/__init__.py` | Exports for Hermes settings |
| `scratchpad/plans/2026-01-20-config-system-plan.md` | Implementation plan |

### Modified Files

| File | Changes |
|------|---------|
| `src/mnemosyne/storage/chroma_store.py` | Accept optional StorageSettings in __init__ |
| `src/mnemosyne/server.py` | Import shared constants, load settings on startup |
| `src/mnemosyne/cli.py` | Load settings from environment, pass to server |
| `src/hermes/cli.py` | Import settings classes, use env-configurable defaults |
| `src/psyche/tools/implementations/memory_tools.py` | Use MEMORY_SUMMARY_LENGTH constant |
| `tests/elpis/integration/test_mcp_server.py` | Fix tool count (8 -> 9 for get_capabilities) |
| `tests/psyche/unit/test_memory_tools.py` | Use MEMORY_SUMMARY_LENGTH in test |

## Environment Variable Prefixes

| Package | Prefix | Example |
|---------|--------|---------|
| Elpis | `ELPIS_` | Already exists |
| Mnemosyne | `MNEMOSYNE_` | `MNEMOSYNE_STORAGE__PERSIST_DIRECTORY=./data/memory` |
| Psyche | `PSYCHE_` | `PSYCHE_CONTEXT__MAX_TOKENS=24000` |
| Hermes | `HERMES_` | `HERMES_CONNECTION__SERVER_URL=http://localhost:8741` |

## Outstanding Work

### 1. Full Psyche Migration (Deferred)

The Psyche dataclasses (`ContextConfig`, `CoreConfig`, `MemoryHandlerConfig`, etc.) still exist alongside the new Pydantic settings. A future session should:
- Update all Psyche components to construct configs from settings
- Possibly deprecate the dataclasses or keep them as convenience wrappers

### 2. Context Length Query Fix (Deferred)

The `hermes/cli.py` TODO for querying Elpis capabilities before creating CoreConfig remains. The issue is architectural:
- In local mode, MCP clients connect inside the Textual app
- But CoreConfig must be created before the app starts
- Proper fix requires refactoring the startup sequence

The `ContextSettings.from_elpis_capabilities()` method is ready for when this is fixed.

### 3. Test for Settings Loading

Should add tests that verify:
- Settings load from environment variables
- .env file loading works
- Defaults match expected values

## Verification

All 561 tests pass:
```
================= 561 passed, 1 skipped, 2 warnings in 15.15s ==================
```

## Notes

- Python 3.14 doesn't work yet (onnxruntime lacks wheels), used 3.13
- The venv was recreated with `uv venv venv --python 3.13`
- Pre-existing test failure was fixed (get_capabilities tool count)
