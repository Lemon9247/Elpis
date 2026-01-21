# Session Report: Cross-Package Dependency Refactoring

**Date:** 2026-01-21
**Branch:** `fix/tool-architecture`
**Duration:** ~1 session

## Summary

Implemented the plan from `scratchpad/plans/mnemosyne-constants-source-of-truth.md` to refactor cross-package dependencies by making Mnemosyne the source of truth for memory-related constants. This eliminates the `shared/constants.py` module and establishes clearer ownership of constants.

## Problem

The codebase had hardcoded threshold values (0.6, 0.85) scattered across multiple files in both Mnemosyne and Psyche packages. The `shared/constants.py` module was meant to centralize these, but:
1. Not all files were using it consistently
2. The shared module created unnecessary coupling
3. It wasn't clear which package "owned" which constants

## Solution

Split constants by ownership:
- **Mnemosyne owns memory constants** (`mnemosyne/core/constants.py`):
  - `MEMORY_SUMMARY_LENGTH = 500`
  - `CONSOLIDATION_IMPORTANCE_THRESHOLD = 0.6`
  - `CONSOLIDATION_SIMILARITY_THRESHOLD = 0.85`

- **Psyche owns its own constants** (`psyche/config/constants.py`):
  - `MEMORY_CONTENT_TRUNCATE_LENGTH = 300`
  - `AUTO_STORAGE_THRESHOLD = 0.6`

Psyche imports memory-related constants from Mnemosyne, establishing a clear dependency direction.

## Changes Made

### New Files
| File | Purpose |
|------|---------|
| `src/mnemosyne/core/constants.py` | Memory system constants (source of truth) |
| `src/psyche/config/constants.py` | Psyche-specific constants |
| `tests/mnemosyne/unit/test_constants.py` | Tests for Mnemosyne constants |
| `tests/psyche/unit/test_constants.py` | Tests for Psyche constants |

### Modified Files
**Mnemosyne (3 files):**
- `core/models.py` - ConsolidationConfig uses constants
- `config/settings.py` - ConsolidationSettings uses constants
- `server.py` - Tool schema defaults and handler fallbacks use constants

**Psyche (7 files):**
- `config/settings.py` - Imports from both constant files
- `core/server.py` - Imports from both constant files
- `core/memory_handler.py` - Imports from both constant files
- `handlers/dream_handler.py` - Imports from both constant files
- `mcp/client.py` - Uses Mnemosyne constants for consolidate_memories()
- `memory/importance.py` - Uses Psyche constants
- `server/daemon.py` - Uses Mnemosyne constants for ServerConfig

**Shared:**
- `__init__.py` - Now only exports `apply_mcp_patch`

**Tests (2 files):**
- `tests/mnemosyne/unit/test_settings.py` - Updated imports
- `tests/psyche/unit/test_settings.py` - Updated imports

### Deleted Files
- `src/shared/constants.py`
- `tests/shared/test_shared_constants.py`

## Commits

```
027e473 Add mnemosyne/core/constants.py as source of truth for memory constants
eae47c6 Add psyche/config/constants.py and update imports to use new constant locations
327fdfd Remove shared/constants.py - constants now in package-specific locations
6b000a2 Update tests for new constants locations
```

## Verification

- All 266 tests pass
- No remaining imports from `shared.constants` in source code
- Hardcoded values (0.6, 0.85) now only exist in `mnemosyne/core/constants.py`
- grep verification:
  ```bash
  grep -r "shared.constants" src/  # Returns empty
  grep -r "= 0.6" src/mnemosyne/   # Only in core/constants.py
  grep -r "= 0.85" src/mnemosyne/  # Only in core/constants.py
  ```

## Architecture Notes

The refactoring establishes a clean dependency hierarchy:
```
mnemosyne.core.constants (source of truth for memory)
         ↑
    psyche.* imports from
         ↑
psyche.config.constants (Psyche-specific)
```

This aligns with the principle that Mnemosyne owns memory semantics, while Psyche coordinates higher-level behavior.

## What's Left in `shared/`

The `shared` package now only contains `mcp_patch.py`, which provides a monkey-patch for an MCP library race condition. This is genuinely shared infrastructure that doesn't belong to any specific package.

## Next Steps

The existing plan in `scratchpad/plans/mnemosyne-constants-source-of-truth.md` can be considered complete. The branch `fix/tool-architecture` is ready for PR/merge when appropriate.
