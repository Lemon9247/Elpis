# Plan: Make Mnemosyne Source of Truth for Memory Settings

**Status:** Planned, not yet implemented
**Date:** 2026-01-20
**Branch:** `fix/tool-architecture` (will implement after current work is merged)

## Problem Analysis

The `shared/constants.py` creates confusion about ownership:

| Constant | Mnemosyne | Psyche | Issue |
|----------|-----------|--------|-------|
| `MEMORY_SUMMARY_LENGTH` | Uses (1x) | Uses (6x) | Legitimate contract |
| `MEMORY_CONTENT_TRUNCATE_LENGTH` | No | Yes (4x) | Psyche-only display |
| `AUTO_STORAGE_THRESHOLD` | No | Yes (5x) | Psyche-only logic |
| `CONSOLIDATION_IMPORTANCE_THRESHOLD` | Defines own 0.6 | Uses (3x) | **Duplicated!** |
| `CONSOLIDATION_SIMILARITY_THRESHOLD` | Defines own 0.85 | Uses (2x) | **Duplicated!** |

**Core Issue:** Mnemosyne IS the memory system but Psyche controls memory settings.

**User Direction:** "Mnemosyne should be the source of truth for memory settings"

## Agent Findings

From cross-package analysis:
- **Good isolation** - Packages communicate via MCP/HTTP, not direct imports
- **EmotionalState** naming collision (minor - different purposes)
- **No circular deps** - Clean architecture
- Mnemosyne defines its own thresholds in `mnemosyne/core/models.py:177-181`

## Solution: Mnemosyne Owns Memory Constants

### Architecture After Fix

```
mnemosyne/
  core/constants.py        # Source of truth for ALL memory settings
    - MEMORY_SUMMARY_LENGTH = 500
    - CONSOLIDATION_IMPORTANCE_THRESHOLD = 0.6
    - CONSOLIDATION_SIMILARITY_THRESHOLD = 0.85

psyche/
  config/constants.py      # Psyche-specific (display, heuristics)
    - MEMORY_CONTENT_TRUNCATE_LENGTH = 300
    - AUTO_STORAGE_THRESHOLD = 0.6

shared/
  mcp_patch.py            # Keep - cross-package utility
  constants.py            # DELETE - responsibility unclear
```

### Why This Works
- **Mnemosyne** owns memory storage rules (summary length, consolidation)
- **Psyche** owns coordination heuristics (auto-storage threshold, display truncation)
- Each package controls its own domain

## Files to Modify

### Phase 1: Create mnemosyne/core/constants.py
```python
"""Memory system constants - Mnemosyne is source of truth."""

# Memory content constraints
MEMORY_SUMMARY_LENGTH = 500

# Consolidation thresholds
CONSOLIDATION_IMPORTANCE_THRESHOLD = 0.6
CONSOLIDATION_SIMILARITY_THRESHOLD = 0.85
```

### Phase 2: Create psyche/config/constants.py
```python
"""Psyche-specific constants for memory coordination."""

# Display truncation (UI/logs)
MEMORY_CONTENT_TRUNCATE_LENGTH = 300

# Heuristic threshold for auto-storage decisions
AUTO_STORAGE_THRESHOLD = 0.6
```

### Phase 3: Update Mnemosyne imports
- `src/mnemosyne/server.py` - Use `mnemosyne.core.constants`
- `src/mnemosyne/core/models.py` - Import from constants (remove hardcoded)
- `src/mnemosyne/config/settings.py` - Import from constants

### Phase 4: Update Psyche imports
**For MEMORY_SUMMARY_LENGTH (import from Mnemosyne or duplicate):**
- `src/psyche/core/server.py`
- `src/psyche/core/memory_handler.py`
- `src/psyche/handlers/dream_handler.py`

**For Psyche-only constants:**
- `src/psyche/memory/importance.py` - Use `psyche.config.constants`
- `src/psyche/config/settings.py` - Use `psyche.config.constants`

### Phase 5: Delete shared/constants.py
- Remove `src/shared/constants.py`
- Update `src/shared/__init__.py` to only export mcp_patch
- Move `tests/shared/test_shared_constants.py` appropriately

## Decision: MEMORY_SUMMARY_LENGTH

**User chose:** Psyche imports from Mnemosyne (single source of truth)

This creates a dependency: `psyche -> mnemosyne.core.constants`

## Implementation Order

1. Create `mnemosyne/core/constants.py` with memory settings
2. Update `mnemosyne/` imports to use new constants
3. Create `psyche/config/constants.py` with Psyche-only settings
4. Update `psyche/` imports:
   - MEMORY_SUMMARY_LENGTH from `mnemosyne.core.constants`
   - Other constants from `psyche.config.constants`
5. Delete `shared/constants.py` and update `shared/__init__.py`
6. Update tests

## Verification

1. Run `pytest tests/` - All tests pass
2. `grep -r "shared.constants" src/` returns empty
3. Mnemosyne defines: MEMORY_SUMMARY_LENGTH, CONSOLIDATION_*
4. Psyche defines: MEMORY_CONTENT_TRUNCATE_LENGTH, AUTO_STORAGE_THRESHOLD
5. Psyche imports MEMORY_SUMMARY_LENGTH from Mnemosyne
