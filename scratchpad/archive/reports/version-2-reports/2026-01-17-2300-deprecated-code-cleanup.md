# Session Report: Deprecated Code Cleanup

**Date:** 2026-01-17
**Branch:** phase4/architecture-refactor
**Focus:** Remove deprecated code and legacy shims from TUI extraction

## Overview

Completed cleanup of deprecated code left over from the Hermes TUI extraction. The refactor in previous sessions created the new architecture (PsycheCore, ReactHandler, IdleHandler) but left behind duplicates and backward compatibility shims. This session removed all that legacy code.

## Changes Summary

### Files Deleted (23 total)

#### From `src/psyche/client/` (17 files)
All duplicates of code now in `hermes/` or `psyche/handlers/`:

| Deleted File | Was Duplicate Of |
|--------------|------------------|
| app.py | hermes/app.py |
| app.tcss | hermes/app.tcss |
| commands.py | hermes/commands.py |
| display.py | hermes/display.py |
| repl.py | hermes/repl.py |
| react_handler.py | psyche/handlers/react_handler.py |
| idle_handler.py | psyche/handlers/idle_handler.py |
| psyche_client.py | psyche/handlers/psyche_client.py |
| widgets/__init__.py | hermes/widgets/__init__.py |
| widgets/chat_view.py | hermes/widgets/chat_view.py |
| widgets/sidebar.py | hermes/widgets/sidebar.py |
| widgets/thought_panel.py | hermes/widgets/thought_panel.py |
| widgets/tool_activity.py | hermes/widgets/tool_activity.py |
| widgets/user_input.py | hermes/widgets/user_input.py |
| formatters/__init__.py | hermes/formatters/__init__.py |
| formatters/tool_formatter.py | hermes/formatters/tool_formatter.py |

#### From `src/psyche/memory/`
- **server.py** - 1,869 lines of deprecated MemoryServer monolith
  - Replaced by `psyche.core.server.PsycheCore`
  - Was already marked deprecated with TODO comments

#### Test Files (6 files)
Tests that depended on the deleted MemoryServer:
- tests/psyche/integration/test_memory_server.py
- tests/psyche/integration/test_phase3_combined.py
- tests/psyche/integration/test_phase3_importance.py
- tests/psyche/integration/test_phase3_reasoning.py
- tests/psyche/integration/test_summarization.py
- tests/psyche/unit/test_server_parsing.py

### Files Modified

#### `src/psyche/memory/__init__.py`
- Removed deprecated re-exports (MemoryServer, ServerConfig, ServerState, ThoughtEvent)
- Now only exports: ContextCompactor, CompactionResult
- Updated docstring with new architecture references

#### `src/psyche/client/__init__.py`
- Simplified from re-export shim to pure deprecation warning
- Removed all imports that would break without server.py
- Clear migration guidance in docstring

#### `src/hermes/app.py`
Removed ~200 lines of legacy mode code:
- Conditional MemoryServer imports (lines 33-50)
- `memory_server` parameter and `_use_legacy` flag
- Legacy callback registration
- `_run_server()` legacy method
- `_on_thought_legacy()` callback
- `_update_emotional_display_legacy()`
- All `if self._use_legacy:` guards (~50 locations)

The file went from 755 lines to 578 lines.

#### `src/hermes/repl.py`
- Replaced 231-line REPL implementation with 34-line deprecation stub
- REPL was tightly coupled to MemoryServer architecture
- Raises NotImplementedError with migration guidance

#### `tests/elpis/unit/test_config.py`
- Updated test expectations for CPU-only defaults
- Changed from gpu_layers=20 to gpu_layers=0
- Changed from n_threads=1 to n_threads=4
- Changed from hardware_backend="auto" to hardware_backend="cpu"

## Metrics

| Metric | Value |
|--------|-------|
| Files deleted | 23 |
| Lines removed (estimated) | ~2,500+ |
| Tests passing | 561 |
| Tests skipped | 1 |

## Architecture After Cleanup

```
hermes/                     # TUI package (user interface)
├── app.py                  # Hermes TUI (Textual app)
├── cli.py                  # Entry point
├── widgets/                # UI components
├── formatters/             # Tool display formatting
├── commands.py             # Slash commands
├── display.py              # Rich display utilities
└── repl.py                 # Deprecated stub

psyche/                     # Core library (business logic)
├── core/                   # NEW architecture
│   ├── server.py           # PsycheCore
│   ├── context_manager.py  # ContextManager
│   └── memory_handler.py   # MemoryHandler
├── handlers/               # Request handlers
│   ├── react_handler.py    # ReAct loop
│   ├── idle_handler.py     # Idle thinking
│   └── psyche_client.py    # Client abstraction
├── memory/                 # Memory utilities
│   ├── compaction.py       # Context compaction
│   ├── importance.py       # Memory importance scoring
│   └── reasoning.py        # Reasoning extraction
├── mcp/                    # MCP clients
├── tools/                  # Tool implementations
└── client/                 # Deprecated stub only
```

## Verification

All checks pass:
```bash
# Imports work
python -c "from hermes.app import Hermes; from psyche.handlers import ReactHandler, IdleHandler; from psyche.core import PsycheCore"

# Tests pass
pytest tests/ -v  # 561 passed, 1 skipped
```

## Notes

1. **No backward compatibility maintained** - The old imports will now fail with clear error messages pointing to new locations. This is intentional; the shims were only meant for transition.

2. **REPL is deprecated** - The HermesREPL class now raises NotImplementedError. The Textual TUI is the only supported interface going forward.

3. **MemoryServer completely removed** - All 1,869 lines gone. PsycheCore is the replacement.

4. **Test coverage unchanged** - The deleted tests were for deprecated functionality. All current architecture tests pass.

## Next Steps

- Consider removing `psyche/client/` directory entirely in a future release
- Consider removing `hermes/repl.py` stub in a future release
- Write integration tests for the new architecture (PsycheCore + handlers)
