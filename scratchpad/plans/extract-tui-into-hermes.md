# Plan: Extract TUI into Hermes Package

## Overview

Extract the TUI client from `psyche` into a new standalone package called `hermes`. Psyche remains the core library (business logic), and Hermes becomes the user interface layer.

Named for the Greek messenger god - Hermes is the voice and interface of Psyche.

## Architecture After Refactor

```
hermes (TUI package)
├── cli.py          # Entry point (moved from psyche/cli.py)
├── app.py          # Hermes class (renamed from PsycheApp)
├── widgets/        # Textual widgets
├── formatters/     # Tool display formatting
├── commands.py     # Slash commands
├── display.py      # Rich display for REPL
└── repl.py         # Interactive REPL

psyche (core library - no hermes dependency)
├── core/           # PsycheCore, ContextManager, MemoryHandler
├── handlers/       # NEW: ReactHandler, IdleHandler, PsycheClient
├── mcp/            # ElpisClient, MnemosyneClient
├── tools/          # ToolEngine and implementations
├── memory/         # Compaction, importance, reasoning
├── cli.py          # NEW: Headless server stub (for future Phase 5)
└── mcp_patch.py    # MCP library patch
```

## Key Decisions

1. **Handlers move to `psyche.handlers/`** - ReactHandler and IdleHandler are reusable business logic, not TUI-specific
2. **Rename PsycheApp to Hermes** - Greek messenger god, fits the mythology theme
3. **Entry points**:
   - `hermes` - TUI client (the main user-facing command)
   - `psyche` - Headless server stub (placeholder for Phase 5 HTTP/WebSocket server)

## Implementation Waves

### Wave 1: Create psyche.handlers package
Move handlers from `psyche/client/` to new `psyche/handlers/`:
- `psyche/client/react_handler.py` -> `psyche/handlers/react_handler.py`
- `psyche/client/idle_handler.py` -> `psyche/handlers/idle_handler.py`
- `psyche/client/psyche_client.py` -> `psyche/handlers/psyche_client.py`
- Create `psyche/handlers/__init__.py` with exports

### Wave 2: Create hermes package structure
```
src/hermes/
├── __init__.py
├── cli.py              # From psyche/cli.py, updated imports
├── app.py              # From psyche/client/app.py, rename class to Hermes
├── app.tcss            # From psyche/client/app.tcss
├── commands.py         # From psyche/client/commands.py
├── display.py          # From psyche/client/display.py
├── repl.py             # From psyche/client/repl.py
├── widgets/            # From psyche/client/widgets/
└── formatters/         # From psyche/client/formatters/
```

### Wave 3: Update pyproject.toml and create psyche CLI stub
- Add `hermes` to `[project.scripts]` pointing to `hermes.cli:main`
- Update `psyche` to point to new `psyche.cli:main` (headless server stub)
- Add `hermes` to ruff isort known-first-party
- Create `psyche/cli.py` stub that prints "Headless server not yet implemented"
- Update psyche/__init__.py exports

### Wave 4: Update tests
- Move TUI tests to `tests/hermes/`
- Update handler test imports to `psyche.handlers`

### Wave 5: Deprecation and cleanup
- Add deprecation warning to `psyche/client/__init__.py`
- Delete old files from `psyche/client/` (keep __init__.py with deprecation)

## Files to Modify

| Action | File |
|--------|------|
| Create | `src/psyche/handlers/__init__.py` |
| Move | `src/psyche/client/react_handler.py` -> `src/psyche/handlers/` |
| Move | `src/psyche/client/idle_handler.py` -> `src/psyche/handlers/` |
| Move | `src/psyche/client/psyche_client.py` -> `src/psyche/handlers/` |
| Create | `src/hermes/__init__.py` |
| Create | `src/hermes/cli.py` (from psyche/cli.py) |
| Create | `src/hermes/app.py` (from psyche/client/app.py, rename class to Hermes) |
| Copy | `src/hermes/app.tcss` |
| Copy | `src/hermes/commands.py` |
| Copy | `src/hermes/display.py` |
| Copy | `src/hermes/repl.py` |
| Copy | `src/hermes/widgets/*` |
| Copy | `src/hermes/formatters/*` |
| Create | `src/psyche/cli.py` (new stub for headless server) |
| Modify | `pyproject.toml` |
| Modify | `src/psyche/__init__.py` |
| Modify | `src/psyche/client/__init__.py` (deprecation) |

## Verification

1. Run `uv pip install -e .` to install new entry points
2. Run `hermes` command - should launch TUI
3. Run `psyche` command - should print "Headless server not yet implemented"
4. Run full test suite: `pytest tests/`
5. Verify imports work: `python -c "from hermes.app import Hermes; from psyche.handlers import ReactHandler"`
