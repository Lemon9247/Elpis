# Session Report: Remove Local Mode from Hermes

**Date**: 2026-01-20
**Branch**: `feature/unified-config-settings`
**Commits**: 3 commits pushed

## Summary

Removed local mode from Hermes, making it exclusively connect to a Psyche server via HTTP. This simplifies the codebase by removing ~600 lines of local mode infrastructure and ~200 lines of dead code/outdated comments.

## Changes Made

### Commit 1: Remove local mode from Hermes (`34d9445`)

**Files Deleted:**
- `src/hermes/handlers/react_handler.py` (~544 lines) - ReAct loop handling for local mode
- `src/hermes/handlers/idle_handler.py` (~642 lines) - Idle thinking for local mode
- `tests/hermes/unit/test_idle_handler.py` - Tests for removed handler
- `tests/hermes/unit/test_react_handler.py` - Tests for removed handler

**Files Simplified:**
- `src/hermes/handlers/psyche_client.py` - Removed `LocalPsycheClient` class (~120 lines)
- `src/hermes/handlers/__init__.py` - Only exports `PsycheClient`, `RemotePsycheClient`
- `src/hermes/app.py` - Removed handler params, simplified to remote-only
- `src/hermes/cli.py` - Removed `--elpis-command`, `--mnemosyne-command`, `--no-memory` options
- `src/hermes/config/settings.py` - Removed `IdleSettings`, local mode config
- `configs/hermes.toml` - Simplified to connection/workspace/logging only
- `tests/hermes/unit/test_settings.py` - Updated for simplified settings

**Key Architectural Change:**
- Hermes now always connects to `http://127.0.0.1:8741` by default
- `--server` flag allows connecting to different Psyche servers
- Tools execute locally via `ToolEngine`, server handles inference/memory

### Commit 2: Update documentation (`05ebf6c`)

- **README.md**: Removed local mode architecture diagram, updated usage instructions
- **QUICKSTART.md**: Removed local mode option, updated quick start flow
- **examples/README.md**: Updated to show `psyche-server` + `hermes` workflow
- Updated roadmap to mark Phase 7 (unified config) as complete

### Commit 3: Clean up dead code and outdated references (`52ab58e`)

**Files Deleted:**
- `src/hermes/display.py` (~173 lines) - Unused module with broken import

**Dead Code Removed:**
- `ThoughtEvent` class from `app.py` (unused)
- `_on_thought()` method from `app.py` (never called)
- Unused `dataclass` import

**Test Data Updated:**
- `tests/test_data/hermes.toml` - Removed obsolete settings

**Outdated Comments Removed:**
- References to `ReactHandler` and `IdleHandler` in:
  - `psyche/core/server.py`
  - `psyche/handlers/dream_handler.py`
  - `psyche/config/settings.py`
  - `psyche/config/__init__.py`
  - `psyche/server/daemon.py`

## Architecture After Changes

```
┌──────────────────────────────────────────────────────────┐
│                     PSYCHE SERVER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  PsycheCore  │  │    Elpis     │  │  Mnemosyne   │    │
│  │              │  │  (Inference) │  │   (Memory)   │    │
│  │ - Context    │  │ - LLM gen    │  │ - ChromaDB   │    │
│  │ - Memory     │  │ - Emotion    │  │ - Clustering │    │
│  │ - Dreams     │  │              │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                         MCP (stdio)                      │
└─────────────────────────┬────────────────────────────────┘
                          │ HTTP API (OpenAI-compatible)
                          │
                   ┌──────┴──────┐
                   │   Hermes    │
                   │    (TUI)    │
                   │             │
                   │ - Chat view │
                   │ - Tools     │
                   └─────────────┘
```

## Lines of Code Impact

| Category | Lines Removed |
|----------|---------------|
| Handler code (ReactHandler, IdleHandler) | ~1,186 |
| LocalPsycheClient | ~120 |
| Handler tests | ~789 |
| CLI local mode code | ~230 |
| Dead code (display.py, ThoughtEvent) | ~185 |
| Config/settings simplification | ~75 |
| Outdated comments | ~15 |
| **Total** | **~2,600 lines** |

## Testing

- All existing Hermes settings tests pass (7 tests)
- Import verification successful for all modified modules
- No broken imports detected after cleanup

## Workflow Now

```bash
# Start the server
psyche-server

# Connect with Hermes (default: localhost:8741)
hermes

# Or connect to a specific server
hermes --server http://myserver:8741
```

## Future Considerations

1. **ThoughtPanel widget** - Still exists in Hermes but isn't populated since `_on_thought()` was removed. Could be repurposed or removed.

2. **Idle settings in psyche.toml** - The server still has consolidation settings. These work independently of the removed client-side idle handler.

3. **Remote streaming interruption** - Currently can't interrupt a generation in progress when using remote mode. Would need server-side cancellation support.

## Session Notes

- The plan was well-structured and followed the implementation order exactly
- Found additional cleanup items (display.py, ThoughtEvent, outdated comments) during review
- All changes maintain backward compatibility with the Psyche server API

## Codebase Review Summary

A comprehensive codebase review was performed after the refactoring. **Overall score: 8/10** - production-ready with minor improvements suggested.

### Strengths Identified
- Clean MCP-based architecture with good separation of concerns
- Well-documented modules with comprehensive docstrings
- 583 tests passing with good core coverage
- Consistent logging with loguru throughout
- TOML-based config with pydantic validation

### Outstanding Issues

The following issues were identified and reported as GitHub issues:

1. **RemotePsycheClient lacks error recovery** (Medium priority)
   - HTTP connection errors could crash Hermes
   - Needs exponential backoff reconnection logic

2. **No tests for Hermes handlers** (Medium priority)
   - Only settings tests exist
   - Need tests for RemotePsycheClient, tool execution flow

3. **No tests for DreamHandler** (Medium priority)
   - Dream generation and consolidation trigger untested

4. **Streaming state management has no size limit** (Low priority)
   - `StreamState` dict in elpis/server.py could leak
   - Needs max stream count and TTL

5. **Missing OpenAI API documentation** (Low priority)
   - HTTP server provides OpenAI-compatible API but no docs

6. **Missing deployment/production guide** (Low priority)
   - How to run Psyche in production not documented

### Recommended Next Steps

**Priority 1 (0.5 session):**
- Add error recovery to RemotePsycheClient with exponential backoff

**Priority 2 (1 session):**
- Add test coverage for Hermes handlers and CLI
- Add DreamHandler tests

**Priority 3 (1-2 sessions):**
- Add streaming state limits with TTL
- Document OpenAI API compatibility
- Add deployment/production guide
