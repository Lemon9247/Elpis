# Session Report: Tool Architecture Refactor

**Date:** 2026-01-20
**Branch:** `fix/tool-architecture`

## Summary

Refactored the codebase to fix cross-package dependencies and move tools to their correct packages:

1. **Created `src/shared/` package** - Moved constants and MCP patch from psyche to a shared location
2. **Created `src/hermes/tools/` package** - Moved file/bash/search tools from psyche to hermes
3. **Added memory tool schemas** - Proper tool definitions for memory tools in http.py
4. **Consolidated memory tool schemas** - Created `psyche/memory/tool_schemas.py` to avoid duplication

## Changes Made

### New Files Created
- `src/shared/__init__.py` - Shared package init with constant exports
- `src/shared/constants.py` - Memory constants used across packages
- `src/shared/mcp_patch.py` - MCP library patch for race condition
- `src/hermes/tools/__init__.py` - Tool package init
- `src/hermes/tools/tool_engine.py` - Tool execution orchestrator
- `src/hermes/tools/tool_definitions.py` - Tool schemas and input validation
- `src/hermes/tools/implementations/__init__.py`
- `src/hermes/tools/implementations/bash_tool.py`
- `src/hermes/tools/implementations/file_tools.py`
- `src/hermes/tools/implementations/search_tool.py`
- `src/hermes/tools/implementations/directory_tool.py`
- `src/psyche/memory/tool_schemas.py` - Shared memory tool definitions
- `tests/hermes/__init__.py`
- `tests/hermes/unit/__init__.py`
- `tests/shared/__init__.py`

### Files Deleted
- `src/psyche/shared/` (entire directory)
- `src/psyche/mcp_patch.py`
- `src/psyche/tools/` (entire directory)
- `tests/psyche/unit/test_memory_tools.py`

### Files Moved
- `tests/psyche/unit/test_bash_tool.py` -> `tests/hermes/unit/`
- `tests/psyche/unit/test_directory_tool.py` -> `tests/hermes/unit/`
- `tests/psyche/unit/test_file_tools.py` -> `tests/hermes/unit/`
- `tests/psyche/unit/test_search_tool.py` -> `tests/hermes/unit/`
- `tests/psyche/unit/test_tool_definitions.py` -> `tests/hermes/unit/`
- `tests/psyche/unit/test_tool_engine.py` -> `tests/hermes/unit/`
- `tests/psyche/unit/test_shared_constants.py` -> `tests/shared/`

### Import Updates
Updated imports in multiple files across the codebase:
- `src/mnemosyne/server.py` - Uses `shared.constants`
- `src/psyche/core/server.py` - Uses `shared.constants`
- `src/psyche/core/memory_handler.py` - Uses `shared.constants`
- `src/psyche/handlers/dream_handler.py` - Uses `shared.constants`
- `src/psyche/memory/importance.py` - Uses `shared.constants`
- `src/psyche/config/settings.py` - Uses `shared.constants`
- `src/hermes/cli.py` - Uses `shared.mcp_patch` and `hermes.tools`
- `src/psyche/cli.py` - Uses `shared.mcp_patch`
- `src/psyche/server/http.py` - Uses `psyche.memory.tool_schemas`
- `src/psyche/server/mcp.py` - Uses `psyche.memory.tool_schemas`

## Architecture After Refactor

```
src/
├── shared/                   # Cross-package utilities
│   ├── constants.py          # Memory constants
│   └── mcp_patch.py          # MCP library patch
│
├── hermes/
│   └── tools/                # Client-side tools (file/bash/search)
│       ├── tool_engine.py
│       ├── tool_definitions.py
│       └── implementations/
│
├── psyche/
│   ├── memory/
│   │   └── tool_schemas.py   # Memory tool definitions (shared by http/mcp)
│   └── server/
│       ├── http.py           # Injects memory tools into tool list
│       └── mcp.py            # Uses shared tool schemas
│
├── mnemosyne/                # Imports from shared
└── elpis/                    # No cross-deps (unchanged)
```

## Verification

- All 392 tests pass
- No remaining imports from `psyche.shared` or `psyche.tools` in source files
- Memory tools properly defined in `psyche/memory/tool_schemas.py`
- HTTP API injects memory tool schemas into combined tool list

## Next Steps

1. Merge branch to main after review
2. Consider moving docs/psyche/api/tools.rst to docs/hermes/
