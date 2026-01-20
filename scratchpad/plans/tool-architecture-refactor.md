# Plan: Fix Tool Architecture & Cross-Package Dependencies

## Summary

Fix architectural issues identified in codebase review:
1. **Move file/bash/search tools to Hermes** - Client-side tools in wrong package
2. **Add memory tool schemas server-side** - LLM needs proper schemas
3. **Create shared package** - Move constants and MCP patch to top-level shared module
4. **Clean up dead code** - Remove unused `MemoryTools` class

## Branch

Create new branch: `fix/tool-architecture`

## Current Architecture Problems

```
Hermes ──imports──> psyche.tools.ToolEngine      ← WRONG (tools are client-side)
Hermes ──imports──> psyche.mcp_patch             ← WRONG (should be shared)
Mnemosyne ──imports──> psyche.shared.constants   ← WRONG (circular dependency)

Memory tools: LLM knows from system prompt text only, no formal schemas
```

## Target Architecture

```
src/
├── shared/                   ← NEW: Cross-package utilities
│   ├── __init__.py
│   ├── constants.py          ← Moved from psyche/shared/
│   └── mcp_patch.py          ← Moved from psyche/
│
├── hermes/
│   ├── tools/                ← NEW: Client-side tools
│   │   ├── tool_engine.py
│   │   ├── tool_definitions.py
│   │   └── implementations/
│   └── ...
│
├── psyche/
│   ├── server/http.py        ← Injects memory tool schemas
│   └── ...                   ← No more psyche/tools/
│
├── mnemosyne/                ← Imports from shared, not psyche
└── elpis/                    ← Unchanged (no cross-deps)
```

## Files to Modify

### Phase 1: Create Shared Package

**Create:**
- `src/shared/__init__.py`
- `src/shared/constants.py` (copy from `psyche/shared/constants.py`)
- `src/shared/mcp_patch.py` (copy from `psyche/mcp_patch.py`)

**Update imports in:**
- `src/mnemosyne/server.py` - Change `from psyche.shared.constants` to `from shared.constants`
- `src/psyche/core/server.py` - Change to `from shared.constants`
- `src/psyche/core/memory_handler.py` - Change to `from shared.constants`
- `src/psyche/handlers/dream_handler.py` - Change to `from shared.constants`
- `src/psyche/memory/importance.py` - Change to `from shared.constants`
- `src/psyche/config/settings.py` - Change to `from shared.constants`
- `src/hermes/cli.py` - Change `from psyche.mcp_patch` to `from shared.mcp_patch`
- `src/psyche/cli.py` - Change `from psyche.mcp_patch` to `from shared.mcp_patch`

**Delete:**
- `src/psyche/shared/` directory
- `src/psyche/mcp_patch.py`

**Update tests:**
- `tests/psyche/unit/test_shared_constants.py` - Change imports to `from shared.constants`
- `tests/psyche/unit/test_settings.py` - Change imports
- `tests/mnemosyne/unit/test_settings.py` - Change imports

### Phase 2: Move Tools to Hermes

**Create:**
- `src/hermes/tools/__init__.py`
- `src/hermes/tools/tool_engine.py` (copy from psyche, update imports)
- `src/hermes/tools/tool_definitions.py` (copy, remove memory input models)
- `src/hermes/tools/implementations/__init__.py`
- `src/hermes/tools/implementations/bash_tool.py`
- `src/hermes/tools/implementations/file_tools.py`
- `src/hermes/tools/implementations/directory_tool.py`
- `src/hermes/tools/implementations/search_tool.py`

**Update imports:**
- `src/hermes/cli.py` - Change `from psyche.tools` to `from hermes.tools`
- `src/hermes/app.py` - Change `from psyche.tools` to `from hermes.tools`

### Phase 3: Add Memory Tool Schemas Server-Side

**Modify:** `src/psyche/server/http.py`

**How it works:**
1. Server receives tools from Hermes (file/bash/search)
2. Server appends memory tool schemas to the list
3. `_format_tool_descriptions()` converts ALL tools to markdown for system prompt
4. LLM generates `\`\`\`tool_call` blocks for any tool
5. `_separate_tool_calls()` checks `MEMORY_TOOLS` set to route execution
6. Memory tools execute server-side via `_execute_memory_tool()`

Add memory tool definitions that get appended to client-provided tools:

```python
# After line 39, add:
MEMORY_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Search for relevant memories from long-term storage",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "n_results": {"type": "integer", "default": 5, "description": "Number of results"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "store_memory",
            "description": "Store important information in long-term memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to store"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags"}
                },
                "required": ["content"]
            }
        }
    }
]
```

In `chat_completions()` (around line 266), inject memory tools:
```python
# Combine client tools with server-side memory tools
all_tools = list(request.tools or [])
all_tools.extend([Tool(**t) for t in MEMORY_TOOL_DEFINITIONS])
```

### Phase 4: Delete Dead Code

**Delete files:**
- `src/psyche/tools/` (entire directory)
- `tests/psyche/unit/test_memory_tools.py`

### Phase 5: Update Tests

**Move/update:**
- Move `tests/psyche/unit/test_shared_constants.py` to `tests/shared/`
- Any tool-related tests from `tests/psyche/` to `tests/hermes/`

## Implementation Order

1. Create new branch `fix/tool-architecture`
2. Create `src/shared/` package with constants and mcp_patch
3. Update all imports for shared package
4. Delete `src/psyche/shared/` and `src/psyche/mcp_patch.py`
5. Create `src/hermes/tools/` directory structure
6. Copy tool files from psyche to hermes (update internal imports)
7. Update `hermes/cli.py` and `hermes/app.py` imports
8. Add memory tool schemas to `psyche/server/http.py`
9. Delete `src/psyche/tools/` directory
10. Update/move tests
11. Run tests and verify

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/psyche/shared/constants.py` | Constants to move to shared |
| `src/psyche/mcp_patch.py` | MCP patch to move to shared |
| `src/mnemosyne/server.py:46` | Imports psyche.shared.constants |
| `src/hermes/cli.py:42,49-50` | Imports mcp_patch and psyche.tools |
| `src/hermes/app.py:39` | Current psyche.tools import |
| `src/psyche/server/http.py:38-39` | MEMORY_TOOLS constant |
| `src/psyche/server/http.py:266-268` | Tool description injection point |
| `src/psyche/server/http.py:651-682` | Memory tool execution |

## Verification

1. Run `pytest tests/` - All tests pass
2. Verify no imports from `psyche.shared` or `psyche.tools` remain (except in psyche itself)
3. Start server: `psyche-server`
4. Start client: `hermes`
5. Test file tool: Ask to read a file
6. Test memory tool: Ask to "remember this: test memory" then "recall memories about test"
7. Verify memory tool shows in tool activity with proper schema

## Summary of Changes

| Package | Before | After |
|---------|--------|-------|
| **shared** | N/A | NEW: constants.py, mcp_patch.py |
| **hermes** | Imports from psyche.tools | Has own tools/ directory |
| **psyche** | Has tools/, shared/, mcp_patch | No tools/, imports from shared |
| **mnemosyne** | Imports from psyche.shared | Imports from shared |
