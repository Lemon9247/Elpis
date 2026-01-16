# Session Summary - 2026-01-16

## Objective
Investigate and fix Psyche shutdown and memory recall issues based on logs in `~/.psyche` and `~/.elpis`.

## Issues Identified

### 1. Memory Recall Failure (Critical)
**Location**: `src/psyche/memory/server.py:562`

The code called `mnemosyne_client.recall_memory()` but `MnemosyneClient` only has a `search_memories()` method. This caused all automatic memory retrieval to fail with:
```
WARNING | Failed to retrieve memories: 'MnemosyneClient' object has no attribute 'recall_memory'
```

### 2. Shutdown Memory Store Errors
**Location**: `src/psyche/memory/server.py:1426`

During shutdown, `_store_messages_to_mnemosyne()` called `self.client.get_emotion()` without checking if Elpis was still connected, causing empty error messages.

### 3. `/quit` Command Not Triggering Consolidation
**Locations**:
- `src/psyche/client/app.py:272` - TUI just called `self.exit()`
- `src/psyche/client/repl.py:207` - REPL just called `stop()`

Neither triggered the proper `shutdown_with_consolidation()` flow.

### 4. MCP Library Race Condition (Upstream)
The MCP library v1.25.0 has a bug at `session.py:448` where it iterates `_response_streams.items()` without copying, causing `RuntimeError: dictionary changed size during iteration` during shutdown.

### 5. store_memory Tool TypeError (Critical)
**Location**: `src/psyche/tools/tool_definitions.py:184-187`

The `store_memory` tool was failing with:
```
TypeError: MemoryTools.store_memory() got an unexpected keyword argument 'emotional_context'
```

**Root Cause**: `StoreMemoryInput` had an `emotional_context` field (added as a hack to handle LLM hallucinations), but when Pydantic's `model_dump()` was called, it included `emotional_context=None` even when the LLM didn't provide it. The handler `MemoryTools.store_memory()` doesn't accept this parameter.

**Flow**:
1. LLM sends args without `emotional_context` (correctly following schema)
2. `StoreMemoryInput` validates and sets default `emotional_context=None`
3. `model_dump()` includes all fields including `emotional_context=None`
4. Handler called with `**model_dump()` → TypeError

**Evidence in logs** (`~/.psyche/psyche.log`):
- Multiple failures on 2026-01-16 20:21-20:22 when user tried to store memory about Elpis/Mnemosyne/Psyche composition
- Psyche kept retrying but same error repeated

## Fixes Applied

### server.py
- Line 562: `recall_memory` -> `search_memories`
- Lines 1423-1448: Added connection check before `get_emotion()`, gracefully handles disconnected state

### app.py
- Line 272: `/quit` now calls `await self.action_quit()` instead of `self.exit()`
- Lines 340-351: Added visual feedback during shutdown (notifications + chat message)

### repl.py
- Lines 206-210: Added `shutdown_with_consolidation()` call and info message before `stop()`

### tool_definitions.py
- Lines 183-187: Removed `emotional_context` field from `StoreMemoryInput`
- The field was a hack to handle LLM hallucinations, but caused TypeError when `model_dump()` passed it to the handler
- Emotional context is auto-fetched by `MemoryTools.store_memory()` internally via `get_emotion_fn`

## Memory Database Review

Checked ChromaDB at `data/memory/`:
- **short_term_memory**: 10 items (semantic + episodic)
- **long_term_memory**: 0 items (nothing promoted yet)

Key memories stored:
- User name: Willow
- Creator info: Willow
- Test memory about Python
- Recent conversation snippets

## Commit
```
b0e8b9b Fix memory recall bug and improve shutdown handling
```

Pushed to `main` branch.

## Files Changed
- `src/psyche/memory/server.py` - Bug fixes
- `src/psyche/client/app.py` - /quit fix + visual feedback
- `src/psyche/client/repl.py` - /quit fix
- `src/psyche/tools/tool_definitions.py` - Remove emotional_context from StoreMemoryInput
- `scratchpad/reports/2026-01-16-psyche-diagnostics.md` - Detailed diagnostic report

## Recommendations
1. Monitor for "Generation failed: Connection closed" errors (may indicate Elpis stability issues)
2. File upstream issue for MCP library dictionary iteration race condition
3. Consider implementing connection health checks and automatic reconnection
