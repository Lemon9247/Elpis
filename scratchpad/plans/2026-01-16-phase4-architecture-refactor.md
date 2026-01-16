# Phase 4: Architecture Refactor Plan

**Goal:** Extract a clean Psyche Core (~500 lines) that handles only memory coordination, moving tools and ReAct loop to the TUI client layer.

**Branch:** Create `phase4/architecture-refactor` off `main` (after pulling merged phase3)

**Approach:** Parallel subagents (hive-mind) where possible

---

## Pre-Implementation Setup

```bash
git checkout main
git pull origin main
git checkout -b phase4/architecture-refactor
```

---

## Parallel Execution Strategy

### Wave 1: Extract & Prepare (3 parallel agents)

| Agent | Focus | Files |
|-------|-------|-------|
| **Context Agent** | Extract context management | `core/context_manager.py` |
| **Memory Agent** | Extract memory handling | `core/memory_handler.py` |
| **TUI Prep Agent** | Prepare TUI handler stubs | `client/react_handler.py`, `client/idle_handler.py` stubs |

### Wave 2: Integration (2 parallel agents)

| Agent | Focus | Files |
|-------|-------|-------|
| **Core Agent** | Create PsycheCore, wire modules | `core/server.py`, `core/__init__.py` |
| **TUI Agent** | Implement TUI handlers, refactor app.py | `client/react_handler.py`, `client/idle_handler.py`, `client/app.py` |

### Wave 3: Testing & Polish (1-2 agents)

| Agent | Focus | Files |
|-------|-------|-------|
| **Test Agent** | Integration tests, PsycheClient | `client/psyche_client.py`, tests |
| **Cleanup Agent** | Remove dead code, update imports | Various |

---

## Wave 1: Extract & Prepare (3 parallel agents)

### Context Agent

**Creates:** `src/psyche/core/context_manager.py` (~200 lines)

Extract from `src/psyche/memory/server.py`:
- ContextCompactor integration and wrapper
- `_maybe_checkpoint()` logic
- Message history management
- `clear_context()` method
- `get_context_summary()` method

**Tests:** `tests/psyche/unit/test_context_manager.py`

**File ownership:** Only touches `core/context_manager.py` and its test file

---

### Memory Agent

**Creates:** `src/psyche/core/memory_handler.py` (~200 lines)

Extract from `src/psyche/memory/server.py`:
- `_retrieve_relevant_memories()`
- `_store_messages_to_mnemosyne()`
- `_save_to_local_fallback()`
- `_store_conversation_summary()`
- `_summarize_conversation()`
- `_handle_compaction_result()`
- `get_pending_fallback_files()`

**Tests:** `tests/psyche/unit/test_memory_handler.py`

**File ownership:** Only touches `core/memory_handler.py` and its test file

---

### TUI Prep Agent

**Creates stubs for:**
- `src/psyche/client/react_handler.py` - interface + stubs
- `src/psyche/client/idle_handler.py` - interface + stubs

**Analyzes:** What methods from server.py will move to these handlers

**File ownership:** Only touches new stub files in `client/`

---

## Wave 2: Integration (2 parallel agents)

### Core Agent

**Creates:** `src/psyche/core/server.py` (~300 lines) + `src/psyche/core/__init__.py`

PsycheCore class that coordinates:
- ContextManager instance
- MemoryHandler instance
- Elpis client connection
- Mnemosyne client connection (optional)
- Emotional state tracking
- System prompt building
- Importance scoring (wires existing `memory/importance.py`)

**Interface:**
```python
class PsycheCore:
    """Memory coordination layer - NO tools, NO ReAct."""
    async def add_message(role: str, content: str) -> None
    async def generate(tools: list = None) -> ChatResponse
    async def generate_stream(tools: list = None) -> AsyncIterator[str]
    async def retrieve_memories(query: str, n: int = 3) -> list
    async def store_memory(content: str, importance: float = 0.5) -> str
    async def get_emotion() -> EmotionState
    async def update_emotion(event: str, intensity: float = 1.0) -> EmotionState
    async def checkpoint() -> None
    async def consolidate() -> None
    async def shutdown() -> None
```

**Tests:** `tests/psyche/unit/test_psyche_core.py`

**File ownership:** `core/server.py`, `core/__init__.py`, test file

---

### TUI Agent

**Implements:**
- `src/psyche/client/react_handler.py` (~300 lines)
- `src/psyche/client/idle_handler.py` (~200 lines)

**react_handler.py** extracts from server.py:
- `_process_user_input()` → ReAct loop
- `_parse_tool_call()`
- `_execute_parsed_tool_call()`
- Tool iteration logic
- Interrupt handling

**idle_handler.py** extracts from server.py:
- `_generate_idle_thought()`
- `_get_reflection_prompt()`
- `_can_start_idle_thinking()`
- `_can_use_idle_tools()`
- `_validate_idle_tool_call()`
- `_is_safe_idle_path()`
- `_maybe_consolidate_memories()`

**Tests:** Unit tests for both handlers

**File ownership:** `client/react_handler.py`, `client/idle_handler.py`, test files

---

## Wave 3: Testing & Polish (2 parallel agents)

### Test Agent

**Creates:**
- `src/psyche/client/psyche_client.py` (~150 lines)
  - `PsycheClient` ABC
  - `LocalPsycheClient` (wraps PsycheCore)
  - `RemotePsycheClient` (stub for Phase 5)

**Updates:** `src/psyche/client/app.py`
- Use PsycheClient abstraction
- Integrate ReactHandler and IdleHandler
- Wire up callbacks

**Updates:** `src/psyche/cli.py`
- Add `--server` flag for remote mode (stubbed)

**Tests:** Integration tests for full flow

---

### Cleanup Agent

**Removes:** Dead code from `src/psyche/memory/server.py`
- Either slim to compatibility shim (~100 lines)
- Or remove entirely if PsycheCore replaces it

**Updates:** Imports across codebase
- Any file importing from `memory/server.py` → use `core/` instead

**Verifies:** All existing tests pass

---

## Architecture After Refactor

```
User Input → PsycheApp
                ↓
            ReactHandler (ReAct loop)
                ↓
            PsycheCore.generate() ← returns tool_calls
                ↓
            ToolEngine.execute() ← TUI executes tools
                ↓
            PsycheCore.add_message(tool_result)
                ↓
            Loop until no tool_calls
                ↓
            Display response
```

---

## Coordination File

Create: `scratchpad/reports/phase4/hive-mind-architecture-refactor.md`

Agents use this to:
- Claim file ownership
- Report progress
- Flag blockers
- Coordinate handoffs between waves

---

## File Changes Summary

### New Files (8)
| File | Lines | Purpose |
|------|-------|---------|
| `src/psyche/core/__init__.py` | ~20 | Package exports |
| `src/psyche/core/context_manager.py` | ~200 | Context/compaction |
| `src/psyche/core/memory_handler.py` | ~200 | Mnemosyne integration |
| `src/psyche/core/server.py` | ~300 | PsycheCore class |
| `src/psyche/client/react_handler.py` | ~300 | ReAct loop |
| `src/psyche/client/idle_handler.py` | ~200 | Dreaming/reflection |
| `src/psyche/client/psyche_client.py` | ~150 | Client abstraction |
| `tests/psyche/unit/test_psyche_core.py` | ~300 | Core tests |

### Modified Files (4)
| File | Changes |
|------|---------|
| `src/psyche/memory/server.py` | Slim down to ~100 lines (compatibility shim or remove) |
| `src/psyche/client/app.py` | Use PsycheClient, integrate handlers |
| `src/psyche/cli.py` | Add --server flag |
| Various test files | Update imports |

### Target Line Counts
- **PsycheCore (total):** ~500 lines (context_manager + memory_handler + server)
- **TUI handlers:** ~500 lines (react_handler + idle_handler)
- **Old server.py:** Remove or keep as thin compatibility layer

---

## Success Criteria

- [ ] PsycheCore < 500 lines, handles only memory coordination
- [ ] Tools executed by TUI layer, not Core
- [ ] ReAct loop runs in TUI, not Core
- [ ] LocalPsycheClient works for current TUI mode
- [ ] RemotePsycheClient stubbed for Phase 5
- [ ] All existing tests pass after refactor
- [ ] New unit tests for extracted modules
- [ ] Psyche TUI works identically to before refactor

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing functionality | Keep old server.py as compatibility shim initially |
| Async complexity in split | Careful interface design, clear ownership of event loops |
| Callback chain breaks | Test callbacks at each extraction step |
| Import cycles | Clean dependency graph: core → (no deps), client → core |
