# Phase 4: Architecture Refactor - Hive Mind Coordination

**Date:** 2026-01-16
**Branch:** `phase4/architecture-refactor`
**Goal:** Extract Psyche Core (~500 lines) from MemoryServer (1828 lines)

---

## File Ownership

Agents MUST only modify files they own. Check this table before writing.

### Wave 1 Ownership

| Agent | Owned Files | Status |
|-------|-------------|--------|
| **Context Agent** | `src/psyche/core/context_manager.py`, `tests/psyche/unit/test_context_manager.py` | Complete |
| **Memory Agent** | `src/psyche/core/memory_handler.py`, `tests/psyche/unit/test_memory_handler.py` | Complete |
| **TUI Prep Agent** | `src/psyche/client/react_handler.py` (stub), `src/psyche/client/idle_handler.py` (stub) | Complete |

### Wave 2 Ownership

| Agent | Owned Files | Status |
|-------|-------------|--------|
| **Core Agent** | `src/psyche/core/server.py`, `src/psyche/core/__init__.py`, `tests/psyche/unit/test_psyche_core.py` | Pending |
| **TUI Agent** | `src/psyche/client/react_handler.py` (impl), `src/psyche/client/idle_handler.py` (impl), `tests/psyche/unit/test_react_handler.py`, `tests/psyche/unit/test_idle_handler.py` | Pending |

### Wave 3 Ownership

| Agent | Owned Files | Status |
|-------|-------------|--------|
| **Test Agent** | `src/psyche/client/psyche_client.py`, `src/psyche/client/app.py`, `src/psyche/cli.py`, integration tests | Pending |
| **Cleanup Agent** | `src/psyche/memory/server.py` (removal), import updates | Pending |

---

## Agent Reports

### Wave 1

#### Context Agent
- **Status:** Complete
- **Report:**
  - Created `src/psyche/core/context_manager.py` (72 lines, 100% test coverage)
  - Created `tests/psyche/unit/test_context_manager.py` (28 tests, all passing)
  - **Key classes:**
    - `ContextConfig`: Dataclass for configuration (max_tokens, reserve_tokens, checkpoint settings)
    - `ContextManager`: Wraps ContextCompactor with higher-level API
  - **Methods implemented:**
    - `set_system_prompt()` / `update_system_prompt()`: System prompt management
    - `add_message()` / `add_raw_message()`: Add messages with compaction triggering
    - `get_api_messages()`: Format messages for API calls
    - `should_checkpoint()` / `get_checkpoint_messages()`: Checkpoint support
    - `clear()`: Clear context (preserves system prompt)
    - `get_summary()`: Return context statistics
    - `mark_important()`: Mark messages to prevent compaction
  - **Notes for Core Agent:**
    - ContextManager is designed to be used as a component of a larger server
    - It wraps ContextCompactor from `psyche.memory.compaction`
    - Compaction is handled automatically when token limits are exceeded
    - The `mnemosyne_client` parameter is accepted but not used directly by ContextManager (checkpoint storage should be handled by the caller or a separate MemoryHandler)

#### Memory Agent
- **Status:** Complete
- **Report:**
  - Created `src/psyche/core/memory_handler.py` (174 lines, 95% test coverage)
  - Created `tests/psyche/unit/test_memory_handler.py` (47 tests, all passing)
  - **Key classes:**
    - `MemoryHandlerConfig`: Dataclass for configuration (auto-retrieval, auto-storage settings)
    - `MemoryHandler`: Handles long-term memory storage via Mnemosyne with fallback
  - **Methods implemented:**
    - `retrieve_relevant(query, n)`: Retrieve memories from Mnemosyne for context injection
    - `format_memories_for_context(memories)`: Format retrieved memories for conversation context
    - `store_messages(messages, emotional_context)`: Store messages to Mnemosyne
    - `store_summary(summary, emotional_context)`: Store conversation summary as semantic memory
    - `summarize_conversation(messages)`: Use Elpis to generate conversation summary
    - `store_conversation_summary(messages, emotional_context)`: Generate and store summary
    - `handle_compaction(result, emotional_context)`: Handle compaction with staged message storage
    - `save_to_fallback(messages, reason)`: Save to local JSON when Mnemosyne unavailable
    - `get_pending_fallbacks()`: List pending fallback files for recovery
    - `flush_staged_messages(emotional_context)`: Immediately flush staged messages (for shutdown)
    - `staged_message_count` property: Track staged messages
    - `clear_staged_messages()`: Clear staged messages without storing
    - `is_mnemosyne_available` property: Check Mnemosyne availability
  - **Design decisions:**
    1. Uses dependency injection for testability (mnemosyne_client, elpis_client)
    2. Implements delayed storage: messages staged for one compaction cycle before storage
    3. Fallback storage creates timestamped JSON files in configurable directory
    4. System messages are always filtered out from storage
    5. Emotional context is optional and passed through to Mnemosyne
    6. Summarization uses Elpis with low temperature (0.3) for consistency
  - **Notes for Core Agent:**
    - MemoryHandler does NOT integrate with ContextManager directly - the Core Agent should coordinate between them
    - `handle_compaction()` expects a `CompactionResult` from the compactor
    - The handler tracks `_staged_messages` internally - call `flush_staged_messages()` on shutdown
    - Use `is_mnemosyne_available` property to check before operations (returns False if client is None or disconnected)

#### TUI Prep Agent
- **Status:** Complete
- **Report:**
  - Created `src/psyche/client/react_handler.py` (stub, ~150 lines)
  - Created `src/psyche/client/idle_handler.py` (stub, ~250 lines)
  - **Interfaces defined:**

  **ReactHandler (react_handler.py):**
  - `ReactConfig`: Configuration dataclass (max_tool_iterations, max_tool_result_chars, generation_timeout, emotional_modulation)
  - `ToolCallResult`: Dataclass for tracking tool execution results
  - `ReactHandler` class with methods:
    - `__init__(elpis_client, tool_engine, context_manager, memory_handler, config)`: Constructor with DI
    - `process_input(text, on_token, on_tool_call, on_response, on_thought)`: Main ReAct loop entry point
    - `parse_tool_call(response_text)`: Extract tool call JSON from LLM response
    - `execute_tool(tool_call, on_tool_call)`: Execute a tool and return result
    - `interrupt()` / `clear_interrupt()`: Interrupt handling
    - `is_processing` property: Check if currently processing

  **IdleHandler (idle_handler.py):**
  - `SAFE_IDLE_TOOLS`: FrozenSet of allowed read-only tools (read_file, list_directory, search_codebase, recall_memory)
  - `SENSITIVE_PATH_PATTERNS`: FrozenSet of blocked path patterns (.ssh, .env, credentials, etc.)
  - `IdleConfig`: Configuration dataclass (post_interaction_delay, idle_tool_cooldown_seconds, startup_warmup_seconds, max_idle_tool_iterations, etc.)
  - `ThoughtEvent`: Dataclass for thought events (mirrors server.py)
  - `IdleHandler` class with methods:
    - `__init__(elpis_client, context_manager, tool_engine, memory_handler, mnemosyne_client, config)`: Constructor with DI
    - `can_start_thinking()`: Check if idle thinking can begin (post-interaction delay)
    - `can_use_tools()`: Check rate limiting for idle tools
    - `generate_thought(on_token, on_tool_call, on_thought)`: Main entry point for idle thinking
    - `get_reflection_prompt()`: Get a reflection prompt
    - `validate_tool_call(tool_call)`: Validate tool call is safe
    - `is_safe_path(path)`: Check path safety
    - `maybe_consolidate()`: Run memory consolidation if needed
    - `record_user_interaction()` / `record_tool_use()`: Timing helpers
    - `interrupt()` / `clear_interrupt()`: Interrupt handling
    - `is_thinking` property: Check if generating thoughts

  - **Key methods from server.py that will move:**
    - ReactHandler: `_process_user_input()` (lines 663-843), `_parse_tool_call()` (lines 850-909), `_execute_parsed_tool_call()` (lines 911-977)
    - IdleHandler: `_generate_idle_thought()` (lines 1104-1276), `_get_reflection_prompt()` (lines 1278-1313), `_can_start_idle_thinking()` (lines 979-1001), `_can_use_idle_tools()` (lines 1003-1035), `_validate_idle_tool_call()` (lines 1078-1102), `_is_safe_idle_path()` (lines 1037-1076), `_maybe_consolidate_memories()` (lines 1416-1477)

  - **Design decisions for Wave 2 TUI Agent:**
    1. Both handlers use dependency injection for testability (elpis_client, tool_engine, context_manager, etc.)
    2. Callbacks are passed to methods rather than constructor to allow flexible UI integration
    3. Configuration is via dataclasses mirroring ServerConfig fields
    4. `ThoughtEvent` is duplicated in idle_handler.py to avoid circular imports (could be moved to shared module later)
    5. Safety constants (SAFE_IDLE_TOOLS, SENSITIVE_PATH_PATTERNS) are module-level frozensets for immutability
    6. Interrupt handling is explicit via `interrupt()` / `clear_interrupt()` methods
    7. All async methods that do inference should support interruption and timeout

---

### Wave 2

#### Core Agent
- **Status:** Not started
- **Report:** (To be filled by agent)

#### TUI Agent
- **Status:** Not started
- **Report:** (To be filled by agent)

---

### Wave 3

#### Test Agent
- **Status:** Not started
- **Report:** (To be filled by agent)

#### Cleanup Agent
- **Status:** Not started
- **Report:** (To be filled by agent)

---

## Blockers & Questions

*Agents should add any blockers or questions here for coordination.*

---

## Key References

- **Source file:** `src/psyche/memory/server.py` (1828 lines)
- **Compaction:** `src/psyche/memory/compaction.py`
- **Importance:** `src/psyche/memory/importance.py`
- **Reasoning:** `src/psyche/memory/reasoning.py`
- **Tool engine:** `src/psyche/tools/tool_engine.py`
- **App:** `src/psyche/client/app.py`

---

## Method Extraction Map

### To Context Manager (from server.py)
- `_maybe_checkpoint()` - lines 581-610
- `clear_context()` - lines 1822-1827
- `get_context_summary()` - lines 1400-1414
- ContextCompactor integration

### To Memory Handler (from server.py)
- `_retrieve_relevant_memories()` - lines 612-661
- `_store_messages_to_mnemosyne()` - lines 1574-1629
- `_save_to_local_fallback()` - lines 1631-1681
- `_store_conversation_summary()` - lines 1534-1572
- `_summarize_conversation()` - lines 1479-1532
- `_handle_compaction_result()` - lines 1695-1734
- `get_pending_fallback_files()` - lines 1683-1693

### To React Handler (from server.py)
- `_process_user_input()` - lines 663-843
- `_parse_tool_call()` - lines 850-909
- `_execute_parsed_tool_call()` - lines 911-977

### To Idle Handler (from server.py)
- `_generate_idle_thought()` - lines 1104-1276
- `_get_reflection_prompt()` - lines 1278-1313
- `_can_start_idle_thinking()` - lines 979-1001
- `_can_use_idle_tools()` - lines 1003-1035
- `_validate_idle_tool_call()` - lines 1078-1102
- `_is_safe_idle_path()` - lines 1037-1076
- `_maybe_consolidate_memories()` - lines 1416-1477
