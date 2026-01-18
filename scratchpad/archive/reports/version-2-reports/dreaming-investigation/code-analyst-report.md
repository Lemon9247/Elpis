# Code Analyst Report: IdleHandler Implementation Analysis

**Date:** 2026-01-17
**Analyst:** Claude Code (Code Analyst Agent)
**Status:** Complete

## Executive Summary

The IdleHandler implementation in `src/psyche/handlers/idle_handler.py` has several bugs, architectural issues, and limitations that would make it unsuitable for headless API server deployment without significant refactoring. The most critical issues are:

1. Race conditions in the idle loop timing
2. Missing error recovery mechanisms
3. Tight coupling to TUI assumptions
4. No persistence or resumability for dream state

---

## Bug Analysis

### Bug 1: Race Condition in Idle Loop Timing (Critical)

**Location:** `src/hermes/app.py` lines 220-253

```python
async def _run_idle_loop(self) -> None:
    # ...
    idle_interval = self._idle_handler.config.post_interaction_delay

    while True:
        # Wait for idle interval
        await asyncio.sleep(idle_interval / 2)  # <-- Bug: arbitrary division by 2

        if self.is_processing or not self._idle_handler.can_start_thinking():
            continue
```

**Problem:** The idle loop waits for `idle_interval / 2` seconds between checks, but this is arbitrary and creates a race condition. If `post_interaction_delay` is 60 seconds:
- Loop wakes every 30 seconds
- `can_start_thinking()` checks if 60 seconds have passed since last interaction
- This means thinking can start anywhere from 60-90 seconds after interaction

**Impact:** Unpredictable timing for dream initiation. Could start thinking while user is mid-typing.

**Fix:** Remove the `/2` division and use exact timing, or implement an event-based wake-up system.

---

### Bug 2: Interrupt Event Not Always Cleared (Medium)

**Location:** `src/psyche/handlers/idle_handler.py` lines 293-295, 311-315

```python
# Check for interrupt before starting generation
if self._interrupt_event.is_set():
    self._interrupt_event.clear()  # Cleared here
    logger.info("Idle thought interrupted before generation")
    return None

# ... later in the loop ...
if self._interrupt_event.is_set():
    self._interrupt_event.clear()  # Also cleared here
    logger.info("Idle thought interrupted during generation")
    interrupted = True
    break
```

**Problem:** The interrupt event is cleared in multiple places, but if an exception occurs between checks, the event may remain set, causing the next idle thought to be immediately cancelled.

**Impact:** Phantom interruptions - a single interrupt could cancel multiple thinking cycles.

**Fix:** Use a try/finally to ensure the event is cleared at the end of `generate_thought()`, or reset it at the start of each cycle.

---

### Bug 3: Disconnect During Tool Execution Not Handled (Medium)

**Location:** `src/psyche/handlers/idle_handler.py` lines 382-413

```python
try:
    formatted_call = {
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call.get("arguments", {})),
        }
    }
    tool_result = await self.tool_engine.execute_tool_call(formatted_call)
    # ... no check for connection status after tool execution ...
```

**Problem:** After executing a tool, the code doesn't verify the client is still connected before continuing. If the connection dropped during tool execution, the next iteration will fail with a cryptic error.

**Impact:** Poor error messages, potential for uncaught exceptions during idle thinking.

**Fix:** Add connection check after tool execution and before continuing the loop.

---

### Bug 4: Memory Consolidation Timing Drift (Low)

**Location:** `src/psyche/handlers/idle_handler.py` lines 644-649

```python
# Check if enough time has passed since last consolidation check
now = time.time()
time_since_last_check = now - self._last_consolidation_check
if time_since_last_check < self.config.consolidation_check_interval:
    return False

self._last_consolidation_check = now
```

**Problem:** The consolidation check timestamp is updated *after* the interval check, but the consolidation itself may take time. If consolidation takes 30 seconds and interval is 300 seconds, the actual interval becomes 330+ seconds.

**Impact:** Consolidation timing slowly drifts. Not critical but inaccurate.

**Fix:** Update timestamp at the start of the check, not after.

---

### Bug 5: SAFE_IDLE_TOOLS Mismatch with Tool Engine (Low)

**Location:** `src/psyche/handlers/idle_handler.py` lines 30-35

```python
SAFE_IDLE_TOOLS: FrozenSet[str] = frozenset({
    "read_file",
    "list_directory",
    "search_codebase",
    "recall_memory",
})
```

**Problem:** The `recall_memory` tool is listed as safe, but it's only registered if Mnemosyne is connected (see `src/hermes/cli.py` lines 236-237). If Mnemosyne is disabled, attempting to use `recall_memory` during idle will fail.

**Impact:** Tool execution failures when Mnemosyne is disabled.

**Fix:** Dynamically build SAFE_IDLE_TOOLS based on actually registered tools, or validate tool availability before allowing.

---

## Architectural Issues

### Issue 1: Tight Coupling to TUI Callbacks

The `generate_thought()` method accepts callbacks (`on_token`, `on_tool_call`, `on_thought`) that assume real-time UI streaming:

```python
async def generate_thought(
    self,
    on_token: Optional[Callable[[str], None]] = None,
    on_tool_call: Optional[Callable[[str, Dict[str, Any], Optional[Dict[str, Any]]], None]] = None,
    on_thought: Optional[Callable[[ThoughtEvent], None]] = None,
) -> Optional[str]:
```

**Problem for Headless:** In a headless API server:
- No UI to stream tokens to
- Tool calls need to be logged/stored, not displayed
- Thoughts should be persisted, not just displayed

**Recommendation:** Introduce an event emitter pattern or abstract observer interface that can be implemented differently for TUI vs headless modes.

---

### Issue 2: No Dream State Persistence

**Problem:** Dreams are ephemeral - generated thoughts are not persisted anywhere. If the server restarts:
- All dream history is lost
- No way to review what Psyche was "thinking" about
- No dream journal functionality

**Recommendation for Headless:**
- Store dreams in Mnemosyne as a special memory type
- Add a "dream journal" API endpoint
- Consider dream importance scoring for consolidation

---

### Issue 3: Workspace-Centric Dream Content

The reflection prompts assume a filesystem workspace:

```python
prompts = [
    base_instruction + "What exists in this workspace? Think about what you're curious about, then explore.",
    # ...
]
```

**Problem for Headless:** A headless API server may not have a meaningful workspace:
- Server might run in a container with no user files
- Workspace could be `/tmp` or empty
- Dreams about "workspace" are meaningless

**Recommendation:** Create different dream modes:
- `workspace` mode: current behavior for TUI
- `memory` mode: reflect on stored memories
- `self` mode: introspective thoughts about emotional state
- `creative` mode: generate poetry, stories, ideas

---

### Issue 4: Single Compactor Shared State

The `IdleHandler` shares a `ContextCompactor` with the `ReactHandler`:

```python
# From src/hermes/cli.py
compactor = core._context.compactor

react_handler = ReactHandler(
    # ...
    compactor=compactor,
)

idle_handler = IdleHandler(
    # ...
    compactor=compactor,  # Same instance!
)
```

**Problem:** Idle thoughts are added to the same conversation context as user interactions. This means:
- Dreams pollute the user conversation history
- User might see dream artifacts in responses
- Context window fills up with dream content

**Recommendation:** Use a separate compactor for dreams, or mark dream messages distinctly so they can be filtered when responding to users.

---

### Issue 5: No Rate Limiting Persistence

Rate limiting state is in-memory only:

```python
self._last_idle_tool_use: float = 0.0
self._last_user_interaction: float = 0.0
self._last_consolidation_check: float = 0.0
```

**Problem for Headless:** If the server restarts, rate limits reset:
- Could immediately trigger expensive consolidation
- Tool cooldowns lost
- Interaction timing reset

**Recommendation:** Persist timing state to disk or Mnemosyne for headless mode.

---

## Suitability for Headless API Mode

### What Works
- Core generation logic is async and could work headlessly
- Tool execution is well-abstracted
- Rate limiting logic is sound (just not persistent)
- Safety constraints (path validation, tool filtering) are appropriate

### What Doesn't Work
- Callback-based token streaming (needs refactoring)
- Workspace-centric dream prompts (meaningless in headless)
- No dream persistence (critical for headless)
- Shared conversation context (dreams pollute user interactions)
- No mechanism for external wake-up (API request should wake from dream)

### Recommended Changes for Headless

1. **Introduce DreamMode enum:**
   ```python
   class DreamMode(Enum):
       WORKSPACE = "workspace"  # Explore filesystem
       MEMORY = "memory"        # Reflect on memories
       INTROSPECTIVE = "introspective"  # Self-reflection
       CREATIVE = "creative"    # Generate creative content
   ```

2. **Add dream persistence:**
   ```python
   async def persist_dream(self, thought: ThoughtEvent) -> str:
       """Store dream in Mnemosyne with dream metadata."""
       return await self.mnemosyne_client.store_memory(
           content=thought.content,
           memory_type="dream",
           tags=["dream", thought.thought_type],
       )
   ```

3. **Separate dream context:**
   ```python
   def __init__(self, ...):
       # Use separate compactor for dream context
       self._dream_compactor = ContextCompactor(max_tokens=4000)
   ```

4. **Event-based wake protocol:**
   ```python
   async def wake_for_request(self) -> bool:
       """Gracefully interrupt dreaming for an incoming request."""
       self._wake_event.set()
       # Wait for current dream to complete or timeout
       await asyncio.wait_for(self._thinking_complete.wait(), timeout=5.0)
       return True
   ```

5. **Replace callbacks with events:**
   ```python
   class DreamEvent:
       type: str  # "token", "tool_start", "tool_complete", "thought_complete"
       data: Any

   async def generate_thought(self) -> AsyncIterator[DreamEvent]:
       """Generate dream as a stream of events."""
       # Yield events instead of calling callbacks
   ```

---

## Specific Line-by-Line Issues

| File | Line | Issue | Severity |
|------|------|-------|----------|
| `idle_handler.py` | 183 | `asyncio.Event()` created in `__init__` - not safe if instance reused across event loops | Low |
| `idle_handler.py` | 281 | `reflection_messages` grows unbounded in tool loop (no pruning) | Medium |
| `idle_handler.py` | 294 | Double-clear of interrupt event (redundant) | Low |
| `idle_handler.py` | 391 | `record_tool_use()` called before verifying tool succeeded | Low |
| `idle_handler.py` | 430 | `maybe_consolidate()` called even on max iterations without successful thought | Low |
| `app.py` | 231 | `idle_interval / 2` is unexplained magic number | Medium |
| `app.py` | 243-244 | No backoff on generation failure - will retry immediately | Medium |

---

## Conclusion

The IdleHandler is functional for the current TUI use case but requires significant refactoring for headless API deployment. The core issues are:

1. **Immediate fixes needed:** Race condition in timing, interrupt event handling
2. **Architectural changes needed:** Dream persistence, separate context, event-based streaming
3. **New features needed:** Multiple dream modes, wake protocol, dream journal API

Estimated effort: 2-3 sessions for bug fixes, 4-5 sessions for headless adaptation.

---

*Report generated by Code Analyst Agent*
