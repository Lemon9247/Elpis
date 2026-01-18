# Wake Protocol Design Document

**Agent:** Wake Protocol Designer
**Date:** 2026-01-17
**Status:** Complete

## Executive Summary

This document defines the protocol for cleanly "waking up" Psyche from a dream state when a client connects to the headless API server. The protocol must ensure:
1. No corrupted state from interrupted operations
2. Fast response time for connecting clients
3. Preservation of dream insights
4. Graceful context transition

---

## Current Implementation Analysis

### Existing Interrupt Mechanism

The current `IdleHandler` uses an `asyncio.Event()` for interrupt signaling:

```python
# From idle_handler.py
self._interrupt_event = asyncio.Event()

def interrupt(self) -> bool:
    if self._is_thinking:
        self._interrupt_event.set()
        return True
    return False
```

**Interrupt Check Points:**
1. Before starting generation (line 293)
2. During token streaming - after each token (line 311)
3. After generation completes, before tool execution (not currently present for idle mode)

**Strengths:**
- Non-blocking cooperative interruption
- Checks interrupt flag during streaming (responsive)
- Clean state tracking via `_is_thinking` flag

**Weaknesses:**
- No state preservation - interrupted dreams are simply discarded
- No "safe stopping points" defined for tool operations
- No dream journal / insight persistence
- Interrupt clearing happens in multiple places (potential race conditions)

---

## Proposed Wake-Up Protocol

### State Machine

```
                    +------------------+
                    |                  |
         +--------->|     SLEEPING     |<---------+
         |          |   (baseline)     |          |
         |          +--------+---------+          |
         |                   |                    |
         |          idle_timeout                  |
         |                   |                    |
         |                   v                    |
         |          +--------+---------+          |
         |          |                  |          |
         |          |    DREAMING      |          |
         |          | (active thought) |          |
         |          +--------+---------+          |
         |                   |                    |
         |     client_connect|                    |
         |                   v                    |
         |          +--------+---------+          |
         |          |                  |          |
         |          |     WAKING       |          |
         |          | (transition)     |          |
         |          +--------+---------+          |
         |                   |                    |
         |        context_ready                   |
         |                   |                    |
         |                   v                    |
         |          +--------+---------+          |
         |          |                  |          |
         +----------+      AWAKE       +----------+
        client_     |   (serving)      |   client_
        disconnect  +------------------+   timeout
```

### State Definitions

| State | Description | Allowed Operations |
|-------|-------------|-------------------|
| SLEEPING | Idle, no client, not dreaming | Start dream timer |
| DREAMING | Actively generating/exploring | Inference, memory ops, tool calls |
| WAKING | Transition period after interrupt | Save dream state, clear context |
| AWAKE | Client connected, serving requests | Full request handling |

### Wake-Up Protocol Steps

```
WAKE-UP PROTOCOL
================

Phase 1: Signal Receipt (< 1ms)
-------------------------------
1. Client connection detected by API server
2. Set interrupt_event flag immediately
3. Record wake_request_timestamp
4. Enter WAKING state

Phase 2: Safe Stopping (10-500ms target)
----------------------------------------
1. Dream loop detects interrupt_event.is_set()
2. Identify current operation type:
   a) Token generation -> stop after current token
   b) Tool execution -> wait for completion (with timeout)
   c) Memory operation -> wait for completion (with timeout)
3. Capture DreamState snapshot:
   - tokens_generated: list of generated tokens
   - current_prompt: reflection prompt used
   - tool_calls_made: list of executed tools
   - insights_partial: any partial conclusions
   - emotional_context: current valence/arousal
4. Signal safe_stop_complete

Phase 3: State Preservation (< 50ms)
------------------------------------
1. If dream produced meaningful content:
   a) Calculate dream_significance score
   b) If significance > threshold:
      - Store to dream journal
      - Tag with emotional context
2. Clear dream-specific context
3. Preserve emotional state (dreams can affect mood)
4. Signal state_preserved

Phase 4: Context Switch (< 10ms)
--------------------------------
1. Load client greeting/system prompt
2. Initialize fresh response context
3. Enter AWAKE state
4. Signal ready_for_client
5. Begin processing client request
```

---

## State to Preserve When Interrupted

### Dream State Snapshot

```python
@dataclass
class DreamState:
    """Snapshot of dream state at interruption."""

    # What was being explored
    prompt: str
    tokens_generated: List[str]

    # Tools used during dream
    tool_calls: List[ToolCallRecord]

    # Partial insights/conclusions
    partial_content: str

    # Emotional context
    emotional_state: EmotionalState

    # Timing
    dream_start: float
    interrupt_time: float

    # Progress markers
    iteration: int
    max_iterations: int
```

### What to Keep vs. Discard

| Keep | Discard |
|------|---------|
| Emotional state changes | Incomplete tool results |
| Significant insights | Partial generation buffer |
| Tool call history (for audit) | Reflection prompt text |
| Dream duration stats | Pending memory operations |

---

## Safe Stopping Points

### During Inference (Token Generation)

**Safe points:**
- After any complete token (current implementation)
- After a sentence boundary (punctuation + space)
- After a paragraph (double newline)

**Unsafe points:**
- Mid-token (would corrupt output)
- During embedding computation

**Implementation:**

```python
# Check interrupt at safe points only
async for token in self.client.generate_stream(...):
    response_tokens.append(token)

    # Safe stopping point: after each token
    if self._interrupt_event.is_set():
        # Capture what we have
        self._dream_state = DreamState(
            tokens_generated=response_tokens.copy(),
            partial_content="".join(response_tokens),
            interrupt_time=time.time(),
            ...
        )
        return None
```

### During Tool Execution

**Problem:** Tool calls may involve I/O, file operations, or memory writes that shouldn't be interrupted mid-execution.

**Solution:** Tool operations are atomic - wait for completion with timeout.

```python
async def execute_tool_with_interrupt_support(self, tool_call, timeout=5.0):
    """Execute tool with interrupt awareness."""

    # Check interrupt BEFORE starting
    if self._interrupt_event.is_set():
        return None  # Skip tool entirely

    # Execute with timeout
    try:
        async with asyncio.timeout(timeout):
            result = await self.tool_engine.execute_tool_call(tool_call)

        # Tool completed - record it even if we're waking
        self._record_tool_call(tool_call, result)
        return result

    except asyncio.TimeoutError:
        # Tool took too long - force stop
        logger.warning(f"Tool {tool_call['name']} timed out during wake")
        return {"error": "interrupted", "timeout": True}
```

### During Memory Operations

Memory operations (via MnemosyneClient) should complete atomically:

```python
# Memory ops are protected by the client's session lock
# They're already atomic at the MCP level

# For consolidation during dreams:
if self._interrupt_event.is_set():
    logger.info("Skipping consolidation due to wake request")
    return False  # Skip consolidation entirely
```

---

## Dream Journal

### Purpose

Persist insights from dreams so they can:
1. Inform future conversations
2. Build long-term "subconscious" knowledge
3. Track emotional patterns over time

### Structure

```python
@dataclass
class DreamEntry:
    """A single dream journal entry."""

    # Identification
    dream_id: str
    timestamp: datetime

    # Content
    content: str  # The dream thought/reflection
    significance: float  # 0.0 - 1.0, computed score

    # Context
    emotional_valence: float
    emotional_arousal: float

    # Metadata
    duration_seconds: float
    was_interrupted: bool
    tool_calls_made: int

    # Tags for retrieval
    topics: List[str]  # Auto-extracted topics
```

### Storage Strategy

Dreams go to Mnemosyne as a special memory type:

```python
async def persist_dream(self, dream_state: DreamState):
    """Persist significant dream content to memory."""

    # Calculate significance
    significance = self._calculate_dream_significance(dream_state)

    if significance < 0.3:  # Not worth saving
        logger.debug("Dream below significance threshold, discarding")
        return

    # Store as dream memory type
    await self.mnemosyne_client.store_memory(
        content=dream_state.partial_content,
        summary=f"Dream reflection: {dream_state.partial_content[:100]}...",
        memory_type="dream",  # New type for dreams
        tags=["dream", "reflection", "autonomous"],
        emotional_context={
            "valence": dream_state.emotional_state.valence,
            "arousal": dream_state.emotional_state.arousal,
        },
        metadata={
            "was_interrupted": True,
            "significance": significance,
            "duration": dream_state.interrupt_time - dream_state.dream_start,
        }
    )
```

### Significance Calculation

```python
def _calculate_dream_significance(self, dream_state: DreamState) -> float:
    """Calculate how significant/valuable a dream thought was."""

    score = 0.0

    # Length contributes to significance
    content_len = len(dream_state.partial_content)
    if content_len > 200:
        score += 0.2
    if content_len > 500:
        score += 0.2

    # Tool usage indicates active exploration
    if dream_state.tool_calls:
        score += 0.2

    # Emotional intensity suggests meaningful content
    emotional_intensity = abs(dream_state.emotional_state.valence) + \
                         abs(dream_state.emotional_state.arousal)
    score += min(0.2, emotional_intensity * 0.1)

    # Completed thoughts (has conclusion markers)
    if any(marker in dream_state.partial_content for marker in
           [". ", ".\n", "I think", "I notice", "interesting"]):
        score += 0.2

    return min(1.0, score)
```

---

## Context Switch: Dream to Client

### The Problem

Dream context contains:
- Reflection prompts ("This is your private thinking time...")
- Tool results from exploration
- Internal monologue

This shouldn't leak into client interactions.

### Solution: Clean Context Boundary

```python
async def switch_to_client_context(self):
    """Prepare context for client interaction."""

    # 1. Preserve emotional state (dreams affect mood)
    emotional_carryover = await self._get_current_emotion()

    # 2. Clear dream-specific messages
    self._context_manager.clear()

    # 3. Set client-facing system prompt
    self._context_manager.set_system_prompt(
        self._get_client_system_prompt()
    )

    # 4. Optionally inject dream insights as background context
    if self._recent_dream_insights:
        self._context_manager.add_message(
            "system",
            f"[Background context from recent reflection]\n{self._recent_dream_insights}"
        )

    # 5. Apply emotional carryover
    # (Emotional state persists - if Psyche was content while dreaming,
    # that positive state carries into the conversation)
```

### What Transfers Between Contexts

| From Dreams | To Client Context |
|-------------|-------------------|
| Emotional state | Yes - affects response tone |
| Specific thoughts | No - private |
| Significant insights | Yes - as background context |
| Tool call history | No |
| Partial generations | No |

---

## Maximum Latency Guarantees

### Target Latencies

| Phase | Target | Maximum | Notes |
|-------|--------|---------|-------|
| Signal Receipt | < 1ms | 5ms | Just setting an event flag |
| Safe Stopping | 10-100ms | 500ms | Depends on operation in progress |
| State Preservation | < 50ms | 100ms | Memory store is async |
| Context Switch | < 10ms | 50ms | Mostly in-memory |
| **Total Wake Time** | **< 170ms** | **655ms** | End-to-end |

### Guarantees

1. **Soft guarantee:** Client receives response within 200ms of connection
2. **Hard guarantee:** Client receives response within 1 second (with forced abort)

### Timeout Escalation

```python
async def wake_with_timeout(self):
    """Wake up with escalating timeouts."""

    # Signal interrupt
    self._interrupt_event.set()

    # Wait for cooperative stop (preferred)
    try:
        async with asyncio.timeout(0.5):  # 500ms cooperative
            await self._wait_for_safe_stop()
            return
    except asyncio.TimeoutError:
        logger.warning("Cooperative stop timed out, forcing abort")

    # Force abort (may lose dream state)
    self._force_abort()
    self._dream_state = None  # Can't preserve
```

---

## Edge Cases

### 1. Interrupt During Tool Execution

**Scenario:** Client connects while a read_file tool is running.

**Handling:**
- Wait for tool completion (up to 5s timeout)
- Record result for audit trail
- Then wake

```python
if self._tool_in_progress:
    try:
        async with asyncio.timeout(5.0):
            await self._current_tool_task
    except asyncio.TimeoutError:
        self._current_tool_task.cancel()
```

### 2. Interrupt During Memory Consolidation

**Scenario:** Client connects during maybe_consolidate().

**Handling:**
- Consolidation is checked via `should_consolidate()` first
- If already running, let it complete (it's important)
- Set flag to skip next consolidation check

### 3. Rapid Connect/Disconnect

**Scenario:** Client connects, disconnects, another connects quickly.

**Handling:**
- Each connect triggers wake
- Maintain wake lock to prevent re-sleeping during rapid cycles
- Minimum awake duration: 5 seconds after any client activity

```python
self._wake_lock_until = time.time() + 5.0

def can_start_dreaming(self) -> bool:
    if time.time() < self._wake_lock_until:
        return False
    return True
```

### 4. Interrupt During Streaming to... No One

**Scenario:** Headless mode, dream generates tokens, but no UI to stream to.

**Handling:**
- Tokens still collected in buffer
- Buffer becomes dream state on interrupt
- No streaming callback in headless mode

### 5. Multiple Simultaneous Wake Requests

**Scenario:** Two clients try to connect at once (even though we said one at a time).

**Handling:**
- First interrupt wins
- Second request waits for AWAKE state
- Queue at API server level, not dream level

```python
# At API server level
async with self._client_lock:
    if self._state == DreamState.DREAMING:
        await self._wake()
    # Now in AWAKE state
    return await self._handle_client(request)
```

### 6. Dream Generates Something Actually Important

**Scenario:** Dream discovers a bug or has a genuine insight.

**Handling:**
- Significance calculation catches this
- High-significance dreams always persist
- Consider: "dream digest" sent to user on next connect?

---

## Implementation Approach

### Phase 1: Core Infrastructure (1 session)

1. Add `DreamState` dataclass to `idle_handler.py`
2. Modify interrupt handling to capture state
3. Add `_dream_state` property to IdleHandler

### Phase 2: Safe Stopping Points (1 session)

1. Refactor tool execution to be interrupt-aware
2. Add timeout wrappers for async operations
3. Add interrupt checks at tool boundaries

### Phase 3: Dream Journal (1 session)

1. Add "dream" memory type to Mnemosyne
2. Implement `persist_dream()` method
3. Add significance calculation
4. Wire up auto-persistence on interrupt

### Phase 4: Context Switching (1 session)

1. Implement `switch_to_client_context()`
2. Add emotional state carryover
3. Add dream insight injection (optional)
4. Test clean separation

### Phase 5: API Server Integration (separate investigation)

1. Wire wake protocol into request handler
2. Add latency monitoring
3. Implement timeout escalation
4. Add wake lock mechanism

---

## Testing Strategy

### Unit Tests

- Interrupt at each safe stopping point
- Dream state capture accuracy
- Significance calculation edge cases
- Timeout behavior

### Integration Tests

- Full wake cycle with mock client
- Context isolation verification
- Emotional state carryover
- Dream journal persistence

### Performance Tests

- Wake latency measurements
- Memory/CPU during rapid wake cycles
- Dream journal size growth over time

---

## Open Questions for Team Discussion

1. **Dream depth levels?**
   - Should there be "light sleep" (can wake instantly) vs "deep sleep" (needs full protocol)?

2. **Dream insight sharing?**
   - Should clients be informed of relevant dream insights?
   - Privacy implications?

3. **Dream scheduling?**
   - Fixed intervals vs. emotional need vs. memory pressure?

4. **Metrics and monitoring?**
   - What telemetry do we need for dream health?

---

## Summary

The wake protocol provides a structured approach to transitioning Psyche from dreaming to serving clients with:

- **Clear state machine** with well-defined transitions
- **Cooperative interruption** with timeout fallback
- **State preservation** via dream journal
- **Clean context boundaries** between dream and client
- **Latency guarantees** within 200ms typical, 1s maximum

The protocol builds on the existing `asyncio.Event()` mechanism while adding the infrastructure needed for state preservation and graceful transitions.
