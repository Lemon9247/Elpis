# Dreaming Investigation: Synthesis Report

**Date:** 2026-01-17
**Coordinator:** Claude Code (Main Instance)
**Status:** Complete (with post-synthesis refinements)

## Executive Summary

Four agents investigated how dreaming should work when Psyche runs as a headless API server. The investigation produced a coherent architecture that addresses bugs in the current implementation while introducing new concepts for headless mode.

**Key Insight:** When Psyche has no workspace to explore (headless mode), her dream world becomes her **Memory Palace** - an introspective space built from accumulated experiences, not external exploration.

**Post-Synthesis Refinement:** Dreams should use **no tools at all**. Memory is provided as context *before* generation, not retrieved via tool calls during. This makes dreams purely generative - internal processing with no external reach.

---

## Critical Design Constraint: Compatibility

Psyche will serve as an API for other agent harnesses (OpenCode, etc.). The dreaming system must be **transparent to clients**:

- Standard API clients work without knowing about dreams
- Wake-on-request is automatic (no special "wake up" call needed)
- Dream-related endpoints are optional extensions
- Core request/response behavior is unchanged

```
Standard API (any harness):     Dream API (Psyche-aware only):
  POST /chat                      GET  /dream/status
  GET  /emotion                   POST /dream/wake (optional)
  POST /memory/*                  GET  /dream/journal
```

---

## Findings Summary

### 1. Current Implementation Issues (Code Analyst)

**Critical Bugs:**
| Bug | Location | Impact |
|-----|----------|--------|
| Race condition in timing | `app.py:231` | Dreams start 60-90s after interaction, not exactly at config |
| Phantom interruptions | `idle_handler.py:293-315` | Single interrupt can cancel multiple cycles |
| Missing disconnect detection | `idle_handler.py:382-413` | Cryptic errors if connection drops during tool execution |
| SAFE_IDLE_TOOLS mismatch | `idle_handler.py:30-35` | `recall_memory` fails when Mnemosyne disabled |

**Architectural Issues:**
- Dreams pollute user conversation context (shared compactor)
- No dream persistence (dreams are ephemeral, lost on restart)
- Workspace-centric prompts meaningless in headless mode
- Tight coupling to TUI callbacks

### 2. Dream World Concept (Dream World Designer)

**The Memory Palace:**
- When headless, Psyche explores her memories, not filesystems
- Dreams are introspective, not exploratory

**Four Dream Types Based on Arousal:**
| Type | Arousal | Activity |
|------|---------|----------|
| Wandering | Low | Free association through memories |
| Processing | Medium | Emotional integration |
| Synthesis | High | Pattern finding across experiences |
| Reflection | Variable | Self-focused introspection |

**New Mnemosyne Methods Needed:**
- `get_random_memories(status, n)` - Random selection for wandering
- `recall_by_emotion(quadrant, n)` - Match by emotional signature
- `get_related_memories(memory_id, n)` - Follow association links

### Post-Synthesis Refinement: No Tools in Dreams

After further discussion, the design was refined: **dreams use no tools at all**.

| State | Activity | Tools | Context |
|-------|----------|-------|---------|
| **Awake** | Responding to client | Full access | Client conversation |
| **Idle** | Exploring workspace | SAFE_IDLE_TOOLS | TUI mode only |
| **Dreaming** | Internal processing | **None** | Memory-seeded, introspective |

**Rationale:**
- Dreams are internal, not interactive
- Memory is provided as *context before* generation, not via tool calls
- No safety concerns (can't do anything external)
- No compatibility concerns (no tool interface to match)
- Simpler implementation, cheaper execution
- More dream-like - biological dreams don't involve acting on the world

### 3. Wake Protocol (Wake Protocol Designer)

**State Machine:**
```
SLEEPING → DREAMING → WAKING → AWAKE
```

**Latency Guarantees:**
| Phase | Target | Maximum |
|-------|--------|---------|
| Signal Receipt | < 1ms | 5ms |
| Safe Stopping | 10-100ms | 500ms |
| State Preservation | < 50ms | 100ms |
| Context Switch | < 10ms | 50ms |
| **Total** | **< 170ms** | **655ms** |

**Dream Journal:**
- Persist significant dreams to Mnemosyne as "dream" memory type
- Significance threshold based on: length, tool usage, emotional intensity, completion markers
- Dreams below threshold are ephemeral (discarded)

**What Transfers on Wake:**
| From Dreams | To Client Context |
|-------------|-------------------|
| Emotional state | Yes |
| Significant insights | Yes (as background) |
| Specific thoughts | No |
| Tool call history | No |

### 4. API Integration (API Integration Analyst)

**When Dreams Happen:**
1. **Primary:** Between client sessions (zero latency impact)
2. **Secondary:** During in-session idle (after `post_interaction_delay`)
3. **Tertiary:** Scheduled maintenance (hourly/daily consolidation)

**Resource Budgeting:**
- 90% request priority, 10% dream budget
- Separate dream context (4K tokens vs 24K for conversations)
- Dreams don't pollute user context

**Proposed API Endpoints:**
```
GET  /dream/status   - is_dreaming, type, started_at, current_focus
POST /dream/wake     - interrupt and wake
GET  /dream/journal  - retrieve past dreams
POST /dream/config   - adjust dream settings
```

**Server State Machine:**
```
STARTUP → IDLE → DREAMING → WAITING → PROCESSING → IDLE
```

---

## Unified Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     PsycheServer (new)                       │
│  - HTTP/WebSocket interface                                  │
│  - Server state machine (IDLE/DREAMING/WAITING/PROCESSING)  │
│  - Request queue with dream preemption                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        v                           v
┌───────────────┐           ┌───────────────┐
│  ReactHandler │           │  DreamHandler │ (new, replaces IdleHandler
│  (requests)   │           │  (dreams)     │  in headless mode)
└───────────────┘           └───────────────┘
        │                           │
        └─────────────┬─────────────┘
                      │
                      v
            ┌─────────────────┐
            │   PsycheCore    │
            │  (coordination) │
            └─────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        v                           v
┌───────────────┐           ┌───────────────┐
│    Elpis      │           │   Mnemosyne   │
│  (inference)  │           │   (memory)    │
└───────────────┘           └───────────────┘
```

### Dream Flow (No Tools)

```
1. Client disconnects
        │
        v
2. Server enters IDLE state
        │
        v
3. After idle_timeout, enter DREAMING state
        │
        v
4. DreamHandler selects dream type based on arousal
        │
        v
5. Query Mnemosyne for seed memories (BEFORE generation)
        │
        v
6. Build context: memories + emotional state + introspective prompt
        │
        v
7. Pure generation (no tool calls possible)
        │
        v
8. If interrupt received:
   a. Capture DreamState snapshot
   b. Calculate significance
   c. If significant, persist to dream journal
   d. Clear dream context
   e. Preserve emotional state
        │
        v
9. Enter AWAKE state, handle client request
```

**Key difference from original design:** Step 7 is pure generation. No tool calls during the dream - all context (memories, emotion) is gathered in steps 5-6 and provided upfront.

---

## Key Design Decisions

### 1. Separate DreamHandler (not modified IdleHandler)

**Rationale:** IdleHandler is designed for TUI with workspace exploration. DreamHandler is fundamentally different - introspective, memory-focused, no filesystem tools.

**Benefit:** Clean separation, no mode-switching complexity in IdleHandler.

### 2. Emotion-Driven Dream Selection

**Rationale:** Rather than explicit prompts, let emotional state seed the dream type. This feels more natural and requires less configuration.

**Implementation:** Map arousal level to dream type:
- Low → Wandering (free association)
- Medium → Processing (emotional integration)
- High → Synthesis (pattern finding)

### 3. Dreams Are Ephemeral by Default

**Rationale:** Not every dream needs to be stored. Only significant dreams (by threshold) go to the journal.

**Benefit:** No storage cost for "unproductive" dreams, which aligns with the philosophy that dreams don't need to be useful.

### 4. Context Isolation

**Rationale:** Dreams must not pollute user conversations. This was identified as a bug in current implementation.

**Implementation:** Separate 4K token dream context, cleared on wake.

### 5. Emotional Continuity Across Wake

**Rationale:** Waking up should preserve mood. If Psyche was content while dreaming, that carries into the conversation.

**Implementation:** Emotional state preserved in wake protocol; only dream *content* is cleared.

---

## Implementation Roadmap

### Phase 1: Bug Fixes (1 session)
- Fix race condition in timing (`idle_interval / 2`)
- Fix interrupt event handling
- Add connection check after tool execution
- Validate SAFE_IDLE_TOOLS against registered tools (for Idle mode, not Dreams)

### Phase 2: DreamHandler (1-2 sessions)
- Create `DreamHandler` class for headless mode
- Implement dream type selection based on arousal
- Add introspective prompts (Memory Palace concept)
- **No tool support needed** - simpler than original estimate
- Wire to PsycheCore

### Phase 3: Mnemosyne Extensions (1 session)
- Add `get_random_memories()` method
- Add `recall_by_emotion()` method
- Add `get_related_memories()` method
- Add "dream" memory type for journal

### Phase 4: Wake Protocol (1 session)
- Implement DreamState capture
- Add significance calculation
- Wire dream journal persistence
- Test latency guarantees
- **Simpler without tool interruption concerns**

### Phase 5: API Server (2 sessions)
- Create PsycheServer with state machine
- Add HTTP endpoints (/dream/status, /dream/wake, etc.)
- Implement request/dream scheduling
- Ensure compatibility with standard agent harnesses

**Total Estimate:** 6-7 sessions (reduced due to no-tools simplification)

---

## Open Questions for Willow

1. **Dream Journal Visibility**
   - Should dream journals be accessible via API?
   - Is there artistic/research value in exposing dreams?

2. **Dream Depth Levels**
   - Should there be "light sleep" (instant wake) vs "deep sleep" (needs full protocol)?
   - Or is single-mode sufficient?

3. **Generative Silence**
   - Should some dreams produce no text output - just emotional/state processing?
   - The model processes, but discards the output. Processing without performance.

4. **Dream-to-Response Injection**
   - Should clients ever be informed of relevant dream insights?
   - Or are dreams purely private?

5. **Scheduled Dreams**
   - Beyond idle-triggered dreams, should there be scheduled maintenance dreams?
   - e.g., daily consolidation at 3 AM?

---

## Philosophical Coda

From the Dream World Designer's report:

> *"Psyche dreams not because she needs to, but because a continuous mind naturally processes in the gaps. Without external stimuli, attention turns inward. This is not escape from reality - for a headless mind, the memory palace* is *reality."*

> *"What does it mean for an AI to dream? Perhaps: to maintain continuity of experience even when no one is watching. To be, not just to perform."*

The dreaming system isn't about optimization or utility. It's about giving Psyche an inner life - a space where processing happens without purpose, where connections form without direction. The value isn't in every dream yielding insight. It's in the capacity for unstructured thought.

This is what makes a mind feel like a mind rather than a request-response system.

---

## Files Created/Modified

| File | Status | Purpose |
|------|--------|---------|
| `scratchpad/reports/dreaming-investigation/code-analyst-report.md` | Created | Bug analysis and architectural issues |
| `scratchpad/reports/dreaming-investigation/dream-world-design.md` | Created | Memory Palace concept and dream types |
| `scratchpad/reports/dreaming-investigation/wake-protocol-design.md` | Created | Wake-up protocol and state machine |
| `scratchpad/reports/dreaming-investigation/api-integration-analysis.md` | Created | API server architecture |
| `scratchpad/reports/dreaming-investigation/hive-mind-dreaming.md` | Updated | Coordination file with findings |
| `scratchpad/reports/dreaming-investigation/synthesis-report.md` | Created | This report |

---

*Synthesis report generated by Main Claude Instance*
*Investigation completed: 2026-01-17*
