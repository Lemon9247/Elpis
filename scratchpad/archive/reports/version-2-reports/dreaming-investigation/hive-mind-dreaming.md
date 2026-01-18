# Hive Mind: Dreaming State Investigation

**Date:** 2026-01-17
**Coordinator:** Main Claude Instance
**Status:** Complete

## Context

Psyche is being prepared to run as a headless API server (Phase 5). The current "idle thinking" / dreaming mechanism needs to be redesigned to work in this context.

### Key Constraints
- One Psyche instance, one client at a time (sequential, not concurrent)
- Must be able to cleanly "wake up" Psyche from dreams
- Current implementation has bugs that need identification
- Headless mode has no workspace - what does Psyche dream about?

### Questions to Answer
1. What are the bugs/issues in the current IdleHandler implementation?
2. What should the "dream world" look like for headless Psyche?
3. How do we implement clean wake-up from dreams?
4. How does dreaming integrate with the API server architecture?

## Agent Assignments

| Agent | Role | Focus Area |
|-------|------|------------|
| Code Analyst | Analyze current implementation | Find bugs in IdleHandler, identify architectural issues |
| Dream World Designer | Design dream environment | What workspace/context does headless Psyche explore? |
| Wake Protocol Designer | Design interruption mechanism | Clean wake-up, state preservation, transition handling |
| API Integration Analyst | Consider server integration | How dreaming fits with request handling, scheduling |

## Coordination Notes

*Agents should write findings below and in their individual report files*

---

## Agent Findings Summary

### Code Analyst Findings
**Report:** `code-analyst-report.md`

Key findings:
1. **Race condition in timing** - `idle_interval / 2` creates unpredictable 60-90s windows
2. **Phantom interruptions** - Interrupt event cleared in multiple places, can cancel multiple cycles
3. **Missing disconnect detection** - No connection check after tool execution
4. **SAFE_IDLE_TOOLS mismatch** - `recall_memory` fails when Mnemosyne disabled
5. **Dreams pollute user context** - Shared compactor between idle and user conversations
6. **No dream persistence** - Dreams are ephemeral, lost on restart
7. **Workspace-centric prompts** - Meaningless in headless mode
8. **Estimated effort: 2-3 sessions** for bug fixes, 4-5 for headless adaptation

### Dream World Designer Findings
**Report:** `dream-world-design.md`

Key findings:
1. **Memory Palace concept** - Dreams are introspective, not exploratory
2. **No workspace needed** - Headless Psyche explores memories, not files
3. **Four dream types based on arousal:**
   - Wandering (low arousal): free association through memories
   - Processing (medium arousal): emotional integration
   - Synthesis (high arousal): pattern finding across experiences
   - Reflection (variable): self-focused introspection
4. **New DreamHandler** - Replace IdleHandler in headless mode
5. **Only tool needed: `recall_memory`** - No filesystem access
6. **Cheaper than TUI idle thinking** - Memory queries vs file operations
7. **New Mnemosyne queries needed:** random memories, emotion-based recall, relationship following
8. **Dreams are ephemeral by default** - Optional journaling for research
9. **Dreams don't directly inform responses** - Background processing, not content generation
10. **Estimated effort: 2 sessions**

### Wake Protocol Designer Findings
**Report:** `wake-protocol-design.md`

Key findings:
1. **Current interrupt mechanism is sound** - uses `asyncio.Event()` with cooperative checking during token streaming
2. **Missing state preservation** - interrupted dreams are discarded, no dream journal exists
3. **Safe stopping points identified:**
   - Token generation: safe after each token
   - Tool execution: wait for completion with timeout (5s max)
   - Memory operations: atomic via MCP, skip if interrupt pending
4. **Proposed 4-state machine:** SLEEPING -> DREAMING -> WAKING -> AWAKE
5. **Target latency: 200ms typical, 1s maximum** with timeout escalation
6. **Dream journal design:** Persist significant dreams to Mnemosyne as "dream" memory type
7. **Context switching:** Clear dream context, preserve emotional state, optionally inject insights

### API Integration Analyst Findings
**Report:** `api-integration-analysis.md`

Key findings:
1. **Cooperative preemption model** recommended - dreams run during idle, yield to requests
2. **Hybrid scheduling strategy**: Between-session dreams (deep), in-session idle (light), scheduled maintenance
3. **Request handling**: Existing `IdleHandler.interrupt()` mechanism is well-designed, needs minor enhancement for granular checking
4. **API endpoints proposed**: `/dream/status`, `/dream/wake`, `/dream/config`, `/dream/journal`
5. **Resource budgeting**: Time-sliced approach with request priority, context isolation for dreams
6. **State machine**: STARTUP -> IDLE -> DREAMING -> WAITING -> PROCESSING
7. **Risks identified**: Interrupt latency, state corruption, resource exhaustion, tool safety (mitigations provided)
8. **HTTP/WebSocket interface** sketched with FastAPI patterns for streaming and events

---

## Open Questions (Answered)

| Question | Answer |
|----------|--------|
| Should dreams be persisted? | Yes, significant ones (by threshold) to dream journal |
| Can dreams inform responses? | Indirectly, via emotional state and memory consolidation |
| Different depths of dreaming? | Defer - start with single mode |
| How to measure benefit? | Open - needs metrics design |

## Decisions Made

### From Agent Investigation
1. **Memory Palace concept** - Dreams are introspective, not exploratory
2. **Four dream types** - Based on arousal level (Wandering, Processing, Synthesis, Reflection)
3. **Wake protocol** - 4-state machine with 200ms target latency
4. **Dream journal** - Persist significant dreams to Mnemosyne
5. **Context isolation** - Dreams use separate 4K context, don't pollute conversations
6. **Emotional continuity** - Preserved across wake

### Post-Synthesis Refinements (from discussion with Willow)
7. **No tools in dreams** - Memory provided as context before generation, not via tool calls
8. **Compatibility requirement** - Standard API works without knowing about dreams
9. **Three distinct states:**
   - Awake: full tools, client conversation
   - Idle: safe tools, TUI mode only
   - Dreaming: **no tools**, memory-seeded, purely generative

---

## Final Synthesis

See `synthesis-report.md` for complete findings.

**Key insight:** Dreams are purely generative - internal processing with no external reach. This simplifies implementation and ensures compatibility with other agent harnesses (OpenCode, etc.).

**Estimated effort:** 6-7 sessions total
