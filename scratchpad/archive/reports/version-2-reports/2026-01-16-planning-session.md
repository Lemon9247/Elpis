# Planning Session Report

**Date**: 2026-01-16
**Focus**: Workplan finalization and external agent architecture

---

## Session Summary

This session focused on refining the comprehensive workplan created from earlier investigation and architecture review work, with particular attention to how Elpis can serve as a backend for external coding agents.

---

## Background: Prior Investigation Work

### Bug Investigation (2026-01-14)

A systematic bug investigation was conducted across all three packages using parallel subagents:

| Package | Agent | Bugs Found |
|---------|-------|------------|
| Elpis | Elpis Bug Agent | 13 bugs |
| Mnemosyne | Mnemosyne Bug Agent | 12 bugs |
| Psyche | Psyche Bug Agent | 14 bugs |
| **Total** | | **39 bugs** |

**Severity breakdown**: 5 critical, 11 high, 11 medium, 12 low

**Key findings**:
- Fire-and-forget async tasks in Elpis (critical)
- Memory staging bug in Psyche - `_staged_messages` never populated (critical)
- Bare except clauses swallowing errors in Mnemosyne
- Threading issues causing SIGSEGV crashes during streaming

### Streaming Stability Work (2026-01-14 to 2026-01-16)

Investigated and fixed threading crashes in Elpis inference:

**Root cause**: `llama_context` is not thread-safe; the `_stream_in_thread()` approach caused segfaults

**Fix implemented** (committed `e36c856`):
- Removed threading bridges from both llama_cpp and transformers backends
- Direct iteration with `await asyncio.sleep(0)` for cooperative yielding
- Verified with GPU offload testing

### Architecture Review (2026-01-16)

Launched 4 parallel research subagents to analyze Psyche's architecture:

| Agent | Focus | Key Findings |
|-------|-------|--------------|
| Codebase Review | Current Psyche structure | Memory staging bug, 1400-line server.py needs refactor |
| Coding Agents Review | OpenCode, Crush, Letta | Provider abstraction, clean tool display, memory blocks patterns |
| Memory Systems Review | FocusLLM paper, memory approaches | FocusLLM not applicable (training-time), agent-driven memory recommended |
| Reasoning Workflows Review | o1, DeepSeek-R1, Claude | `<thinking>` tags with existing ThoughtPanel infrastructure |

**Reports generated**:
- `scratchpad/psyche-architecture-review/codebase-review-report.md`
- `scratchpad/psyche-architecture-review/coding-agents-review-report.md`
- `scratchpad/psyche-architecture-review/memory-systems-review-report.md`
- `scratchpad/psyche-architecture-review/reasoning-workflows-review-report.md`
- `scratchpad/psyche-architecture-review/final-architecture-report.md`

### Synthesis into Comprehensive Workplan

All findings were synthesized into a comprehensive workplan with:
- 4 work tracks (A: Stability, B: Memory, C: UX, D: Architecture)
- Priority matrix based on impact and effort
- Session-based estimates (replacing human hour estimates)
- Implementation schedule across 4 phases

This workplan was then iterated upon in the current session to add Track E (External Agent Support).

---

## Key Discussions (This Session)

### 1. Session-Based Estimation

Updated CLAUDE.md and the workplan to use **Claude Code session-based estimates** instead of human hours:

- A "session" = one Claude Code context window
- Each session should be a coherent, testable unit of work
- Provides clearer planning than abstract hour estimates

**Result**: 21 sessions across 5 phases

### 2. External Agent Architecture (Major Reframe)

**Original proposal (rejected)**: D3 Provider Abstraction - make Psyche consume multiple LLM backends (Ollama, Anthropic, etc.)

**Adopted approach**: Make Elpis usable as a backend for OTHER coding agents (Aider, Continue, Cursor, Claude Code)

**Research findings**:
- OpenAI-compatible `/v1/chat/completions` is the de-facto standard
- Nearly all coding agents support this format
- MCP is emerging for tool interfaces

**Architecture**:
```
Elpis Process
├── MCP Server (stdio) - for Psyche (unchanged)
├── HTTP Server (:8000) - OpenAI-compatible for external agents
└── Shared: InferenceEngine, EmotionalState

Emotion Coordinator (separate MCP server)
├── Connects to Elpis HTTP for emotion
└── Connects to Mnemosyne for memory
```

### 3. Emotion Coordinator Purpose

Clarified that the Emotion Coordinator allows external agents to **participate in Elpis's emotional feedback loop**:

1. **Read emotional state** - valence/arousal for display or decisions
2. **Send emotional events** - success/failure affects future inference
3. **Access memory** - store/recall via Mnemosyne

This mirrors what Psyche does internally, but exposed via MCP for external agents.

### 4. Design Decisions Resolved

| Question | Decision |
|----------|----------|
| HTTP API auth | Optional API key for network; local requests unauthenticated |
| Tool calling | Yes, OpenAI-format tool calling in E1 scope |
| Emotion Coordinator location | Separate package: `src/emotion_coordinator/` |

### 5. Workplan Consistency Fixes

Identified and fixed several inconsistencies:

- Track E session count (4 → 3, E2a/E2b combined)
- Confusing D session numbering (D5/D6 → D3 part 1/2)
- C4 naming confusion (removed inline labels)
- Added track breakdown math to Session Summary

---

## Documents Created/Modified

| Document | Action |
|----------|--------|
| `CLAUDE.md` | Added session-based planning guidelines |
| `scratchpad/plans/2026-01-16-comprehensive-workplan.md` | Original workplan with session estimates |
| `scratchpad/plans/2026-01-16-external-agent-architecture.md` | **New** - Complete merged workplan with external agent architecture |

---

## Final Workplan Structure

### Tracks

| Track | Sessions | Focus |
|-------|----------|-------|
| A | 2 | Stability & bug fixes |
| B | 3 | Memory system overhaul |
| C | 4 | User experience |
| D | 6 | Architecture evolution |
| E | 3 | External agent support |
| Integration | 3 | Cross-track testing |
| **Total** | **21** | |

### Phases

1. **Phase 1** (5 sessions): Critical fixes - streaming, memory persistence
2. **Phase 2** (4 sessions): UX quick wins - tool display, interruption
3. **Phase 3** (3 sessions): Memory & reasoning workflows
4. **Phase 4** (7 sessions): Architecture & interoperability
5. **Phase 5** (2 sessions): External agent integration

### Track E Details (New)

| Task | Description |
|------|-------------|
| E1 | HTTP API with OpenAI-compatible `/v1/chat/completions`, tool calling, optional auth |
| E2 | Emotion Coordinator MCP server at `src/emotion_coordinator/` |
| E3 | Integration testing with Aider, Continue, etc. |

---

## Next Steps

1. **Recommended start**: Session 1 (test streaming fix + async bugs)
2. Phases 1-3 focus on stability and UX for Psyche
3. Phases 4-5 add external agent support

---

## Session Artifacts

### From Prior Investigation Work
- Bug investigation synthesis: `scratchpad/bug-investigation/synthesis-report.md`
- Architecture review reports: `scratchpad/psyche-architecture-review/*.md`
- Threading fix report: `scratchpad/reports/2026-01-16-threading-fix.md`
- Original comprehensive workplan: `scratchpad/plans/2026-01-16-comprehensive-workplan.md`

### From This Session
- **Final workplan**: `scratchpad/plans/2026-01-16-external-agent-architecture.md`
- Updated CLAUDE.md with session-based planning guidelines
- This report: `scratchpad/reports/2026-01-16-planning-session.md`

---

**Session Status**: Complete
**Duration**: Extended planning session
**Outcome**: Finalized 21-session workplan with external agent architecture
