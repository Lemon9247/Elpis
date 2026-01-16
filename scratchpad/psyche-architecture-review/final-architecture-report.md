# Psyche Architecture Review - Final Synthesis Report

**Date**: 2026-01-16
**Prepared by**: Orchestrating Agent
**Research Team**:
- Codebase Review Agent
- Coding Agents Review Agent
- Memory Systems Review Agent
- Reasoning Workflows Review Agent

---

## Executive Summary

This report synthesises findings from four parallel research efforts examining Psyche's current architecture and pathways for improvement. The goal was to assess the current state, learn from other coding agents, and develop an actionable improvement plan.

### Key Findings

1. **Critical Bug**: Memory storage is broken - `_staged_messages` buffer is never populated, causing conversation loss on shutdown
2. **Architecture is Sound**: Good foundations (async, callbacks, state machines) but execution flaws
3. **Clear Patterns Exist**: Other agents (OpenCode, Crush, Letta) provide proven patterns to adopt
4. **FocusLLM Not Applicable**: The paper addresses training-time context extension, not runtime memory management
5. **Reasoning is Feasible**: Can be implemented with prompt engineering using existing ThoughtPanel infrastructure

### Priority Matrix

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| P0 | Fix memory storage bug | Medium | Critical |
| P1 | Add interruption support | High | High |
| P1 | Implement clean tool display | Low | High |
| P2 | Add reasoning workflow | Medium | Medium |
| P2 | Stream dream state | Medium | Medium |
| P3 | Provider abstraction | Medium | Medium |
| P3 | MCP standardization | High | Medium |

---

## 1. Codebase Analysis Summary

### Current State Assessment

Psyche is a functional prototype with:

**Strengths**:
- Clean separation between UI (Textual) and logic (MemoryServer) via callbacks
- Proper async-first design throughout
- Working tool system with Pydantic validation
- Emotional modulation via Elpis backend
- ChromaDB-backed memory via Mnemosyne

**Critical Issues**:
- **Memory loss**: Compaction drops messages without staging them for storage
- **No interruption**: Cannot stop streaming or tool execution
- **Poor tool UX**: Raw JSON dumped to context instead of human-readable summaries
- **No reasoning phase**: LLM responds immediately without thinking step
- **Hardcoded servers**: Only Elpis/Mnemosyne supported, no external MCP

### Architecture Debt

| Component | Lines | Health | Notes |
|-----------|-------|--------|-------|
| `server.py` | ~1400 | Needs refactor | Too many responsibilities |
| `compaction.py` | ~150 | Incomplete | Summarization unimplemented |
| `app.py` | ~300 | Good | Clean TUI implementation |
| `tool_engine.py` | ~350 | Good | Solid async orchestration |

---

## 2. Coding Agents Review Summary

Three agents were analyzed: OpenCode, Crush, and Letta.

### Key Patterns to Adopt

#### 2.1 Provider Abstraction Layer
OpenCode demonstrates how to support 75+ LLM providers through a unified interface:

```python
class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, messages, tools=None) -> AsyncIterator:
        pass
```

**Recommendation for Psyche**: Create `LLMProvider` interface to support Elpis, Ollama, cloud APIs.

#### 2.2 Clean Tool Display
Both OpenCode and Crush transform tool calls into human-readable summaries:

```
Instead of:
{"name": "edit_file", "arguments": {...}}

Display:
Editing /path/to/file.py
  + Added: 5 lines
  - Removed: 2 lines
```

**Recommendation for Psyche**: Implement `ToolDisplayFormatter` with templates per tool type.

#### 2.3 Letta-Style Memory Blocks
Structure context as discrete, editable memory blocks:

```python
blocks = {
    "system": MemoryBlock(label="system", limit=1000, ...),
    "human": MemoryBlock(label="human", limit=2000, ...),
    "working_memory": MemoryBlock(label="working", limit=4000, ...),
}
```

**Recommendation for Psyche**: Adopt for better context management than naive sliding window.

#### 2.4 MCP as Standard Interface
Letta shows native MCP integration for tool discovery and execution.

**Recommendation for Psyche**: Route all tools through MCP, including internal ones.

### Anti-Patterns to Avoid

1. Raw JSON tool display
2. Tight backend coupling
3. Unbounded context growth
4. Custom tool protocols
5. Blocking tool execution

---

## 3. Memory Systems Review Summary

### FocusLLM Paper Assessment

**Verdict: NOT APPLICABLE**

FocusLLM is a training-time architecture modification:
- Requires retraining/fine-tuning models
- Uses parallel context encoding with "focus tokens"
- Cannot be applied to external models (llama-cpp, transformers)

**Conceptual Insight**: The "focus token" concept maps to what good summarization should do - extract salient information from context chunks.

### Current Memory Issues

1. **Shutdown storage broken**: Race condition in `shutdown_with_consolidation`
2. **Passive compaction only**: Only stores on overflow, not explicitly
3. **Naive summarization**: Uses `content[:100]` truncation
4. **No semantic deduplication**: Similar memories not merged
5. **1-hour consolidation delay**: Short sessions don't consolidate

### Recommended Memory Workflow

```
User Message
    |
    v
[Auto-Retrieve Relevant Memories] --> Inject into prompt
    |
    v
[Generate Response]
    |
    v
[Agent decides: store_memory?] --> Yes --> Store to Short-Term
    |                              |
    |                              v
    |                    [Periodic Checkpoints]
    |                              |
    v                              v
[Working Memory Only]     [Idle Consolidation] --> Long-Term
```

**Key Recommendations**:
1. Fix shutdown storage bug immediately
2. Add automatic context retrieval on user messages
3. Implement periodic checkpoints (every N messages)
4. Use LLM for structured summarization
5. Add local fallback storage for disconnections

---

## 4. Reasoning Workflows Review Summary

### Reasoning Model Landscape

| Model | Type | Availability | Approach |
|-------|------|--------------|----------|
| OpenAI o1/o3 | Hidden reasoning | Cloud only | RL-trained internal CoT |
| DeepSeek-R1 | Visible reasoning | Open weights | Explicit `<think>` tags |
| Claude Extended | Configurable | Cloud API | Separate thinking block |

### Recommended Approach for Psyche

**Phase 1: Basic Reasoning (2-3 hours)**
1. Update system prompt to request `<thinking>` tags
2. Parse reasoning from streamed output
3. Route to existing ThoughtPanel
4. Add `/thinking on|off` command

**System Prompt Addition**:
```
When responding, first think through it inside <thinking> tags.
Consider:
- What the user is asking for
- What information or tools you need
- Your approach to solving it

After thinking, provide your response outside the tags.
```

**Phase 2: Streaming Reasoning (4-6 hours)**
- Implement `ReasoningStreamParser` for real-time tag handling
- Dual-stream UI (thinking to ThoughtPanel, response to ChatView)
- Progress indication during thinking phase

**Performance Considerations**:
- Reasoning adds ~30-100% to generation time
- Recommend 512 token budget for thinking
- Allow user to skip reasoning for simple queries (`/quick [message]`)

---

## 5. Proposed Target Architecture

```
+------------------+
|   Psyche TUI     |
| (Textual Python) |
+--------+---------+
         |
         | Tool Display Layer
         | (human-readable summaries)
         |
+--------v---------+
|  Agent Runtime   |
| - Context Manager|  <-- Memory Blocks
| - Reasoning      |  <-- <thinking> parser
| - Tool Registry  |
+--------+---------+
         |
         | Provider Abstraction Layer
         |
+--------v---------+
|  LLM Provider    |
| - Elpis (local)  |
| - Ollama         |
| - Anthropic      |
| - OpenAI         |
+--------+---------+
         |
         | MCP Protocol
         |
+--------v---------+        +------------------+
|  Tool Servers    | <----> |   Mnemosyne      |
| - File Ops       |        |  (Memory MCP)    |
| - Shell          |        +------------------+
| - Search         |
+------------------+        +------------------+
                    <----> |    Elpis         |
                           | (Inference MCP)  |
                           +------------------+
```

**Key Properties**:
1. **UI Layer**: Only handles display, receives structured events
2. **Agent Runtime**: Core logic with separated concerns
3. **Provider Layer**: Abstracts LLM backends
4. **Tool Layer**: All tools via MCP standard

---

## 6. Implementation Roadmap

### Phase 1: Critical Fixes (Immediate)

| Task | Files | Effort | Impact |
|------|-------|--------|--------|
| Fix memory staging bug | `server.py`, `compaction.py` | 4h | Critical |
| Add shutdown signal handlers | `cli.py`, `app.py` | 2h | High |
| Local storage fallback | `server.py` | 4h | Medium |

### Phase 2: UX Improvements (Short-term)

| Task | Files | Effort | Impact |
|------|-------|--------|--------|
| Implement ToolDisplayFormatter | `server.py`, new file | 6h | High |
| Add interruption support | `server.py`, `mcp/client.py` | 12h | High |
| Show help on startup | `app.py` | 2h | Low |
| Stream dream state | `server.py`, `app.py` | 8h | Medium |

### Phase 3: Architecture (Medium-term)

| Task | Files | Effort | Impact |
|------|-------|--------|--------|
| Implement reasoning workflow | `server.py`, `app.py` | 8h | Medium |
| Refactor MemoryServer | `server.py` -> multiple | 20h | High |
| Memory block system | `memory/` | 16h | High |
| Provider abstraction | `mcp/`, new files | 16h | Medium |

### Phase 4: Extensibility (Long-term)

| Task | Files | Effort | Impact |
|------|-------|--------|--------|
| Dynamic MCP server loading | `cli.py`, `mcp/` | 20h | Medium |
| Tool result caching | `tool_engine.py` | 8h | Low |
| Reasoning cache | new file | 12h | Low |

---

## 7. Key Decisions Needed

Before proceeding with implementation, clarification is needed on:

1. **Provider priority**: Which LLM backends beyond Elpis should be supported first? (Ollama, Anthropic, OpenAI)

2. **Memory architecture**: Adopt Letta-style memory blocks, or fix current sliding window approach?

3. **Reasoning scope**: Always-on reasoning, or opt-in per message?

4. **MCP vs internal tools**: Should all tools go through MCP, or keep some internal for performance?

5. **Refactor strategy**: Big-bang refactor of `server.py`, or incremental extraction?

---

## 8. Conclusion

Psyche has a solid foundation but needs focused work on:

1. **Fix the critical memory bug** - This is causing data loss
2. **Improve user control** - Interruption support is essential
3. **Better tool UX** - Clean display instead of JSON
4. **Add reasoning** - Leverage existing ThoughtPanel

The research shows clear patterns from other agents (OpenCode, Crush, Letta) that can be adopted. The FocusLLM paper is not applicable, but the memory systems research provides a solid roadmap for fixing the current issues.

The architecture is sound enough that these improvements can be made incrementally. The highest-impact, lowest-effort wins are:

1. **Fix memory staging** (Critical bug fix)
2. **Implement ToolDisplayFormatter** (Big UX win, low effort)
3. **Add basic reasoning** (Medium effort, leverages existing infrastructure)

---

## Appendix: Report Files

| Report | Location |
|--------|----------|
| Codebase Review | `codebase-review-report.md` |
| Coding Agents Review | `coding-agents-review-report.md` |
| Memory Systems Review | `memory-systems-review-report.md` |
| Reasoning Workflows Review | `reasoning-workflows-review-report.md` |
| Coordination | `hive-mind-psyche-review.md` |
| This Report | `final-architecture-report.md` |

---

**Report Status**: Complete
**Total Research Agents**: 4
**Total Reports**: 6 (including synthesis)
**Date**: 2026-01-16
