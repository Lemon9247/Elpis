# Elpis/Psyche Comprehensive Workplan

**Date**: 2026-01-16
**Status**: Planning Complete - Ready for Implementation
**Estimation Method**: Claude Code Sessions (not human hours)

> A "session" = one Claude Code context window, typically completing 1-3 focused tasks.
> Each session should be a coherent, testable unit of work.

---

## Executive Summary

This workplan synthesizes findings from:
- Bug investigation (2026-01-14): 39 bugs found across 3 packages
- Streaming stability work (2026-01-14 to 2026-01-16): Threading fix implemented
- Memory storage fixes (2026-01-14): Partial implementation
- Architecture review (2026-01-16): Full codebase analysis with external research

### Current State

| Component | Status | Blockers |
|-----------|--------|----------|
| Elpis Streaming | FIXED | Threading removed, needs testing |
| Memory Storage | BROKEN | `_staged_messages` never populated |
| Tool Display | Poor UX | Raw JSON shown |
| Interruption | Missing | Cannot cancel streaming/tools |
| Reasoning | Missing | No thinking step |
| Interoperability | Limited | Hardcoded MCP servers |

### Work Tracks

```
Track A: Stability & Bug Fixes (P0)
Track B: Memory System Overhaul (P0-P1)
Track C: User Experience (P1-P2)
Track D: Architecture Evolution (P2-P3)
```

---

## Track A: Stability & Bug Fixes

### A1. Streaming Stability Verification (READY TO TEST)

**Status**: Code complete (committed 2026-01-16: `e36c856`)

**What was done**:
- Removed `_stream_in_thread()` from llama_cpp backend
- Removed redundant outer threading from transformers backend
- Direct iteration with `await asyncio.sleep(0)` cooperative yielding

**Testing needed**:
| Test | Scenario | Pass Criteria |
|------|----------|---------------|
| Basic streaming | Send chat message | Tokens stream without crash |
| Dream state | Let idle for 60s | Idle thought generates without SIGSEGV |
| GPU offload | 20 GPU layers | nvidia-smi shows ~4GB usage |
| Long generation | 1000+ tokens | Completes without crash |
| Rapid requests | 5 requests in 10s | All complete, no orphan threads |

**Session**: Can batch with A2 tasks

---

### A2. Critical Async Bug Fixes

**Source**: Bug investigation synthesis (2026-01-14)

#### A2.1 Fire-and-Forget Tasks (CRITICAL)
**File**: `src/elpis/server.py:419`
```python
# CURRENT (BAD): Task created without reference
asyncio.create_task(producer())

# FIX: Store task, handle exceptions
self._stream_tasks[stream_id] = asyncio.create_task(producer())
```

#### A2.2 Race Condition in Stream State
**File**: `src/elpis/server.py:456-479`
- Stream deleted while producer still running
- Need proper lifecycle management

#### A2.3 Bare Except Clauses
**Files**: `src/mnemosyne/storage/chroma_store.py:139-140, 147-148`
```python
# CURRENT (BAD): Swallows ALL exceptions
except:
    return []

# FIX: Specific exceptions with logging
except chromadb.errors.ChromaError as e:
    logger.error(f"ChromaDB error: {e}")
    return []
```

#### A2.4 Server Task Dies Silently
**File**: `src/psyche/client/app.py:87-93`
- Background server task can die without UI knowing
- Need health monitoring

**Track A Total**: 2 sessions
- Session A1: Test streaming fix + A2.1-A2.2 (async bugs)
- Session A2: A2.3-A2.4 + verification testing

---

## Track B: Memory System Overhaul

### B1. Fix Critical Memory Loss Bug (P0)

**Problem**: `_staged_messages` buffer NEVER gets populated during normal operation.

**Root Cause** (from architecture review):
```python
# compaction.py drops messages but doesn't stage them
# server.py expects staged messages but never receives them
```

**Location**: `src/psyche/memory/server.py`, `src/psyche/memory/compaction.py`

#### B1.1 Fix Staging Mechanism
```python
# In compaction.py, when dropping messages:
def compact(self) -> CompactionResult:
    dropped = self.messages[:drop_count]
    self.messages = self.messages[drop_count:]
    return CompactionResult(
        dropped_messages=dropped,  # <-- These should become staged
        ...
    )

# In server.py, _handle_compaction_result:
self._staged_messages.extend(result.dropped_messages)  # <-- ADD THIS
```

#### B1.2 Shutdown Signal Handlers
```python
# In cli.py or app.py:
import signal

def handle_shutdown(signum, frame):
    asyncio.get_event_loop().run_until_complete(
        memory_server.shutdown_with_consolidation()
    )
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)
```

#### B1.3 Local Fallback Storage
When Mnemosyne disconnected, write to local JSON:
```python
# On storage failure:
checkpoint = {
    "timestamp": datetime.now().isoformat(),
    "messages": [m.to_dict() for m in messages],
    "pending": True
}
Path("~/.psyche/pending_memories.json").write_text(json.dumps(checkpoint))
```

**Session B1**: B1.1 + B1.2 + B1.3 (all related to memory persistence)

---

### B2. Memory Workflow Improvements (P1)

**Source**: Memory systems review (FocusLLM paper analysis + Letta research)

#### B2.1 Automatic Context Retrieval
On each user message, query Mnemosyne for relevant memories:
```python
async def _process_user_input(self, text: str):
    # NEW: Retrieve relevant memories first
    if self.mnemosyne_client and self.mnemosyne_client.is_connected:
        memories = await self.mnemosyne_client.recall_memory(text, n_results=3)
        if memories:
            memory_context = self._format_memories_for_context(memories)
            # Inject into context
```

#### B2.2 Periodic Checkpoints
Don't wait for shutdown - save every N messages:
```python
CHECKPOINT_INTERVAL = 20  # messages

async def _maybe_checkpoint(self):
    if self._message_count % CHECKPOINT_INTERVAL == 0:
        await self._save_checkpoint()
```

#### B2.3 Structured Summarization
Replace truncation with LLM-generated summaries (partially done):
- Verify existing `_summarize_conversation` works correctly
- Add structured format (topics, facts, decisions)

**Session B2**: B2.1 + B2.2 (retrieval + checkpoints)
**Session B3**: B2.3 + testing all memory improvements

**Track B Total**: 3 sessions

---

## Track C: User Experience

### C1. Clean Tool Display (P1)

**Source**: Coding agents review (OpenCode, Crush patterns)

**Current**: Raw JSON dumped to context
**Target**: Human-readable summaries

#### C1.1 ToolDisplayFormatter Class
```python
class ToolDisplayFormatter:
    TEMPLATES = {
        "read_file": "Reading {path}",
        "write_file": "Writing {path} ({lines} lines)",
        "execute_bash": "Running: {command}",
        "recall_memory": "Recalling: {query}",
    }

    def format(self, tool_name: str, args: dict) -> str:
        template = self.TEMPLATES.get(tool_name, f"Using {tool_name}")
        return template.format(**args)
```

#### C1.2 Update Tool Activity Widget
Show more than just name + icon - show formatted action and result summary.

**Session C1**: C1.1 + C1.2 (complete tool display overhaul)

---

### C2. Interruption Support (P1)

**Source**: Architecture review + bug investigation

#### C2.1 Streaming Interruption
```python
class InterruptableStreamParser:
    def __init__(self):
        self._interrupted = asyncio.Event()

    async def stream_with_interrupt(self, generator):
        async for token in generator:
            if self._interrupted.is_set():
                yield "[Interrupted]"
                return
            yield token

    def interrupt(self):
        self._interrupted.set()
```

#### C2.2 Tool Interruption
Add timeout and cancellation to tool execution:
```python
async def execute_with_timeout(self, tool_call, timeout=30):
    try:
        return await asyncio.wait_for(
            self._execute(tool_call),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return {"error": "Tool execution timed out"}
```

#### C2.3 UI Keybindings
- `Ctrl+C` during streaming: Cancel generation
- `Ctrl+C` during tool: Cancel tool (with confirmation)

**Session C2**: C2.1 + C2.3 (streaming interruption + keybindings)
**Session C3**: C2.2 (tool interruption - more complex async work)

---

### C3. Help & Discoverability (P2)

#### C3.1 Show Help on Startup
First launch or empty input shows available commands.

#### C3.2 Command Aliases
Document and advertise short aliases (`/q`, `/h`, `/c`).

**Session C4**: C3.1 + C3.2 + C4 (quick wins, can batch)

---

### C4. Dream State Streaming (P2)

Make idle thinking stream tokens like main chat:
```python
async def _generate_idle_thought(self):
    async for token in self.client.generate_stream(...):
        if self.on_thinking:
            self.on_thinking(token)  # Stream to ThoughtPanel
```

**Track C Total**: 4 sessions

---

## Track D: Architecture Evolution

### D1. Reasoning Workflow (P2)

**Source**: Reasoning workflows review (o1, DeepSeek-R1, Claude extended thinking)

#### D1.1 System Prompt Update
```
When responding, first think through it inside <thinking> tags.
Consider: what is being asked, what tools you need, your approach.
After thinking, provide your response outside the tags.
```

#### D1.2 Reasoning Parser
```python
def parse_reasoning(text: str) -> tuple[str, str]:
    match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        response = text[match.end():].strip()
        return thinking, response
    return "", text
```

#### D1.3 Route to ThoughtPanel
Leverage existing infrastructure - add "reasoning" thought type.

#### D1.4 Toggle Command
`/thinking on|off` and `Ctrl+R` keybind.

**Session D1**: D1.1-D1.4 (complete reasoning workflow)

---

### D2. Refactor MemoryServer (P2-P3)

**Problem**: Single 1400-line class does too much.

**Target Structure**:
```
memory/
  server.py          # Coordination only
  inference.py       # LLM interaction
  context_manager.py # Memory blocks, compaction
  tool_handler.py    # Tool execution
  memory_handler.py  # Mnemosyne interaction
```

**Session D2**: Extract inference.py + context_manager.py
**Session D3**: Extract tool_handler.py + memory_handler.py
**Session D4**: Wire up, test, fix integration issues

---

### D3. Provider Abstraction (P3)

**Source**: Coding agents review (OpenCode pattern)

```python
class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, messages, tools=None) -> AsyncIterator:
        pass

class ElpisProvider(LLMProvider): ...
class OllamaProvider(LLMProvider): ...
class AnthropicProvider(LLMProvider): ...
```

**Session D5**: Define interface + ElpisProvider
**Session D6**: OllamaProvider + integration

---

### D4. MCP Standardization (P3)

Route ALL tools through MCP, including internal ones.
Add dynamic MCP server registration.

**Session D7**: Internal tools as MCP
**Session D8**: Dynamic server registration + config

**Track D Total**: 8 sessions

---

## Implementation Schedule

### Phase 1: Critical Fixes (5 sessions)

| Session | Track | Tasks | Deliverable |
|---------|-------|-------|-------------|
| 1 | A | A1 + A2.1-A2.2 | Streaming verified, async bugs fixed |
| 2 | A | A2.3-A2.4 | Exception handling, health monitoring |
| 3 | B | B1.1-B1.3 | Memory staging fixed, shutdown handlers, fallback |
| 4 | B | B2.1-B2.2 | Auto retrieval, periodic checkpoints |
| 5 | - | Integration testing | All Phase 1 verified working together |

**Phase 1 Gate**: Streaming stable, memory persists on shutdown

### Phase 2: UX Quick Wins (4 sessions)

| Session | Track | Tasks | Deliverable |
|---------|-------|-------|-------------|
| 6 | C | C1 | Human-readable tool display |
| 7 | C | C2.1 + C2.3 | Streaming can be interrupted |
| 8 | C | C2.2 | Tool execution can be cancelled |
| 9 | C | C3 + C4 | Help on startup, dream streaming |

**Phase 2 Gate**: User can interrupt any operation, tools display clearly

### Phase 3: Memory & Reasoning (3 sessions)

| Session | Track | Tasks | Deliverable |
|---------|-------|-------|-------------|
| 10 | B | B2.3 | Structured summarization verified |
| 11 | D | D1 | Reasoning workflow complete |
| 12 | - | Integration testing | Memory + reasoning verified |

**Phase 3 Gate**: Memories structured, thinking visible in ThoughtPanel

### Phase 4: Architecture (8 sessions)

| Session | Track | Tasks | Deliverable |
|---------|-------|-------|-------------|
| 13 | D | D2 (part 1) | inference.py + context_manager.py extracted |
| 14 | D | D2 (part 2) | tool_handler.py + memory_handler.py extracted |
| 15 | D | D2 (part 3) | Refactor complete, tests pass |
| 16 | D | D3 (part 1) | LLMProvider interface + ElpisProvider |
| 17 | D | D3 (part 2) | OllamaProvider + integration |
| 18 | D | D4 (part 1) | Internal tools via MCP |
| 19 | D | D4 (part 2) | Dynamic server registration |
| 20 | - | Final integration | All systems verified |

**Phase 4 Gate**: Clean architecture, multiple providers work

---

## Bug Inventory (from Investigation)

### Critical (5) - All in Track A

| Bug | Location | Status |
|-----|----------|--------|
| Fire-and-forget tasks | Elpis server.py:419 | To Fix (A2.1) |
| Race in stream state | Elpis server.py:456-479 | To Fix (A2.2) |
| Blocking sync calls | Mnemosyne (ChromaDB) | Deferred (needs asyncio.to_thread) |
| Bare except | Mnemosyne chroma_store.py | To Fix (A2.3) |
| Server task silent death | Psyche app.py:87-93 | To Fix (A2.4) |

### High (11) - Mixed across tracks

| Bug | Location | Track/Phase |
|-----|----------|-------------|
| Thread join timeout | Elpis inference.py | N/A (threading removed) |
| Steering hook leak | Elpis transformers | D4 (refactor) |
| Unreliable __del__ | Elpis transformers | D4 (refactor) |
| Type mismatch | Elpis settings.py | Quick fix |
| No Mnemosyne cleanup | Mnemosyne server.py | B1.2 |
| Memory tools no connection check | Psyche memory_tools.py | B1.1 |
| Staged messages race | Psyche server.py | B1.1 |
| Callbacks thread safety | Psyche app.py | C2.3 |
| Missing `_staged_messages` population | Psyche server.py | B1.1 (P0!) |

### Medium/Low (23) - Tracked for future

See: `scratchpad/bug-investigation/github-issues.md`

---

## Success Criteria

### Phase 1 Complete When:
- [ ] Streaming works without crashes for 1 hour continuous use
- [ ] Conversations are stored to Mnemosyne on shutdown
- [ ] Local fallback saves memories when Mnemosyne unavailable
- [ ] No "unhandled exception in task group" errors

### Phase 2 Complete When:
- [ ] Tool calls display as human-readable summaries
- [ ] User can interrupt streaming with Ctrl+C
- [ ] Help is shown on first launch

### Phase 3 Complete When:
- [ ] Relevant memories auto-retrieved on each message
- [ ] Checkpoints saved every 20 messages
- [ ] Reasoning displayed in ThoughtPanel (toggleable)
- [ ] Dream state streams tokens

### Phase 4 Complete When:
- [ ] MemoryServer split into <300 line modules
- [ ] Ollama backend works via provider abstraction
- [ ] External MCP servers can be registered

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Streaming fix introduces new bugs | Extensive testing in Phase 1 |
| Memory fixes break existing flow | Keep old code paths, gradual migration |
| Refactor scope creep | Strict module boundaries, incremental extraction |
| Provider abstraction complexity | Start with Elpis+Ollama only |

---

## Not In Scope

These items were identified but explicitly deferred:

1. **Multi-agent support** - Architectural complexity too high
2. **Full Letta adoption** - Would require rewrite; adopt patterns instead
3. **FocusLLM implementation** - Not applicable (training-time modification)
4. **New emotion modulation** - Current system adequate
5. **Mobile/web UI** - TUI focus maintained

---

## Appendix: Source Documents

| Document | Location |
|----------|----------|
| Bug Investigation Synthesis | `scratchpad/bug-investigation/synthesis-report.md` |
| Elpis Bug Report | `scratchpad/bug-investigation/elpis-agent-report.md` |
| Mnemosyne Bug Report | `scratchpad/bug-investigation/mnemosyne-agent-report.md` |
| Psyche Bug Report | `scratchpad/bug-investigation/psyche-agent-report.md` |
| Streaming Stability Plan | `scratchpad/plans/2026-01-14-streaming-stability-plan.md` |
| Memory Summarization Plan | `scratchpad/plans/20260114-memory-summarization-plan.md` |
| Stability Fixes Report | `scratchpad/reports/2026-01-14-stability-fixes.md` |
| Threading Fix Report | `scratchpad/reports/2026-01-16-threading-fix.md` |
| Architecture Final Report | `scratchpad/psyche-architecture-review/final-architecture-report.md` |
| Codebase Review | `scratchpad/psyche-architecture-review/codebase-review-report.md` |
| Coding Agents Review | `scratchpad/psyche-architecture-review/coding-agents-review-report.md` |
| Memory Systems Review | `scratchpad/psyche-architecture-review/memory-systems-review-report.md` |
| Reasoning Workflows Review | `scratchpad/psyche-architecture-review/reasoning-workflows-review-report.md` |
| Improvement Ideas | `scratchpad/ideas/psyche-improvements.md` |
| GitHub Issues Draft | `scratchpad/bug-investigation/github-issues.md` |

---

## Session Summary

| Phase | Sessions | Focus |
|-------|----------|-------|
| Phase 1 | 5 | Critical fixes (stability, memory) |
| Phase 2 | 4 | UX quick wins (tools, interruption) |
| Phase 3 | 3 | Memory & reasoning |
| Phase 4 | 8 | Architecture evolution |
| **Total** | **20** | |

**Recommended Start**: Session 1 (test streaming fix + async bugs)

Each session is a coherent unit of work that should leave the codebase in a testable, working state.
