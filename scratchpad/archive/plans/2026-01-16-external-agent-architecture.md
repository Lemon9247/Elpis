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
- External agent architecture discussion (2026-01-16): New interoperability approach

### Current State

| Component | Status | Blockers |
|-----------|--------|----------|
| Elpis Streaming | FIXED | Threading removed, needs testing |
| Memory Storage | BROKEN | `_staged_messages` never populated |
| Tool Display | Poor UX | Raw JSON shown |
| Interruption | Missing | Cannot cancel streaming/tools |
| Reasoning | Missing | No thinking step |
| Interoperability | Limited | Hardcoded MCP servers, no external agent support |

### Work Tracks

```
Track A: Stability & Bug Fixes (P0)
Track B: Memory System Overhaul (P0-P1)
Track C: User Experience (P1-P2)
Track D: Architecture Evolution (P2-P3)
Track E: External Agent Support (P3)
```

---

## Architecture Discussion: External Agent Support

### Original Proposal (Rejected)

The initial architecture review proposed **D3: Provider Abstraction** - making Psyche able to consume multiple LLM backends (Elpis, Ollama, Anthropic, etc.):

```python
# Original proposal - Psyche consumes multiple backends
class LLMProvider(ABC):
    async def complete(self, messages, tools=None) -> AsyncIterator: ...

class ElpisProvider(LLMProvider): ...
class OllamaProvider(LLMProvider): ...
```

### Reframe: Elpis as Backend for External Tools

The actual goal is the inverse: **make Elpis usable as the LLM backend for other coding agent harnesses** (Aider, Continue, Cursor, Claude Code, etc.).

### Research Findings

Investigation into coding agent APIs revealed:

1. **OpenAI-compatible `/v1/chat/completions`** is the de-facto standard
2. Nearly all coding agents support this format (Aider, Continue, Cursor, Cline, Cody)
3. Minimum viable API surface:
   - `POST /v1/chat/completions` (with SSE streaming)
   - `GET /v1/models` (model discovery)
4. MCP is emerging as a standard for agent tool interfaces

### Adopted Architecture

Keep existing Psyche ↔ Elpis MCP communication, and add:
1. **HTTP API on Elpis** for OpenAI-compatible inference
2. **Emotion Coordinator MCP server** for external agents to access emotion + memory

```
┌─────────────────────────────────────────────────────────────────┐
│                         ELPIS PROCESS                          │
│  ┌──────────────────┐      ┌──────────────────┐                │
│  │   MCP Server     │      │   HTTP Server    │                │
│  │   (stdio)        │      │   (:8000)        │                │
│  │                  │      │                  │                │
│  │  - generate      │      │  POST /v1/chat/  │                │
│  │  - generate_stream│     │    completions   │                │
│  │  - get_emotion   │      │  GET /v1/models  │                │
│  │  - process_event │      │  GET /v1/emotion │                │
│  │                  │      │  POST /v1/emotion│                │
│  │                  │      │  (+ tool calling)│                │
│  └────────┬─────────┘      └────────┬─────────┘                │
│           │                         │                          │
│           └──────────┬──────────────┘                          │
│                      ▼                                         │
│           ┌──────────────────────┐                             │
│           │   InferenceEngine    │                             │
│           │   EmotionalState     │                             │
│           │   HomeostasisReg.    │                             │
│           └──────────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
        ▲                    ▲                    ▲
        │ MCP/stdio          │ HTTP               │ MCP/stdio
        │                    │                    │
   ┌────┴────┐          ┌────┴────┐         ┌────┴────────────┐
   │ Psyche  │          │ Aider   │         │ Emotion Coord.  │
   │ (TUI)   │          │ Continue│         │ MCP Server      │
   └────┬────┘          │ Cursor  │         └────────┬────────┘
        │               │ etc.    │                  │
        │ MCP/stdio     └─────────┘                  │ MCP/stdio
        ▼                                            ▼
   ┌──────────┐                             ┌───────────┐
   │Mnemosyne │                             │ Mnemosyne │
   └──────────┘                             └───────────┘
```

**Benefits:**
- Psyche unchanged (no refactor needed)
- External agents get inference via standard HTTP
- MCP-aware agents (Claude Code, etc.) get full emotion + memory integration
- Clean separation of concerns

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
- Session 1: A1 + A2.1 + A2.2 (streaming test + async bugs)
- Session 2: A2.3 + A2.4 (exception handling + health monitoring)

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

**Track B Total**: 3 sessions
- Session 1: B1.1 + B1.2 + B1.3 (memory persistence fixes)
- Session 2: B2.1 + B2.2 (retrieval + checkpoints)
- Session 3: B2.3 (structured summarization + testing)

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

---

### C3. Help & Discoverability (P2)

#### C3.1 Show Help on Startup
First launch or empty input shows available commands.

#### C3.2 Command Aliases
Document and advertise short aliases (`/q`, `/h`, `/c`).

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
- Session 1: C1 (tool display)
- Session 2: C2.1 + C2.3 (streaming interruption + keybindings)
- Session 3: C2.2 (tool interruption)
- Session 4: C3 + C4 (help/aliases + dream streaming)

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

---

### D3. MCP Standardization (P3)

Route ALL tools through MCP, including internal ones.
Add dynamic MCP server registration.

**Track D Total**: 6 sessions
- Session 1: D1 (reasoning workflow)
- Session 2: D2 part 1 (extract inference.py + context_manager.py)
- Session 3: D2 part 2 (extract tool_handler.py + memory_handler.py)
- Session 4: D2 part 3 (wire up, test, fix integration)
- Session 5: D3 part 1 (internal tools as MCP)
- Session 6: D3 part 2 (dynamic server registration + config)

---

## Track E: External Agent Support

### E1. HTTP API Layer (P3)

Add OpenAI-compatible HTTP server to Elpis process.

**Files to modify:**
- `src/elpis/server.py` - Add HTTP server startup
- `src/elpis/http/` (new) - HTTP route handlers
- `src/elpis/cli.py` - Add `--http-port` flag

**Endpoints:**
```python
# POST /v1/chat/completions
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    ctx = get_context()
    if request.stream:
        return StreamingResponse(
            stream_completion(ctx, request),
            media_type="text/event-stream"
        )
    else:
        return await complete(ctx, request)

# GET /v1/models
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "elpis-local",
            "object": "model",
            "owned_by": "local"
        }]
    }

# GET /v1/emotion (Elpis-specific)
@app.get("/v1/emotion")
async def get_emotion():
    ctx = get_context()
    return {
        "valence": ctx.emotion_state.valence,
        "arousal": ctx.emotion_state.arousal
    }

# POST /v1/emotion/event (Elpis-specific)
@app.post("/v1/emotion/event")
async def process_event(event: EmotionEvent):
    ctx = get_context()
    ctx.regulator.process_event(event.event_type, event.intensity)
    return {"status": "ok"}
```

**Request format (OpenAI-compatible):**
```python
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[List[Tool]] = None      # OpenAI-format tool definitions
    tool_choice: Optional[str] = "auto"     # "auto", "none", or specific tool
```

**Authentication:**
```python
# Optional API key for network access
@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if not is_local_request(request) and settings.api_key:
        if request.headers.get("Authorization") != f"Bearer {settings.api_key}":
            return JSONResponse(status_code=401, content={"error": "Invalid API key"})
    return await call_next(request)
```

**Streaming response (SSE):**
```
data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}
data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"}}]}
data: [DONE]
```

**CLI usage:**
```bash
# Enable HTTP server alongside MCP (local only, no auth)
elpis-server --http-port 8000

# With API key for network access
elpis-server --http-port 8000 --api-key "your-secret-key"

# Or via environment
ELPIS_HTTP_PORT=8000 ELPIS_API_KEY="your-secret-key" elpis-server
```

---

### E2. Emotion Coordinator MCP Server (P3)

New MCP server that allows external agents to participate in Elpis's emotional feedback loop. When an external agent (Claude Code, Aider, etc.) uses Elpis as its backend, it can mount this MCP server to:

1. **Read emotional state** - See current valence/arousal for display or decision-making
2. **Send emotional events** - Report success/failure/frustration that affects future inference
3. **Access memory** - Store/recall memories via Mnemosyne

**New package:** `src/emotion_coordinator/`

```
src/emotion_coordinator/
├── __init__.py
├── cli.py           # Entry point: emotion-coordinator
├── server.py        # MCP server implementation
└── clients.py       # Elpis HTTP + Mnemosyne MCP clients
```

**pyproject.toml entry:**
```toml
[project.scripts]
emotion-coordinator = "emotion_coordinator.cli:main"
```

**MCP Tools exposed:**

```python
# Emotional state tools
Tool(
    name="get_emotional_state",
    description="Get current emotional state (valence, arousal)",
    inputSchema={"type": "object", "properties": {}}
)

Tool(
    name="process_emotional_event",
    description="Process an event that affects emotional state",
    inputSchema={
        "type": "object",
        "properties": {
            "event_type": {"type": "string", "enum": ["success", "failure", "frustration", "satisfaction", "neutral"]},
            "intensity": {"type": "number", "minimum": 0, "maximum": 1}
        }
    }
)

# Memory tools (proxy to Mnemosyne)
Tool(
    name="store_memory",
    description="Store a memory for later retrieval",
    inputSchema={
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "memory_type": {"type": "string", "enum": ["episodic", "semantic", "procedural"]},
            "importance": {"type": "number", "minimum": 0, "maximum": 1}
        }
    }
)

Tool(
    name="recall_memories",
    description="Recall relevant memories based on query",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "n_results": {"type": "integer", "default": 5}
        }
    }
)

# Coordination tools
Tool(
    name="get_context_summary",
    description="Get a summary of current emotional state and recent memories",
    inputSchema={"type": "object", "properties": {}}
)
```

---

### E3. Integration Testing (P3)

Test the full flow with real external agents.

**Test matrix:**

| Agent | HTTP Inference | MCP Emotion | Expected |
|-------|---------------|-------------|----------|
| Aider | ✓ | ✗ | Basic inference works |
| Continue | ✓ | ✓ | Full integration |
| Claude Code | ✓ | ✓ | Full integration |
| curl | ✓ | ✗ | API verification |

**Configuration examples:**

```bash
# Aider
aider --openai-api-base http://localhost:8000/v1 --model elpis-local
```

```json
// Continue (.continue/config.json)
{
  "models": [{
    "title": "Elpis Local",
    "provider": "openai",
    "model": "elpis-local",
    "apiBase": "http://localhost:8000/v1",
    "apiKey": "not-needed"
  }]
}
```

```json
// Claude Code (with MCP)
{
  "mcpServers": {
    "emotion-coordinator": {
      "command": "emotion-coordinator",
      "args": ["--elpis-url", "http://localhost:8000"]
    }
  }
}
```

**Track E Total**: 3 sessions
- Session 1: E1 (HTTP API with streaming)
- Session 2: E2 (Emotion Coordinator - emotion + memory tools)
- Session 3: E3 (integration testing + documentation)

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

### Phase 4: Architecture & Interoperability (7 sessions)

| Session | Track | Tasks | Deliverable |
|---------|-------|-------|-------------|
| 13 | D | D2 (part 1) | inference.py + context_manager.py extracted |
| 14 | D | D2 (part 2) | tool_handler.py + memory_handler.py extracted |
| 15 | D | D2 (part 3) | Refactor complete, tests pass |
| 16 | D | D3 (part 1) | Internal tools via MCP |
| 17 | D | D3 (part 2) | Dynamic server registration |
| 18 | E | E1 | HTTP API on Elpis |
| 19 | E | E2 | Emotion Coordinator complete |

### Phase 5: External Agent Integration (2 sessions)

| Session | Track | Tasks | Deliverable |
|---------|-------|-------|-------------|
| 20 | E | E3 | Verified with Aider, Continue |
| 21 | - | Final integration | All systems verified, documentation |

**Phase 5 Gate**: External agents can use Elpis with full emotion/memory support

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
| Steering hook leak | Elpis transformers | D3 (refactor) |
| Unreliable __del__ | Elpis transformers | D3 (refactor) |
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
- [ ] External MCP servers can be registered
- [ ] Elpis HTTP API responds to `/v1/chat/completions`
- [ ] Emotion Coordinator exposes emotion + memory tools

### Phase 5 Complete When:
- [ ] Aider can generate code using Elpis
- [ ] Continue can use Elpis with MCP emotion integration
- [ ] Documentation covers setup for 3+ external agents
- [ ] End-to-end test passes (inference + emotion + memory)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Streaming fix introduces new bugs | Extensive testing in Phase 1 |
| Memory fixes break existing flow | Keep old code paths, gradual migration |
| Refactor scope creep | Strict module boundaries, incremental extraction |
| HTTP API complexity | Start with minimal endpoints, expand later |
| External agent compatibility | Test with real tools early (E3) |

---

## Not In Scope

These items were identified but explicitly deferred:

1. **Multi-agent support** - Architectural complexity too high
2. **Full Letta adoption** - Would require rewrite; adopt patterns instead
3. **FocusLLM implementation** - Not applicable (training-time modification)
4. **New emotion modulation** - Current system adequate
5. **Mobile/web UI** - TUI focus maintained
6. **Provider abstraction for Psyche** - Replaced by external agent support (Track E)

---

## Design Decisions

1. **Auth for HTTP API**: Optional API key for network usage
   - Local (127.0.0.1): No auth required
   - Network: Require `ELPIS_API_KEY` environment variable or `--api-key` flag

2. **Tool calling via HTTP**: Yes, support OpenAI-format tool calling
   - Required for full agent compatibility
   - Include in E1 scope

3. **Emotion Coordinator location**: Separate package at `src/emotion_coordinator/`
   - Independent versioning
   - Clear separation from Elpis core
   - Can be installed separately if needed

4. **Emotion Coordinator purpose**: Allow external agents to participate in Elpis's emotional feedback loop
   - Read current emotional state (valence/arousal)
   - Send emotional events (success, failure, frustration) that affect future inference
   - Access memory via Mnemosyne proxy

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
| Phase 4 | 7 | Architecture & interoperability |
| Phase 5 | 2 | External agent integration |
| **Total** | **21** | |

**Track breakdown**: A(2) + B(3) + C(4) + D(6) + E(3) = 18 track sessions + 3 integration sessions = 21

**Recommended Start**: Session 1 (test streaming fix + async bugs)

Each session is a coherent unit of work that should leave the codebase in a testable, working state.
