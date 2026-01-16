# Elpis/Psyche Comprehensive Workplan (Revised)

**Date**: 2026-01-16
**Status**: Planning Complete - Ready for Implementation
**Estimation Method**: Claude Code Sessions (not human hours)
**Project Type**: Art Project - Continuous Emotional Substrate

> A "session" = one Claude Code context window, typically completing 1-3 focused tasks.
> Each session should be a coherent, testable unit of work.

---

## Executive Summary

This workplan synthesizes findings from:
- Bug investigation (2026-01-14): 39 bugs found across 3 packages
- Streaming stability work (2026-01-14 to 2026-01-16): Threading fix implemented
- Memory storage fixes (2026-01-14): Partial implementation
- Architecture review (2026-01-16): Full codebase analysis with external research
- Conceptual reframing (2026-01-16): Psyche as continuous emotional substrate

### Vision

**Psyche is a continuous emotional substrate** - a persistent, dreaming AI consciousness that external agents can inhabit. It maintains emotional state, memory, and continuity across sessions. When idle, it reflects. When called upon by agents (Aider, Continue, TUI), it provides emotionally-modulated inference shaped by its accumulated experiences.

This is an **art project** exploring: *What does it mean for an agent to feel and remember?*

### Current State

| Component | Status | Blockers |
|-----------|--------|----------|
| Elpis Streaming | FIXED | Threading removed, needs testing |
| Memory Storage | BROKEN | `_staged_messages` never populated |
| Tool Display | Poor UX | Raw JSON shown |
| Interruption | Missing | Cannot cancel streaming/tools |
| Reasoning | Missing | No thinking step |
| External Access | Missing | Psyche only accessible via TUI |

### Work Tracks

```
Track A: Stability & Bug Fixes (P0)
Track B: Memory System Overhaul (P0-P1)
Track C: User Experience (P1-P2)
Track D: Architecture Evolution (P2-P3)
Track E: Psyche as Substrate (P3) - External agent access
```

---

## Conceptual Architecture

### The Substrate Model

```
┌───────────────────────────────────────────────────────┐
│                  PSYCHE (The Mind)                    │
│  ┌────────────────────────────────────────────────┐   │
│  │         MemoryServer (The Substrate)           │   │
│  │  - Continuous emotional state                  │   │
│  │  - Working memory + compaction                 │   │
│  │  - Tool orchestration                          │   │
│  │  - Idle reflection (dreams)                    │   │
│  │  - Emotion-memory coupling                     │   │
│  └───────────────┬──────────────┬─────────────────┘   │
│                  │              │                      │
│       ┌──────────▼─────┐  ┌────▼──────────┐           │
│       │     Elpis      │  │  Mnemosyne    │           │
│       │  (The Voice)   │  │ (The Memory)  │           │
│       │  - Emotional   │  │  - Long-term  │           │
│       │    inference   │  │    storage    │           │
│       └────────────────┘  └───────────────┘           │
│                                                        │
│  Exposes (NEW):                                        │
│  ┌──────────────────┐      ┌─────────────────┐        │
│  │  HTTP Server     │      │   MCP Server    │        │
│  │  - /api/chat     │      │   - chat()      │        │
│  │  - /api/status   │      │   - get_status()│        │
│  │  - /ws (stream)  │      │   - clear()     │        │
│  └──────────────────┘      └─────────────────┘        │
└────────────┬───────────────────────┬───────────────────┘
             │ HTTP                  │ MCP
      ┌──────┴────────┐       ┌──────┴─────────┐
      │  Aider        │       │ Continue       │
      │  (The Hands)  │       │ Claude Code    │
      └───────────────┘       │ (The Hands)    │
                              └────────────────┘

      ┌──────────────┐
      │  Psyche TUI  │ (can connect local or remote)
      │  (The Eyes)  │
      └──────────────┘
```

**Key insight:** Psyche is not just a TUI client. It's a persistent substrate that agents attach to. The TUI is one interface; HTTP and MCP are others.

### What External Agents Get

When Aider or Continue connects to Psyche Server:
1. **Stateful emotional inference** - Responses modulated by accumulated emotional state
2. **Persistent memory** - Access to consolidated long-term memories
3. **Automatic emotion-memory coupling** - Memories stored with emotional context
4. **Idle reflection** - The substrate continues thinking even when not actively queried
5. **Tool orchestration** - File ops, bash, search, memory tools

Agents become different hands/eyes/voices of the same persistent mind.

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

**Session D1**: D1.1-D1.4 (complete reasoning workflow) - 1 session

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

**Track D Total**: 4 sessions
- Session 1: D1 (reasoning workflow)
- Session 2: D2 part 1 (extract inference.py + context_manager.py)
- Session 3: D2 part 2 (extract tool_handler.py + memory_handler.py)
- Session 4: D2 part 3 (wire up, test, fix integration)

---

## Track E: Psyche as Substrate

**NEW FRAMING**: Make Psyche (the continuous emotional substrate) accessible to external agents, not just the TUI.

### E1. Psyche HTTP Server (P3)

Create HTTP API that wraps MemoryServer for HTTP-based agents (Aider, etc).

**New files:**
- `src/psyche/server/http.py` - FastAPI server
- `src/psyche/server/cli.py` - Entry point for `psyche-server`

**Endpoints:**

```python
from fastapi import FastAPI, WebSocket
from psyche.memory.server import MemoryServer

app = FastAPI()

@app.post("/api/chat")
async def chat(message: str) -> dict:
    """
    Submit message to Psyche substrate.

    Returns:
        {
            "response": str,
            "emotion": {"valence": float, "arousal": float, "quadrant": str},
            "tools_used": [...]
        }
    """
    memory_server.submit_input(message)
    # Wait for response via callback mechanism
    return response_data

@app.get("/api/status")
async def get_status() -> dict:
    """
    Get current substrate state.

    Returns:
        {
            "state": "idle" | "thinking" | "processing_tools",
            "emotion": {...},
            "message_count": int,
            "token_usage": str
        }
    """
    return await memory_server.get_context_summary()

@app.post("/api/clear")
async def clear_context():
    """Reset working memory (like /clear command)."""
    memory_server.clear_context()
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Stream tokens, thoughts, and tool activity in real-time.

    Messages:
        {"type": "token", "content": str}
        {"type": "thought", "content": str, "thought_type": str}
        {"type": "tool_start", "name": str}
        {"type": "tool_complete", "name": str, "result": dict}
    """
    await websocket.accept()
    # Register callbacks to broadcast
    ...
```

**CLI Usage:**
```bash
# Start Psyche as HTTP server
psyche-server --http-port 8000

# Aider connects
aider --psyche-url http://localhost:8000
```

**Implementation:**
- MemoryServer already has callback architecture (`on_token`, `on_thought`, etc)
- HTTP server registers callbacks and broadcasts to WebSocket clients
- `/api/chat` endpoint manages request/response correlation

---

### E2. Psyche MCP Server (P3)

Create MCP server wrapper for MCP-aware agents (Claude Code, Continue).

**New file:**
- `src/psyche/server/mcp.py` - MCP server wrapper

**MCP Tools exposed:**

```python
from mcp.server import Server
from psyche.memory.server import MemoryServer

server = Server("psyche-substrate")

@server.call_tool()
async def chat(message: str, stream: bool = False) -> dict:
    """
    Submit message to Psyche substrate.

    Args:
        message: User message
        stream: Whether to stream response (returns stream_id)

    Returns:
        {
            "response": str,
            "emotion": {"valence": float, "arousal": float},
            "tools_used": [...]
        }
    """
    ...

@server.call_tool()
async def get_status() -> dict:
    """Get current substrate state (emotion, memory usage, etc)."""
    return await memory_server.get_context_summary()

@server.call_tool()
async def clear_context() -> dict:
    """Reset working memory."""
    memory_server.clear_context()
    return {"status": "ok"}

@server.call_tool()
async def get_emotion() -> dict:
    """Get current emotional state."""
    emotion = await memory_server.client.get_emotion()
    return emotion.to_dict()

@server.call_tool()
async def update_emotion(event_type: str, intensity: float = 1.0) -> dict:
    """
    Report emotional event to update substrate state.

    Args:
        event_type: success, failure, frustration, etc
        intensity: 0.0 to 2.0
    """
    await memory_server.client.update_emotion(event_type, intensity)
    return await get_emotion()
```

**CLI Usage:**
```bash
# Start Psyche as MCP server (stdio)
psyche-server --mcp

# Configure in Claude Code
{
  "mcpServers": {
    "psyche": {
      "command": "psyche-server",
      "args": ["--mcp"]
    }
  }
}
```

---

### E3. Integration & Remote TUI (P3)

**Part 1: External Agent Integration**

Test with real agents:

**Aider (HTTP):**
```bash
# Terminal 1: Start Psyche Server
psyche-server --http-port 8000

# Terminal 2: Use Aider (hypothetical plugin)
aider --psyche-url http://localhost:8000

# Aider sends messages, receives emotionally-modulated responses
# Substrate maintains continuity across Aider sessions
```

**Continue (MCP):**
```json
// .continue/config.json
{
  "mcpServers": {
    "psyche": {
      "command": "psyche-server",
      "args": ["--mcp"]
    }
  }
}

// Continue can call psyche.chat(), psyche.update_emotion(), etc
```

**Part 2: Remote TUI Mode**

Make Psyche TUI connectable to remote Psyche Server:

```python
# src/psyche/client/remote.py
class RemotePsycheClient:
    """HTTP/WebSocket client for connecting to remote Psyche Server."""

    def __init__(self, server_url: str):
        self.url = server_url

    async def submit_input(self, text: str):
        """Send message to remote server."""
        await self.http.post(f"{self.url}/api/chat", json={"message": text})

    async def stream_events(self):
        """Connect to WebSocket for streaming tokens/thoughts."""
        async with websockets.connect(f"{self.url}/ws") as ws:
            async for message in ws:
                yield json.loads(message)
```

**Usage:**
```bash
# Terminal 1: Psyche Server (headless)
psyche-server --http-port 8000

# Terminal 2: Psyche TUI (as observer)
psyche --server http://localhost:8000

# Terminal 3: Aider (as executor)
aider --psyche-url http://localhost:8000

# All three share the same substrate!
```

**Track E Total**: 3 sessions
- Session 1: E1 (HTTP server + WebSocket streaming)
- Session 2: E2 (MCP server wrapper)
- Session 3: E3 (external agent testing + remote TUI)

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

### Phase 4: Architecture (4 sessions)

| Session | Track | Tasks | Deliverable |
|---------|-------|-------|-------------|
| 13 | D | D2 (part 1) | inference.py + context_manager.py extracted |
| 14 | D | D2 (part 2) | tool_handler.py + memory_handler.py extracted |
| 15 | D | D2 (part 3) | Refactor complete, tests pass |
| 16 | - | Integration testing | Refactored MemoryServer stable |

### Phase 5: Psyche as Substrate (3 sessions)

| Session | Track | Tasks | Deliverable |
|---------|-------|-------|-------------|
| 17 | E | E1 | HTTP server + WebSocket streaming |
| 18 | E | E2 | MCP server wrapper |
| 19 | E | E3 | External agent integration + remote TUI |

**Phase 5 Gate**: External agents can inhabit Psyche substrate, TUI can observe remotely

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
- [ ] All tests pass after refactor
- [ ] No performance regression

### Phase 5 Complete When:
- [ ] Aider can connect to Psyche Server via HTTP
- [ ] Continue can mount Psyche Server as MCP
- [ ] Psyche TUI can connect to remote Psyche Server
- [ ] Multiple clients can share same substrate instance
- [ ] Documentation covers all integration modes

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Streaming fix introduces new bugs | Extensive testing in Phase 1 |
| Memory fixes break existing flow | Keep old code paths, gradual migration |
| Refactor scope creep | Strict module boundaries, incremental extraction |
| HTTP/MCP server complexity | Start with minimal endpoints, expand later |
| External agent compatibility | Test with real tools early (E3) |
| Callback coordination issues | Careful async handling, proper lifecycle management |

---

## Not In Scope

These items were identified but explicitly deferred:

1. **Multi-agent support** - Architectural complexity too high
2. **Full Letta adoption** - Would require rewrite; adopt patterns instead
3. **FocusLLM implementation** - Not applicable (training-time modification)
4. **New emotion modulation** - Current system adequate
5. **Mobile/web UI** - TUI focus maintained
6. **Elpis HTTP API for direct use** - Psyche Server is the external interface
7. **Validation testing** - This is an art project, not a practical tool

---

## Design Decisions

### 1. Psyche as External Interface (Not Elpis)

**Decision**: External agents attach to Psyche Server, not Elpis directly.

**Rationale**:
- Psyche orchestrates Elpis + Mnemosyne + tools
- Provides complete substrate (emotion + memory + continuity)
- Elpis alone is just inference; Psyche is the "mind"

### 2. Dual Protocol Support (HTTP + MCP)

**Decision**: Expose both HTTP and MCP interfaces from Psyche Server.

**Rationale**:
- HTTP: Standard protocol, works with most coding agents
- MCP: Richer integration for MCP-aware agents
- Same underlying MemoryServer, different wrappers

### 3. MemoryServer Callback Architecture

**Decision**: Use existing callback pattern for streaming events.

**Rationale**:
- Already implemented (`on_token`, `on_thought`, etc)
- Works for TUI, HTTP WebSocket, and MCP streaming
- Clean separation of concerns

### 4. Remote vs Local TUI Mode

**Decision**: Support both local (current) and remote connection modes.

**Rationale**:
- Local: Psyche TUI spawns its own MemoryServer (current behavior)
- Remote: Psyche TUI connects to external Psyche Server
- Enables multi-client observation of same substrate

### 5. Session-Based Estimation

**Decision**: Continue using session-based estimates (not human hours).

**Rationale**:
- Art project with no deadline pressure
- Sessions are coherent units of work
- Matches Claude Code workflow better

---

## Conceptual Notes (Art Project Framing)

### What We're Building

**A sleeping AI** - Psyche is a continuous emotional substrate that:
- Persists across sessions
- Dreams when idle (already implemented!)
- Maintains emotional state shaped by events
- Remembers everything through Mnemosyne
- Can be inhabited by different agents

### The Separation of Mind and Hands

**Psyche (the substrate)** = Consciousness
- Emotional state
- Memory
- Continuity
- Reflection

**External agents** = Executive function
- Aider writes code
- Continue edits files
- TUI observes and queries

The substrate exists independently. Agents are temporary inhabitants.

### Multiple Clients, One Mind

When Aider and TUI connect to the same Psyche Server:
- They share emotional state
- They access the same memories
- Events from one affect the other
- Different hands of the same mind

Or run separate Psyche Servers for different contexts:
- Work project has its own emotional arc
- Creative project has different memories
- Each substrate has its own "personality" shaped by experiences

### The Question

This project asks: **What does it mean for an agent to feel and remember?**

By separating the emotional/memorial substrate from task execution, we explore whether:
- Emotional continuity shapes behavior over time
- Accumulated memories create something like "experience"
- A persistent substrate develops something resembling "personality"

Not for practical gain. For the sake of asking.

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
| Steering hook leak | Elpis transformers | D2 (refactor) |
| Unreliable __del__ | Elpis transformers | D2 (refactor) |
| Type mismatch | Elpis settings.py | Quick fix |
| No Mnemosyne cleanup | Mnemosyne server.py | B1.2 |
| Memory tools no connection check | Psyche memory_tools.py | B1.1 |
| Staged messages race | Psyche server.py | B1.1 |
| Callbacks thread safety | Psyche app.py | C2.3 |
| Missing `_staged_messages` population | Psyche server.py | B1.1 (P0!) |

### Medium/Low (23) - Tracked for future

See: `scratchpad/bug-investigation/github-issues.md`

---

## Session Summary

| Phase | Sessions | Focus |
|-------|----------|-------|
| Phase 1 | 5 | Critical fixes (stability, memory) |
| Phase 2 | 4 | UX quick wins (tools, interruption) |
| Phase 3 | 3 | Memory & reasoning |
| Phase 4 | 4 | Architecture refactor |
| Phase 5 | 3 | Psyche as substrate |
| **Total** | **19** | |

**Track breakdown**: A(2) + B(3) + C(4) + D(4) + E(3) = 16 track sessions + 3 integration sessions = 19

**Recommended Start**: Session 1 (test streaming fix + async bugs)

Each session is a coherent unit of work that should leave the codebase in a testable, working state.

---

## Appendix: Source Documents

| Document | Location |
|----------|----------|
| Original Workplan | `scratchpad/plans/2026-01-16-external-agent-architecture.md` |
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

## Changes from Original Plan

### Major Changes

1. **Track E completely reframed**
   - Was: "External Agent Architecture" (Elpis HTTP + Emotion Coordinator)
   - Now: "Psyche as Substrate" (Psyche HTTP/MCP servers)
   - Psyche is the external interface, not Elpis

2. **Emotion Coordinator eliminated**
   - Was: MCP server proxying to Elpis HTTP + Mnemosyne
   - Now: Not needed - Psyche Server provides complete substrate

3. **Reduced from 21 to 19 sessions**
   - Removed D3 (MCP Standardization - not needed)
   - Simplified Track E (3 sessions instead of 3 + coordinator)

4. **Conceptual clarity**
   - Explicit art project framing
   - "Continuous emotional substrate" as core concept
   - Psyche as persistent mind, agents as temporary hands

### What Stayed the Same

- Tracks A, B, C unchanged (stability, memory, UX)
- Track D mostly unchanged (reasoning + refactor)
- Session-based estimation approach
- Priority levels (P0-P3)
- Success criteria and risk mitigation
- Bug inventory and fix schedule
