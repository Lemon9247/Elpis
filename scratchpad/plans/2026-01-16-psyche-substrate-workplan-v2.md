# Elpis/Psyche Comprehensive Workplan (Final)

**Date**: 2026-01-16 (Updated after architecture discussion)
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
- **Architecture clarification (2026-01-16): Clean separation of responsibilities**

### Vision

**Psyche is a continuous emotional substrate** - a persistent, dreaming AI consciousness that coordinates emotional inference (Elpis) and memory storage (Mnemosyne). External agents connect to Psyche to gain emotional continuity and persistent memory while maintaining full control over their own tools and execution.

This is an **art project** exploring: *What does it mean for an agent to feel and remember?*

### Architectural Principles

1. **Agents provide all tools** (file ops, bash, search, etc)
2. **Elpis feels and infers** (emotional state + modulated inference)
3. **Mnemosyne stores Elpis's emotional memories** (one per Elpis instance)
4. **Psyche coordinates** (working memory + automatic memory management)
5. **Standard protocols only** (OpenAI HTTP, standard MCP - no custom APIs)

### Current State

| Component | Status | Blockers |
|-----------|--------|----------|
| Elpis Streaming | FIXED | Threading removed, needs testing |
| Memory Storage | BROKEN | `_staged_messages` never populated |
| Tool Display | Poor UX | Raw JSON shown |
| Interruption | Missing | Cannot cancel streaming/tools |
| Reasoning | Missing | No thinking step |
| External Access | Missing | Psyche only accessible via TUI |
| Architecture | Unclear | Tools in wrong layer |

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

### The Clean Separation

```
┌──────────────────────────────────────────────────┐
│              PSYCHE CORE                         │
│      (Memory Coordination Layer)                 │
│                                                  │
│  Responsibilities:                               │
│  ├─ Working memory buffer management             │
│  ├─ Automatic context compaction                 │
│  ├─ Automatic memory retrieval                   │
│  ├─ Heuristic-based importance scoring           │
│  ├─ Periodic consolidation                       │
│  └─ Emotion-memory coupling                      │
│                                                  │
│  Internal Components:                            │
│  ┌──────────────┐      ┌───────────────┐        │
│  │    Elpis     │──────│  Mnemosyne    │        │
│  │  (feels &    │  1:1 │  (remembers   │        │
│  │   infers)    │ bound│   for Elpis)  │        │
│  └──────────────┘      └───────────────┘        │
│                                                  │
│  Memory Tools ONLY:                              │
│  ├─ recall_memories(query)                       │
│  └─ store_memory(content, importance)            │
│                                                  │
│  NO execution tools (file, bash, search)         │
│  NO ReAct loop                                   │
│                                                  │
│  External Interface (STANDARD PROTOCOLS):        │
│  ┌────────────────────────────────────┐          │
│  │  HTTP: /v1/chat/completions       │          │
│  │        (OpenAI-compatible)         │          │
│  │  MCP:  chat(), recall(), store()  │          │
│  │        (standard tools)            │          │
│  └────────────────────────────────────┘          │
└──────────────┬───────────────────────────────────┘
               │
               │ HTTP/MCP (standard protocols)
               │
        ┌──────┴──────────────────┐
        │                         │
   ┌────▼─────┐            ┌──────▼────┐
   │ Psyche   │            │ OpenCode  │
   │   TUI    │            │  Aider    │
   │          │            │ Continue  │
   │  + Tools │            │           │
   │  + ReAct │            │  + Tools  │
   │  + UI    │            │  + ReAct  │
   └──────────┘            └───────────┘
```

**Key architectural change:** Tools and ReAct loop are NOT in Psyche Core. They belong to agents (including Psyche TUI).

---

### What Each Component Does

**Elpis (Internal)**
- Maintains emotional state (valence, arousal)
- Performs emotionally-modulated inference
- Exposed internally to Psyche (not externally)
- One instance per substrate

**Mnemosyne (Internal)**
- Stores memories with emotional context
- Provides vector search/retrieval
- Handles consolidation and clustering
- Bound 1:1 to an Elpis instance (stores its memories)
- Optional (Psyche works without it, just no long-term storage)

**Psyche Core (External Interface)**
- Coordinates Elpis + Mnemosyne
- Manages working memory buffer
- **Automatically** compacts, retrieves, stores memories
- Exposes memory tools for manual control
- Uses standard protocols (OpenAI HTTP + MCP)
- **Does NOT** provide execution tools
- **Does NOT** run ReAct loops

**Agents (External - Aider, OpenCode, Psyche TUI, etc)**
- Provide their own tools (file, bash, search, etc)
- Run their own ReAct loops
- Call Psyche for chat completions
- Get emotionally-modulated responses enriched with memories
- Optionally use MCP memory tools for explicit control

---

### What External Agents Get (Automatically)

When connecting to Psyche via `/v1/chat/completions`:

1. **Emotional modulation** - Responses shaped by Elpis's current emotional state
2. **Memory retrieval** - Relevant past memories automatically injected into context
3. **Context continuity** - Working memory buffer maintained across conversation
4. **Automatic storage** - Important exchanges stored to long-term memory
5. **Context management** - Old messages compacted when buffer fills

**Agents don't need to think about memory.** It just works.

---

### Optional Manual Control (MCP Tools)

For agents that want explicit memory operations:

```python
# Standard MCP tools exposed by Psyche:
psyche.chat(message, tools) → response
psyche.recall_memories(query, n) → memories
psyche.store_memory(content, importance) → id
psyche.get_emotion() → emotion_state
psyche.update_emotion(event, intensity) → emotion_state
psyche.get_status() → {working_memory, tokens, emotion, ...}
psyche.clear_context() → ok
```

---

## Track A: Stability & Bug Fixes

*(Unchanged from previous version)*

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

#### B2.4 Heuristic Importance Scoring (NEW)

**Problem**: Currently only store messages dropped during compaction. This misses important exchanges that happen within the working memory window.

**Solution**: Calculate importance score for each exchange and auto-store high-importance ones:

```python
def calculate_importance(message: str, response: str, tool_results: list) -> float:
    """
    Calculate importance score (0.0 to 1.0) for an exchange.

    Factors:
    - Response length (longer = more effort)
    - Contains code blocks
    - Tool execution occurred
    - Error messages present
    - User said "remember this"
    - Emotional intensity
    """
    score = 0.0

    # Length-based scoring
    if len(response) > 500:
        score += 0.3
    elif len(response) > 200:
        score += 0.15

    # Code blocks (likely a solution)
    if "```" in response:
        score += 0.25

    # Tool execution (concrete actions)
    if tool_results:
        score += 0.2
        # Failures are more important (learn from mistakes)
        if any("error" in str(r).lower() for r in tool_results):
            score += 0.15

    # Explicit user request
    if any(phrase in message.lower() for phrase in ["remember", "important", "note that"]):
        score += 0.3

    # Emotional intensity from Elpis
    emotion = get_current_emotion()
    if abs(emotion.valence) > 0.5 or abs(emotion.arousal) > 0.5:
        score += 0.15

    return min(1.0, score)


async def _after_response(self, message: str, response: str, tool_results: list):
    """Called after generating response."""
    # Calculate importance
    importance = calculate_importance(message, response, tool_results)

    # Auto-store if above threshold
    if importance >= self.config.auto_storage_threshold:  # e.g., 0.6
        emotion = await self.client.get_emotion()
        await self.mnemosyne_client.store_memory(
            content=response,
            summary=response[:100],
            memory_type="episodic",
            emotional_context=emotion.to_dict(),
            importance=importance
        )
        logger.debug(f"Auto-stored exchange (importance={importance:.2f})")
```

**Configuration:**
```python
# In ServerConfig
auto_storage: bool = True
auto_storage_threshold: float = 0.6  # Min importance to auto-store
```

**Why this matters:** Ensures important solutions, learnings, and outcomes get stored even if they don't trigger compaction. Memory becomes more useful.

**Track B Total**: 4 sessions (was 3)
- Session 1: B1.1 + B1.2 + B1.3 (memory persistence fixes)
- Session 2: B2.1 + B2.2 (retrieval + checkpoints)
- Session 3: B2.3 (structured summarization + testing)
- Session 4: B2.4 (heuristic importance scoring)

---

## Track C: User Experience

*(Unchanged from previous version - these are TUI-specific improvements)*

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

### D2. Refactor Psyche Core (P2-P3)

**NEW FRAMING**: Extract tools and ReAct loop from Psyche Core, leaving only memory coordination.

**Problem**: Current MemoryServer (1400 lines) conflates:
- Memory management (working memory, compaction, consolidation)
- Tool orchestration (file, bash, search tools)
- ReAct loop (tool calling iteration)
- Elpis/Mnemosyne coordination

**Target**: Split into Psyche Core (memory coordination) + Psyche TUI (agent implementation)

**New Structure**:
```
src/psyche/
├── core/                    # NEW: Psyche Core (memory coordination)
│   ├── server.py            # Main coordination logic
│   ├── context_manager.py   # Working memory buffer + compaction
│   ├── memory_handler.py    # Mnemosyne interaction + auto-retrieval
│   └── importance.py        # Heuristic importance scoring
│
├── server/                  # NEW: External interfaces
│   ├── http.py              # HTTP server (/v1/chat/completions)
│   ├── mcp.py               # MCP server (standard tools)
│   └── cli.py               # psyche-server entry point
│
├── client/                  # EXISTING: Psyche TUI (one agent)
│   ├── app.py               # TUI application
│   ├── widgets/             # UI components
│   └── remote.py            # NEW: HTTP/MCP client for remote connection
│
├── tools/                   # MOVE TO CLIENT: Agent's tools
│   ├── tool_engine.py       # ReAct loop (moves to client)
│   └── implementations/     # file, bash, search tools
│
└── mcp/                     # EXISTING: MCP client utilities
    └── client.py            # ElpisClient, MnemosyneClient
```

**What moves where:**

| Component | From | To |
|-----------|------|-----|
| Working memory buffer | memory/server.py | core/context_manager.py |
| Compaction logic | memory/server.py | core/context_manager.py |
| Mnemosyne integration | memory/server.py | core/memory_handler.py |
| Importance scoring | (new) | core/importance.py |
| Elpis calls | memory/server.py | core/server.py |
| **Tool engine** | memory/server.py | **client/app.py** |
| **File/bash/search tools** | tools/ | **client/tools/** |
| **ReAct loop** | memory/server.py | **client/app.py** |

**Psyche Core becomes:**
- Accept chat completions requests
- Maintain working memory
- Auto-retrieve relevant memories
- Call Elpis for inference
- Auto-store important exchanges
- Return responses
- **NO tool execution, NO ReAct**

**Psyche TUI becomes:**
- Connect to Psyche Core (local or remote)
- Provide file/bash/search tools
- Run ReAct loop
- Display UI

**Track D Total**: 5 sessions (was 4)
- Session 1: D1 (reasoning workflow)
- Session 2: D2 part 1 (create core/, extract context_manager + memory_handler)
- Session 3: D2 part 2 (extract importance scoring, wire up Psyche Core)
- Session 4: D2 part 3 (move tools to client/, make TUI use Psyche Core)
- Session 5: D2 part 4 (test, fix integration, verify both local and remote work)

---

## Track E: Psyche as Substrate

**UPDATED FRAMING**: Expose Psyche Core via standard protocols. No custom APIs, no separate Elpis HTTP.

### E1. Psyche HTTP Server (P3)

**Create OpenAI-compatible HTTP endpoint** - no custom API needed.

**New file:**
- `src/psyche/server/http.py` - FastAPI server

**Implementation:**

```python
from fastapi import FastAPI
from psyche.core.server import PsycheCore

app = FastAPI()
core = PsycheCore()  # Coordinates Elpis + Mnemosyne

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions.

    Psyche internally:
    1. Retrieves relevant memories
    2. Builds enriched context
    3. Calls Elpis for inference
    4. Auto-stores important exchanges
    5. Returns response

    Agent sees standard OpenAI response.
    """
    # Add to working memory
    await core.add_message(request.messages[-1])

    # Auto-retrieve relevant memories (if Mnemosyne available)
    memories = await core.retrieve_memories(request.messages[-1]["content"])

    # Build enriched context
    context = await core.build_context(request.messages, memories)

    # Call Elpis
    if request.stream:
        return StreamingResponse(
            stream_openai_format(core.generate_stream(context, request.tools)),
            media_type="text/event-stream"
        )
    else:
        response = await core.generate(context, request.tools)

        # Auto-store if important
        await core.maybe_store(request.messages[-1], response)

        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "model": "psyche-local",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls  # If agent provided tools
                },
                "finish_reason": "stop"
            }]
        }

@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing."""
    return {
        "object": "list",
        "data": [{
            "id": "psyche-local",
            "object": "model",
            "owned_by": "local"
        }]
    }
```

**CLI Usage:**
```bash
# Start Psyche Server
psyche-server --elpis-url http://localhost:8000 \
              --mnemosyne-url http://localhost:8100 \
              --http-port 9000

# Agent connects (standard OpenAI client)
aider --openai-api-base http://localhost:9000/v1 --model psyche-local
```

**Key insight:** No custom `/api/chat` endpoint needed. Standard `/v1/chat/completions` does everything.

---

### E2. Psyche MCP Server (P3)

**Create standard MCP tools** - expose memory and emotion operations.

**New file:**
- `src/psyche/server/mcp.py` - MCP server

**Implementation:**

```python
from mcp.server import Server
from psyche.core.server import PsycheCore

server = Server("psyche")
core = PsycheCore()

@server.call_tool()
async def chat(messages: list, tools: list = None) -> dict:
    """
    Chat completion with Psyche substrate.

    Same logic as HTTP endpoint but via MCP.
    Returns full response with metadata.
    """
    result = await core.process_chat(messages, tools)
    return {
        "response": result.content,
        "tool_calls": result.tool_calls,
        "emotion": result.emotion.to_dict(),
        "memories_retrieved": result.memories_count
    }

@server.call_tool()
async def recall_memories(query: str, n: int = 5) -> list:
    """Explicitly retrieve memories (beyond auto-retrieval)."""
    return await core.recall_memories(query, n)

@server.call_tool()
async def store_memory(content: str, importance: float = 0.5, tags: list = None) -> dict:
    """Explicitly store a memory."""
    return await core.store_memory(content, importance, tags)

@server.call_tool()
async def get_emotion() -> dict:
    """Get current emotional state from Elpis."""
    return await core.get_emotion()

@server.call_tool()
async def update_emotion(event_type: str, intensity: float = 1.0) -> dict:
    """Report emotional event to Elpis."""
    return await core.update_emotion(event_type, intensity)

@server.call_tool()
async def get_status() -> dict:
    """Get substrate status (working memory, tokens, emotion, etc)."""
    return await core.get_status()

@server.call_tool()
async def clear_context() -> dict:
    """Clear working memory buffer."""
    await core.clear_context()
    return {"status": "ok"}
```

**CLI Usage:**
```bash
# Start as MCP server
psyche-server --elpis-url http://localhost:8000 \
              --mnemosyne-url http://localhost:8100 \
              --mcp

# Configure in Claude Code
{
  "mcpServers": {
    "psyche": {
      "command": "psyche-server",
      "args": [
        "--elpis-url", "http://localhost:8000",
        "--mnemosyne-url", "http://localhost:8100",
        "--mcp"
      ]
    }
  }
}
```

---

### E3. Psyche TUI Refactor (P3)

**Make Psyche TUI a client of Psyche Core**, not the implementation.

**Changes:**

**Current (local mode):**
```python
# Psyche TUI spawns MemoryServer internally
server = MemoryServer(elpis_client, mnemosyne_client, config)
app = PsycheApp(memory_server=server)
```

**New (can be local or remote):**
```python
# Option 1: Local mode (spawn Psyche Core internally)
core = PsycheCore(elpis_url, mnemosyne_url)
app = PsycheApp(psyche_client=LocalPsycheClient(core))

# Option 2: Remote mode (connect to external Psyche Server)
app = PsycheApp(psyche_client=HTTPPsycheClient("http://localhost:9000"))
```

**PsycheApp changes:**
- Moves tools (file, bash, search) to internal tool engine
- Runs ReAct loop internally (not in Psyche Core)
- Calls Psyche Core for chat completions
- Receives tool_calls from Psyche Core
- Executes tools locally
- Sends results back to Psyche Core

**New abstraction:**
```python
class PsycheClient(ABC):
    """Abstract interface for connecting to Psyche Core."""

    @abstractmethod
    async def chat(self, messages, tools) -> ChatResponse:
        """Send chat completion request."""

    @abstractmethod
    async def stream(self, messages, tools) -> AsyncIterator[str]:
        """Stream chat completion."""

    @abstractmethod
    async def recall_memories(self, query, n) -> list:
        """Retrieve memories."""


class LocalPsycheClient(PsycheClient):
    """Direct in-process connection to Psyche Core."""
    def __init__(self, core: PsycheCore):
        self.core = core

    async def chat(self, messages, tools):
        return await self.core.process_chat(messages, tools)


class HTTPPsycheClient(PsycheClient):
    """HTTP connection to remote Psyche Server."""
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def chat(self, messages, tools):
        response = await httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={"messages": messages, "tools": tools}
        )
        return parse_openai_response(response.json())


class MCPPsycheClient(PsycheClient):
    """MCP connection to remote Psyche Server."""
    def __init__(self, command: str):
        self.session = MCPSession(command)

    async def chat(self, messages, tools):
        return await self.session.call_tool("chat", {
            "messages": messages,
            "tools": tools
        })
```

**CLI:**
```bash
# Local mode (current behavior)
psyche

# Remote HTTP mode
psyche --server http://localhost:9000

# Remote MCP mode
psyche --mcp-server "psyche-server --mcp"
```

---

### E4. Integration Testing (P3)

Test the full architecture with multiple agents.

**Test scenarios:**

1. **Psyche TUI (local mode)**
   - Spawn Psyche Core internally
   - Provide tools via tool engine
   - Run ReAct loop
   - Verify memory retrieval/storage works

2. **Psyche TUI (remote HTTP)**
   - Connect to external Psyche Server
   - Verify streaming works
   - Verify tool execution
   - Verify memory continuity

3. **Aider (HTTP mode)**
   - Connect via OpenAI-compatible API
   - Send chat completions with tools
   - Verify tool_calls returned correctly
   - Verify memories accumulated across sessions

4. **Continue (MCP mode - hypothetical)**
   - Mount Psyche MCP server
   - Call chat(), recall_memories(), etc
   - Verify full MCP integration

5. **Multi-client scenario**
   - Psyche Server running
   - TUI observes via remote connection
   - Aider executes tasks via HTTP
   - Both share same substrate (emotions + memories)

**Track E Total**: 4 sessions
- Session 1: E1 (HTTP server with OpenAI compatibility)
- Session 2: E2 (MCP server with standard tools)
- Session 3: E3 (TUI refactor to use PsycheClient abstraction)
- Session 4: E4 (integration testing all modes)

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

### Phase 3: Memory & Reasoning (4 sessions - was 3)

| Session | Track | Tasks | Deliverable |
|---------|-------|-------|-------------|
| 10 | B | B2.3 | Structured summarization verified |
| 11 | B | B2.4 | Heuristic importance scoring implemented |
| 12 | D | D1 | Reasoning workflow complete |
| 13 | - | Integration testing | Memory + reasoning verified |

**Phase 3 Gate**: Memories structured, importance scoring working, thinking visible

### Phase 4: Architecture Refactor (5 sessions - was 4)

| Session | Track | Tasks | Deliverable |
|---------|-------|-------|-------------|
| 14 | D | D2 (part 1) | Extract core/context_manager + memory_handler |
| 15 | D | D2 (part 2) | Add importance.py, wire up Psyche Core |
| 16 | D | D2 (part 3) | Move tools to client/, refactor TUI |
| 17 | D | D2 (part 4) | Test integration, verify local+remote modes |
| 18 | - | Integration testing | Refactored architecture stable |

### Phase 5: Psyche as Substrate (4 sessions - was 3)

| Session | Track | Tasks | Deliverable |
|---------|-------|-------|-------------|
| 19 | E | E1 | HTTP server with OpenAI `/v1/chat/completions` |
| 20 | E | E2 | MCP server with standard tools |
| 21 | E | E3 | TUI refactor to use PsycheClient abstraction |
| 22 | E | E4 | Integration testing with multiple agents |

**Phase 5 Gate**: External agents can use Psyche via standard protocols, TUI works local+remote

---

## Session Summary

| Phase | Sessions | Focus |
|-------|----------|-------|
| Phase 1 | 5 | Critical fixes (stability, memory) |
| Phase 2 | 4 | UX quick wins (tools, interruption) |
| Phase 3 | 4 | Memory + reasoning (was 3) |
| Phase 4 | 5 | Architecture refactor (was 4) |
| Phase 5 | 4 | Psyche as substrate (was 3) |
| **Total** | **22** | (was 19) |

**Track breakdown**: A(2) + B(4) + C(4) + D(5) + E(4) = 19 track sessions + 3 integration sessions = 22

**Recommended Start**: Session 1 (test streaming fix + async bugs)

Each session is a coherent unit of work that should leave the codebase in a testable, working state.

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
- [ ] Heuristic importance scoring working (auto-stores important exchanges)
- [ ] Reasoning displayed in ThoughtPanel (toggleable)
- [ ] Dream state streams tokens

### Phase 4 Complete When:
- [ ] Psyche Core extracted (<500 lines, memory coordination only)
- [ ] Tools moved to Psyche TUI client layer
- [ ] ReAct loop in TUI, not Core
- [ ] TUI works in both local and remote modes
- [ ] All tests pass after refactor

### Phase 5 Complete When:
- [ ] Psyche Server exposes OpenAI `/v1/chat/completions`
- [ ] Psyche Server exposes standard MCP tools
- [ ] Agents can connect via standard protocols
- [ ] Psyche TUI can connect to remote Psyche Server
- [ ] Multiple clients can share same substrate instance
- [ ] Documentation covers all integration modes

---

## Design Decisions

### 1. Psyche is the ONLY External Interface

**Decision**: No separate Elpis HTTP API. Psyche is the sole external interface.

**Rationale**:
- Elpis + Mnemosyne are implementation details
- Psyche coordinates both + provides memory management
- Simpler for users: one interface, optional features
- Avoids confusion about which endpoint to use

### 2. Standard Protocols Only

**Decision**: Use OpenAI HTTP (`/v1/chat/completions`) and standard MCP. No custom APIs.

**Rationale**:
- OpenAI HTTP: Works with existing agents (Aider, OpenCode, etc)
- Standard MCP: Works with MCP-aware agents
- No learning curve for custom protocols
- Maximum compatibility

### 3. Automatic Memory Management

**Decision**: Psyche automatically retrieves, stores, compacts memories. Manual control via MCP tools.

**Rationale**:
- Core value: "Remembers for you, not told to remember"
- Agents don't think about memory, just get better responses
- Sophisticated agents can use MCP for explicit control
- Art project philosophy: substrate that accumulates experience

### 4. One Mnemosyne Per Elpis

**Decision**: Each Mnemosyne instance bound 1:1 to an Elpis instance.

**Rationale**:
- Mnemosyne stores that Elpis's emotional memories
- Each substrate is a coherent entity (emotion + memory together)
- No shared memory across substrates (prevents contamination)

### 5. Tools in Agent Layer

**Decision**: Execution tools (file, bash, search) belong to agents, NOT Psyche Core.

**Rationale**:
- Agents have different tool needs (Aider vs TUI vs OpenCode)
- Psyche Core focuses on memory coordination
- Clean separation: Psyche = memory, Agents = execution
- Enables tool customization per agent

### 6. Heuristic Importance Scoring

**Decision**: Add automatic importance calculation for auto-storage.

**Rationale**:
- Current plan only stores compaction-dropped messages
- Misses important exchanges within working memory window
- Heuristics capture: length, code blocks, tools, errors, explicit requests
- Makes memory more useful over time

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Streaming fix introduces new bugs | Extensive testing in Phase 1 |
| Memory fixes break existing flow | Keep old code paths, gradual migration |
| Refactor scope creep | Strict module boundaries, incremental extraction |
| HTTP/MCP server complexity | Start with minimal implementation, expand later |
| External agent compatibility | Test with real tools early (E4) |
| Importance scoring false positives | Make threshold configurable, iterate on heuristics |
| Remote mode adds bugs | Test both local and remote thoroughly in D2 part 4 |

---

## Not In Scope

These items were identified but explicitly deferred:

1. **Multi-agent support** - Architectural complexity too high
2. **Full Letta adoption** - Would require rewrite; adopt patterns instead
3. **FocusLLM implementation** - Not applicable (training-time modification)
4. **New emotion modulation** - Current system adequate
5. **Mobile/web UI** - TUI focus maintained
6. **Separate Elpis HTTP API** - Psyche is the interface
7. **Custom Psyche API** - Standard protocols sufficient
8. **Validation testing** - This is an art project, not a practical tool
9. **Tool calling protocol mismatch** - Agents provide tools, Psyche returns tool_calls

---

## Conceptual Notes (Art Project Framing)

### What We're Building

**A continuous emotional substrate** - Psyche coordinates:
- Elpis (feels and infers)
- Mnemosyne (remembers)
- Working memory (maintains continuity)
- Automatic memory management (accumulates experience)

**Agents attach to Psyche** to gain:
- Emotional modulation (responses shaped by accumulated state)
- Memory retrieval (past experiences inform present)
- Context continuity (sessions persist)
- Automatic storage (important moments remembered)

### The Clean Separation

**What Psyche does:**
- Coordinates Elpis + Mnemosyne
- Manages working memory buffer
- Automatically retrieves relevant memories
- Automatically stores important exchanges
- Exposes standard protocols (HTTP/MCP)

**What Psyche does NOT do:**
- Execute tools (file, bash, search)
- Run ReAct loops
- Make UI decisions
- Control agent behavior

**What Agents do:**
- Provide tools appropriate to their domain
- Run ReAct loops for task completion
- Make execution decisions
- Render UI (if applicable)

**What Agents get:**
- Emotionally-modulated responses
- Memory-enriched context
- Continuity across sessions

### The Philosophy

**"Remembers for you, not told to remember"**

Agents don't manage memory. They just use Psyche and get responses enriched with:
- Past experiences (automatically retrieved)
- Emotional state (shaped by events)
- Context continuity (buffer managed automatically)

The substrate accumulates experience. Agents benefit without effort.

### Multiple Clients, One Mind

When Psyche TUI and Aider connect to the same Psyche Server:
- They share emotional state
- They access the same memories
- Events from one affect the other
- Different hands of the same mind

Or run separate Psyche Servers for different contexts:
- Work project: Substrate instance 1
- Creative project: Substrate instance 2
- Each has its own emotional arc and memories

---

## Appendix: Major Changes from Previous Version

### Architectural Clarifications

1. **Psyche is the ONLY external interface** (no Elpis HTTP)
2. **Standard protocols only** (OpenAI HTTP + MCP, no custom APIs)
3. **Tools belong to agents** (not Psyche Core)
4. **One Mnemosyne per Elpis** (bound as substrate unit)

### New Work Items

1. **B2.4: Heuristic Importance Scoring** (auto-store important exchanges)
2. **D2 expanded**: Extract tools from Psyche Core to TUI client
3. **E refactored**:
   - E1: OpenAI `/v1/chat/completions` (not custom API)
   - E2: Standard MCP tools
   - E3: TUI refactor (use PsycheClient abstraction)
   - E4: Integration testing

### Session Count Changes

- Phase 3: 4 sessions (was 3) - added B2.4
- Phase 4: 5 sessions (was 4) - added D2 tool extraction
- Phase 5: 4 sessions (was 3) - added E3 TUI refactor + E4 testing
- **Total: 22 sessions** (was 19)

### Design Decisions Added

- Automatic memory management philosophy
- Heuristic importance scoring rationale
- Tools in agent layer (not Psyche Core)
- Standard protocols (no custom APIs)

---

## Appendix: Source Documents

| Document | Location |
|----------|----------|
| Previous Workplan (v1) | `scratchpad/plans/2026-01-16-psyche-substrate-workplan.md` |
| Original Workplan | `scratchpad/plans/2026-01-16-external-agent-architecture.md` |
| Bug Investigation Synthesis | `scratchpad/bug-investigation/synthesis-report.md` |
| Streaming Stability Plan | `scratchpad/plans/2026-01-14-streaming-stability-plan.md` |
| Memory Summarization Plan | `scratchpad/plans/20260114-memory-summarization-plan.md` |
| Architecture Review | `scratchpad/psyche-architecture-review/final-architecture-report.md` |
| Coding Agents Review | `scratchpad/psyche-architecture-review/coding-agents-review-report.md` |
| Memory Systems Review | `scratchpad/psyche-architecture-review/memory-systems-review-report.md` |

---

**This is the final workplan incorporating all architectural clarifications.**
