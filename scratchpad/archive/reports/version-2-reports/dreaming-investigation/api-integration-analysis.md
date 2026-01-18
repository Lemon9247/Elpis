# API Integration Analysis: Dreaming in Headless Psyche

**Date:** 2026-01-17
**Agent:** API Integration Analyst
**Status:** Complete

## Executive Summary

This analysis examines how dreaming should integrate with Psyche when running as a headless API server. The key challenge is balancing autonomous dreaming (a form of beneficial background processing) with responsive request handling in a sequential, single-client-at-a-time architecture.

The recommended approach is a **cooperative preemption model** where dreaming runs during idle periods but can be cleanly interrupted when requests arrive, with dreams yielding gracefully to ensure responsive API behavior.

---

## Current Architecture Review

### Existing Components

**PsycheCore** (`src/psyche/core/server.py`)
- Memory coordination layer (~636 lines)
- Handles context management, memory retrieval/storage, emotional state
- Does NOT execute tools or run ReAct loops
- Clean separation of concerns - good foundation for headless mode

**IdleHandler** (`src/psyche/handlers/idle_handler.py`)
- Handles autonomous "dreaming" when user is idle (~722 lines)
- Generates reflective thoughts via `generate_thought()`
- Uses rate limiting (warmup period, cooldowns)
- Has interrupt mechanism via `_interrupt_event` and `interrupt()` method
- Manages memory consolidation during idle periods

**ReactHandler** (`src/psyche/handlers/react_handler.py`)
- Processes user input through ReAct loop (~544 lines)
- Handles tool execution, streaming responses
- Also has interrupt mechanism

**Hermes TUI** (`src/hermes/app.py`)
- Current client implementation using Textual
- Manages state machine: IDLE -> PROCESSING -> THINKING -> DISCONNECTED
- Runs idle loop via `_run_idle_loop()` with sleep-based scheduling

**Client Abstractions** (`src/psyche/handlers/psyche_client.py`)
- `PsycheClient` ABC defines the interface
- `LocalPsycheClient` wraps PsycheCore for in-process use
- `RemotePsycheClient` is a stub for Phase 5

### Key Constraint: One Client at a Time

The system is designed for **one Psyche instance, one client at a time** (sequential, not concurrent). This simplifies the dreaming integration significantly:
- No need for per-client dream state
- No request routing complexity
- Single event loop for dream/request scheduling

---

## When Does Dreaming Happen?

### Option 1: Between Client Sessions (Recommended Primary)

**When:** After a client disconnects and before the next client connects.

**Benefits:**
- Zero impact on request latency
- Dreams can run to completion without interruption
- Natural fit for memory consolidation
- Can explore with tools freely (no responsiveness concern)

**Implementation:**
```python
async def run_server():
    while True:
        # Dream while waiting for connection
        dream_task = asyncio.create_task(dream_loop())

        # Accept client connection
        client = await accept_client()

        # Cancel dreaming, handle client
        dream_task.cancel()
        await handle_client_session(client)

        # Client disconnected, resume dreaming
```

### Option 2: During Idle Periods Within Session

**When:** Client is connected but hasn't sent a request for `post_interaction_delay` seconds (currently 30-60s).

**Benefits:**
- Maintains "presence" feeling for connected clients
- Can share dream insights with the current client
- Already implemented in IdleHandler

**Challenges:**
- Must be interruptible (handled by existing `interrupt()` mechanism)
- May create latency if interrupt is slow
- Resource contention if dream uses heavy inference

**Implementation:** Keep existing IdleHandler pattern but ensure dreams are lightweight and interruptible.

### Option 3: Scheduled Dreams (Background)

**When:** On a fixed schedule (e.g., hourly, daily).

**Benefits:**
- Predictable resource usage
- Can run deeper consolidation
- Doesn't depend on client activity patterns

**Challenges:**
- May conflict with client sessions
- Needs priority/preemption handling

### Recommended Strategy

**Hybrid approach with priorities:**

1. **Priority 1:** Request handling (always preempts everything)
2. **Priority 2:** In-session idle thinking (lightweight, fast to interrupt)
3. **Priority 3:** Between-session dreaming (can run longer, deeper)
4. **Priority 4:** Scheduled maintenance (memory consolidation, cleanup)

---

## Request Handling During Dreams

### Wake-Up Mechanism

The existing `IdleHandler.interrupt()` mechanism is well-designed:

```python
def interrupt(self) -> bool:
    """Request interruption of current idle thinking."""
    if self._is_thinking:
        self._interrupt_event.set()
        logger.debug("Idle thought interrupt requested")
        return True
    return False
```

**Current check points:**
- Before each iteration of the generation loop
- During token generation (checking `_interrupt_event.is_set()`)
- Before tool execution

**Enhancement needed:** Make interrupt checking more granular in the generation stream.

### Request Queuing Strategy

For a single-client-at-a-time model, queuing is simple:

```python
class RequestQueue:
    """Simple FIFO queue with dream preemption."""

    def __init__(self):
        self._queue: asyncio.Queue[Request] = asyncio.Queue()
        self._dream_interrupt = asyncio.Event()

    async def submit(self, request: Request) -> Response:
        """Submit request, interrupting any active dream."""
        self._dream_interrupt.set()
        await self._queue.put(request)
        return await request.wait_for_response()

    async def process_loop(self):
        """Main processing loop."""
        while True:
            try:
                # Wait for request with timeout (allows dream cycles)
                request = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self._idle_timeout
                )
                await self._handle_request(request)
            except asyncio.TimeoutError:
                # No request - time for dreaming
                await self._dream_cycle()
```

### Preventing Request Starvation

**Concern:** A long dream could delay request handling.

**Mitigations:**

1. **Soft interrupt:** Dreams check interrupt flag every N tokens
2. **Hard timeout:** Dreams have a maximum duration (e.g., 60s)
3. **Checkpoint-based:** Dreams save progress and can resume later
4. **Lightweight dreams:** In-session dreams are limited to 512 tokens

```python
class DreamBudget:
    """Manages dream duration and checkpointing."""

    def __init__(self, max_duration: float = 60.0, checkpoint_interval: float = 15.0):
        self.max_duration = max_duration
        self.checkpoint_interval = checkpoint_interval
        self._start_time: Optional[float] = None
        self._last_checkpoint: Optional[float] = None

    def should_yield(self) -> bool:
        """Check if dream should yield to potential requests."""
        if self._start_time is None:
            return False
        elapsed = time.time() - self._start_time
        return elapsed > self.max_duration

    def should_checkpoint(self) -> bool:
        """Check if dream should save progress."""
        if self._last_checkpoint is None:
            return True
        return time.time() - self._last_checkpoint > self.checkpoint_interval
```

---

## API Design Considerations

### Core API Endpoints

**Request/Response (existing patterns):**
```
POST /chat          - Send message, get response (with streaming)
GET  /emotion       - Current emotional state
POST /memory/store  - Store a memory
POST /memory/search - Search memories
```

**Dream-related additions:**

```
GET  /dream/status  - Query current dream state
POST /dream/wake    - Force wake-up (interrupt dream)
GET  /dream/journal - Recent dream summaries
POST /dream/config  - Update dream configuration
```

### Dream State Endpoint

```python
@dataclass
class DreamStatus:
    is_dreaming: bool
    dream_type: str  # "idle", "deep", "consolidation", "none"
    started_at: Optional[datetime]
    can_interrupt: bool
    current_focus: Optional[str]  # What is Psyche thinking about?

# GET /dream/status
{
    "is_dreaming": true,
    "dream_type": "idle",
    "started_at": "2026-01-17T15:30:00Z",
    "can_interrupt": true,
    "current_focus": "Exploring the project structure"
}
```

### Do Not Disturb Mode

Some scenarios may benefit from uninterruptible dreaming:

```python
@dataclass
class DreamConfig:
    enabled: bool = True
    allow_in_session: bool = True
    between_session_mode: str = "deep"  # "deep", "light", "off"
    do_not_disturb: bool = False  # If true, queue requests until dream complete
    dnd_max_duration: float = 300.0  # Max DND period (5 min)
```

**Use case:** User wants to let Psyche "sleep" and process memories without interruption.

```
POST /dream/dnd
{
    "duration_seconds": 300,
    "reason": "memory_consolidation"
}

Response:
{
    "dnd_until": "2026-01-17T15:35:00Z",
    "queue_position": 0  # Requests queued during DND
}
```

### Exposing Dream Configuration

```
GET /dream/config
{
    "post_interaction_delay": 30.0,
    "idle_tool_cooldown_seconds": 60.0,
    "startup_warmup_seconds": 120.0,
    "max_idle_tool_iterations": 3,
    "think_temperature": 0.9,
    "enable_consolidation": true,
    "consolidation_check_interval": 300.0
}

PATCH /dream/config
{
    "post_interaction_delay": 60.0,  # More patient before dreaming
    "enable_consolidation": false     # Disable auto-consolidation
}
```

---

## Resource Budgeting

### Inference Time Budget

**Concern:** Dreams consume the same GPU/inference resources as requests.

**Strategy:** Time-sliced approach with request priority.

```python
class InferenceBudget:
    """Manage inference time between requests and dreams."""

    def __init__(
        self,
        request_priority: float = 0.9,  # 90% budget for requests
        dream_priority: float = 0.1,    # 10% budget for dreams
        window_seconds: float = 60.0,   # Budget window
    ):
        self.request_budget = request_priority * window_seconds
        self.dream_budget = dream_priority * window_seconds
        self._request_usage = 0.0
        self._dream_usage = 0.0
        self._window_start = time.time()

    def can_dream(self, estimated_duration: float) -> bool:
        """Check if there's budget for a dream cycle."""
        self._maybe_reset_window()
        return self._dream_usage + estimated_duration <= self.dream_budget

    def record_request(self, duration: float) -> None:
        """Record request inference time."""
        self._request_usage += duration

    def record_dream(self, duration: float) -> None:
        """Record dream inference time."""
        self._dream_usage += duration
```

**Note:** In a single-client model, the budget is simpler - dreams just shouldn't interfere with requests. The above is more relevant for future multi-client scenarios.

### Memory Usage During Dreams

**Concern:** Long dreams could accumulate context.

**Mitigations:**

1. **Separate dream context:** Dreams don't share context with main conversation
2. **Context limits:** Dream context capped at N tokens
3. **Periodic flush:** Dream insights stored to Mnemosyne, context cleared

```python
class DreamContext:
    """Isolated context for dream sessions."""

    MAX_TOKENS = 4000  # Much smaller than main context

    def __init__(self):
        self._messages: List[Message] = []
        self._token_count = 0

    def add_thought(self, content: str) -> None:
        """Add a dream thought, enforcing limits."""
        tokens = count_tokens(content)
        while self._token_count + tokens > self.MAX_TOKENS:
            self._evict_oldest()
        self._messages.append(Message("assistant", content))
        self._token_count += tokens
```

### Preventing Resource Exhaustion

**Safeguards:**

1. **Dream timeout:** Maximum duration per dream cycle (60s default)
2. **Cooldown:** Minimum gap between dream cycles (30s default)
3. **Memory cap:** Maximum dream context size (4K tokens)
4. **Tool limits:** Maximum tool calls per dream (3 default, already implemented)
5. **Consolidation throttle:** Only consolidate every N minutes

---

## Proposed API Server Architecture

### State Machine

```
                          ┌─────────────────────────────────────┐
                          │                                     │
                          v                                     │
┌──────────┐         ┌─────────┐         ┌──────────┐          │
│  STARTUP │───────> │  IDLE   │───────> │ DREAMING │──────────┘
└──────────┘         └─────────┘         └──────────┘
      │                   │                    │
      │                   v                    │ (request arrives)
      │              ┌─────────┐               │
      │              │ WAITING │ <─────────────┘
      │              └─────────┘        (interrupt dream)
      │                   │
      │                   v
      │              ┌────────────┐
      └─────────────>│ PROCESSING │
                     └────────────┘
                          │
                          v (response complete)
                     ┌─────────┐
                     │  IDLE   │
                     └─────────┘
```

### Server Skeleton

```python
class PsycheServer:
    """Headless Psyche API server with dreaming support."""

    def __init__(self, core: PsycheCore, config: ServerConfig):
        self.core = core
        self.config = config
        self._state = ServerState.STARTUP
        self._dream_task: Optional[asyncio.Task] = None
        self._request_event = asyncio.Event()

    async def run(self):
        """Main server loop."""
        await self._initialize()

        while True:
            if self._state == ServerState.IDLE:
                # Wait for either request or dream timeout
                try:
                    await asyncio.wait_for(
                        self._request_event.wait(),
                        timeout=self.config.idle_before_dream
                    )
                    # Request arrived
                    await self._handle_pending_request()
                except asyncio.TimeoutError:
                    # Start dreaming
                    await self._start_dreaming()

            elif self._state == ServerState.DREAMING:
                # Check for interrupts, let dream run
                if self._request_event.is_set():
                    await self._interrupt_dream()
                    await self._handle_pending_request()
                else:
                    await asyncio.sleep(0.1)  # Yield to other tasks

    async def _start_dreaming(self):
        """Begin a dream cycle."""
        self._state = ServerState.DREAMING
        self._dream_task = asyncio.create_task(
            self._dream_cycle()
        )

    async def _interrupt_dream(self):
        """Wake up from dreaming."""
        if self._dream_task and not self._dream_task.done():
            # Signal interrupt to dream
            self._idle_handler.interrupt()
            # Give it a moment to yield gracefully
            try:
                await asyncio.wait_for(self._dream_task, timeout=1.0)
            except asyncio.TimeoutError:
                self._dream_task.cancel()
        self._state = ServerState.WAITING
```

### HTTP/WebSocket Interface

```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse

app = FastAPI(title="Psyche API")

@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Process chat message with streaming response."""
    server = get_server()
    server.wake_for_request()

    async def stream():
        async for token in server.process_message(request.message):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.get("/dream/status")
async def dream_status() -> DreamStatus:
    """Get current dream state."""
    server = get_server()
    return server.get_dream_status()

@app.post("/dream/wake")
async def wake_up() -> dict:
    """Force wake from dream."""
    server = get_server()
    was_dreaming = server.wake_for_request()
    return {"was_dreaming": was_dreaming}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket for real-time streaming and dream notifications."""
    await ws.accept()
    server = get_server()

    async with server.subscribe_to_events() as events:
        async for event in events:
            await ws.send_json(event.to_dict())
```

---

## Risks and Mitigations

### Risk 1: Dream Interrupt Latency

**Problem:** If a dream is mid-generation, interrupting may take time.

**Mitigation:**
- Check interrupt flag every 10 tokens (currently done)
- Add hard timeout on interrupt wait (1 second)
- In extreme cases, cancel the generation task

**Code:**
```python
async def _interrupt_dream(self):
    self._idle_handler.interrupt()
    try:
        await asyncio.wait_for(self._dream_task, timeout=1.0)
    except asyncio.TimeoutError:
        logger.warning("Dream did not yield in time, force cancelling")
        self._dream_task.cancel()
```

### Risk 2: State Corruption on Interrupt

**Problem:** Interrupting mid-dream could leave state inconsistent.

**Mitigation:**
- Dreams use isolated context (don't modify main conversation)
- Use transaction-like patterns for memory storage
- Implement cleanup in finally blocks

### Risk 3: Resource Exhaustion

**Problem:** Dreams running too often or too long consume resources.

**Mitigation:**
- Time budgets (already partially implemented in IdleHandler)
- Token limits on dream context
- Cooldown periods between dreams
- Configuration for operators to tune

### Risk 4: Dreams Leaking into Responses

**Problem:** Dream content accidentally included in user responses.

**Mitigation:**
- Strict context isolation
- Dream context cleared before handling requests
- Dreams stored separately in Mnemosyne with tags

### Risk 5: Tool Safety in Unattended Dreams

**Problem:** Dreams with tool access could do unintended things.

**Mitigation (already implemented):**
- `SAFE_IDLE_TOOLS` whitelist (read_file, list_directory, search_codebase, recall_memory)
- `SENSITIVE_PATH_PATTERNS` blocklist
- Path validation to stay within workspace
- No bash, no file writes during dreams

---

## Recommendations

### Immediate (Phase 5 MVP)

1. **Keep IdleHandler design:** The current interrupt mechanism is sound
2. **Add server state machine:** Implement IDLE/DREAMING/PROCESSING states
3. **HTTP endpoint for wake:** `POST /dream/wake` to interrupt dreams
4. **Dream status endpoint:** `GET /dream/status` for observability
5. **Configure via environment/config:** Expose dream settings

### Medium Term

1. **Dream journal:** Store dream summaries for later retrieval
2. **WebSocket events:** Real-time notifications of dream state changes
3. **Resource budgeting:** Implement time-sliced inference budget
4. **Configurable dream types:** Light (in-session) vs. deep (between-session)

### Long Term

1. **Dream insights surfacing:** Ways to share relevant dream discoveries with users
2. **Dream scheduling:** Cron-like scheduling for maintenance dreams
3. **Multi-client considerations:** If architecture changes, revisit queuing
4. **Dream quality metrics:** Measure if dreaming is actually beneficial

---

## Appendix: Comparison with Current TUI Implementation

| Aspect | Current TUI (Hermes) | Proposed Headless API |
|--------|---------------------|----------------------|
| Client connection | Always connected | Connect/disconnect per session |
| Dream trigger | Timer-based loop | Idle timeout or explicit |
| Interrupt mechanism | `IdleHandler.interrupt()` | Same + HTTP endpoint |
| State management | `AppState` enum | `ServerState` enum |
| Dream context | Shared with conversation | Isolated |
| Tool access | Via Hermes TUI | Direct (same safety rules) |

The headless implementation can reuse most of the IdleHandler logic, with the main differences being:
1. Different trigger mechanism (HTTP vs timer)
2. Context isolation for dreams
3. HTTP API for control and observability
