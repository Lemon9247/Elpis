# Phase 2 Refactoring Plan
**Date**: 2026-01-12
**Author**: Claude Sonnet 4.5
**Status**: Proposal - Awaiting Review

## Executive Summary

This document outlines the refactoring work required to transform Elpis from a monolithic interactive REPL agent into a three-tier architecture supporting continuous inference, emotional regulation, and extensible memory systems.

### Architecture Transition

**Current (Phase 1)**: Monolithic REPL Application
```
CLI → REPL → Orchestrator → [LLM + Tools]
```

**Target (Phase 2)**: Three-Tier Distributed System
```
Inference Server (Emotional LLM API)
        ↕
Memory Server (Continuous Inference + Storage)
        ↕
User Input Client (Async REPL)
```

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Architecture Overview](#architecture-overview)
3. [Refactoring Strategy](#refactoring-strategy)
4. [Component Specifications](#component-specifications)
5. [Implementation Phases](#implementation-phases)
6. [Migration Path](#migration-path)
7. [Testing Strategy](#testing-strategy)
8. [Risk Assessment](#risk-assessment)

---

## Current State Analysis

### Phase 1 Components

```
src/elpis/
├── cli.py                    # Entry point - wires everything together
├── config/settings.py        # Pydantic configuration
├── llm/
│   ├── inference.py          # LLaMA inference wrapper
│   └── prompts.py            # System prompt builder
├── tools/
│   ├── tool_definitions.py   # Tool schemas
│   ├── tool_engine.py        # Tool orchestrator
│   └── implementations/      # 5 tools (file, bash, search, directory)
├── agent/
│   ├── orchestrator.py       # ReAct loop (max 10 iterations)
│   └── repl.py              # Interactive REPL with Rich formatting
└── utils/
    ├── hardware.py          # GPU detection
    ├── logging.py           # Loguru config
    └── exceptions.py        # Custom exceptions
```

### Current Flow

1. User enters input in REPL
2. REPL passes to Orchestrator
3. Orchestrator runs ReAct loop:
   - Calls LLM for reasoning
   - Executes tools concurrently
   - Returns final response
4. REPL displays response
5. **Loop waits for next user input** ← KEY LIMITATION

### What Works Well

- ✅ Async architecture throughout
- ✅ Tool safety and path sandboxing
- ✅ Concurrent tool execution
- ✅ Clean separation of concerns (mostly)
- ✅ Comprehensive error handling
- ✅ Good test coverage (>80%)

### What Needs to Change

- ❌ **Synchronous request/response model** - No continuous inference
- ❌ **Tight coupling** - LLM, tools, orchestrator all in one process
- ❌ **No memory persistence** - History kept in-memory only
- ❌ **No emotional system** - LLM is stateless
- ❌ **Single client model** - Can't handle multiple input sources
- ❌ **Context window management** - No compaction or consolidation

---

## Architecture Overview

### Component Responsibilities

#### 1. Inference Server (New - Standalone)

**Purpose**: Stateful LLM API with emotional regulation

**Responsibilities**:
- Provide OpenAI-compatible API endpoints
- Maintain emotional state (hormone levels)
- Adjust emotions based on inference content
- Return to homeostasis over time
- Track emotional trajectory

**Interfaces**:
- `POST /v1/chat/completions` - Standard OpenAI format
- `POST /v1/function_call` - Function calling
- `GET /v1/emotions` - Current emotional state
- `POST /v1/emotions/reset` - Reset to baseline

**State**:
- Current hormone levels (valence, arousal, etc.)
- Homeostasis targets
- Decay rates
- Inference history for emotional context

**Independence**:
- No knowledge of tools or harness
- Portable across different systems
- Could be used by multiple harnesses
- Future: Could support multiple models

#### 2. Memory Server (Refactored from current system)

**Purpose**: Continuous inference harness with memory management

**Responsibilities**:
- Run continuous inference loop (not request/response)
- Manage context window (sliding/compaction)
- Store memories to long-term storage
- Query and retrieve relevant memories
- Consolidate memories during "naps"
- Determine emotional importance of memories

**Interfaces**:
- `inject_message(content)` - Interrupt loop with new input
- `get_stream()` - Stream of consciousness output
- `trigger_consolidation()` - Force memory consolidation
- `query_memories(query)` - Retrieve relevant memories

**State**:
- Current context window
- Short-term memory buffer
- Connection to long-term storage (ChromaDB)
- Stream of consciousness output buffer

**Components**:
- Continuous inference loop
- Context window manager
- Memory consolidator
- Long-term storage interface
- Tool execution engine (kept here)

#### 3. User Input Client (New - Async interface)

**Purpose**: Asynchronous user interaction interface

**Responsibilities**:
- Display stream of consciousness
- Accept user input asynchronously
- Interrupt memory server with messages
- Provide REPL-like experience

**Interfaces**:
- Interactive REPL (current)
- Web interface (future)
- API endpoint (future)
- Multiple simultaneous connections (future)

**State**:
- Display buffer
- Command history
- Connection to memory server

---

## Refactoring Strategy

### Phase-Based Approach

We'll refactor in 3 phases to maintain working system at each step:

1. **Phase 2A: Extract Inference Server** (Week 1-2)
   - Create standalone inference server
   - Minimal emotional system (placeholder)
   - API-ify current LLM wrapper
   - Keep rest of system working

2. **Phase 2B: Transform to Continuous Inference** (Week 3-4)
   - Convert orchestrator to continuous loop
   - Implement async message injection
   - Create streaming output
   - Move REPL to client model

3. **Phase 2C: Add Memory System** (Week 5-6)
   - Context window management
   - Memory compaction
   - Long-term storage (ChromaDB)
   - Emotional memory tagging

### Compatibility Strategy

- Maintain backward compatibility during transition
- Use feature flags to enable new components
- Keep old system working until new is proven
- Gradual migration path for users

---

## Component Specifications

### 1. Inference Server Specification

#### Technology Stack
- **Framework**: FastAPI (async, OpenAPI docs, fast)
- **LLM Backend**: llama-cpp-python (existing)
- **State Management**: In-memory (Phase 2A), Redis (future)
- **Emotional System**: Custom (Phase 2A: simple, Phase 3: complex)

#### API Endpoints

##### POST /v1/chat/completions
OpenAI-compatible chat completions endpoint

**Request**:
```json
{
  "model": "elpis-llama-3.1-8b",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false,
  "emotion_context": true  // Elpis extension
}
```

**Response**:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "elpis-llama-3.1-8b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help?"
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
  "emotion_state": {  // Elpis extension
    "valence": 0.2,
    "arousal": 0.1,
    "dominance": 0.0
  }
}
```

##### POST /v1/function_call
Function calling endpoint

**Request**:
```json
{
  "messages": [...],
  "tools": [...],
  "tool_choice": "auto"
}
```

**Response**:
```json
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "function": {
        "name": "read_file",
        "arguments": "{\"file_path\": \"test.py\"}"
      }
    }
  ],
  "emotion_state": {...}
}
```

##### GET /v1/emotions
Get current emotional state

**Response**:
```json
{
  "current": {
    "valence": 0.15,      // -1 (negative) to +1 (positive)
    "arousal": 0.08,      // 0 (calm) to 1 (excited)
    "dominance": 0.0      // -1 (submissive) to +1 (dominant)
  },
  "homeostasis": {
    "valence": 0.0,
    "arousal": 0.0,
    "dominance": 0.0
  },
  "decay_rate": 0.01,
  "time_since_last_inference": 1.5
}
```

##### POST /v1/emotions/adjust
Manually adjust emotional state (for testing)

**Request**:
```json
{
  "delta": {
    "valence": 0.1,
    "arousal": 0.05
  }
}
```

#### Emotional Regulation System (Phase 2A - Simple)

**Hormone Model**:
- Valence: Positive/negative emotion (-1 to +1)
- Arousal: Energy level (0 to 1)
- Dominance: Control/submission (-1 to +1)

**Adjustment Mechanism**:
1. After each inference, analyze output sentiment
2. Adjust valence based on detected sentiment
3. Adjust arousal based on urgency/intensity
4. Apply decay toward homeostasis (0, 0, 0)

**Decay Formula**:
```python
def decay_emotions(current, homeostasis, decay_rate, time_delta):
    """Exponential decay toward homeostasis"""
    return current + (homeostasis - current) * (1 - exp(-decay_rate * time_delta))
```

**Phase 2A Implementation** (Simple):
```python
class EmotionalState:
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    def adjust_from_text(self, text: str):
        """Simple sentiment-based adjustment"""
        # Use simple keyword matching (Phase 2A)
        positive_words = ["good", "great", "success", "works"]
        negative_words = ["error", "failed", "broken", "wrong"]

        # Count occurrences
        pos_count = sum(1 for word in positive_words if word in text.lower())
        neg_count = sum(1 for word in negative_words if word in text.lower())

        # Adjust valence
        delta_valence = (pos_count - neg_count) * 0.05
        self.valence = max(-1, min(1, self.valence + delta_valence))

        # Increase arousal slightly
        self.arousal = min(1, self.arousal + 0.02)

    def decay_toward_homeostasis(self, dt: float, decay_rate: float = 0.01):
        """Return to baseline over time"""
        factor = 1 - math.exp(-decay_rate * dt)
        self.valence *= (1 - factor)
        self.arousal *= (1 - factor)
        self.dominance *= (1 - factor)
```

#### File Structure

```
elpis-inference-server/
├── pyproject.toml
├── README.md
├── src/
│   └── elpis_server/
│       ├── __init__.py
│       ├── main.py              # FastAPI app
│       ├── api/
│       │   ├── __init__.py
│       │   ├── chat.py          # /v1/chat/completions
│       │   ├── functions.py     # /v1/function_call
│       │   └── emotions.py      # /v1/emotions/*
│       ├── llm/
│       │   ├── __init__.py
│       │   └── inference.py     # Moved from main repo
│       ├── emotions/
│       │   ├── __init__.py
│       │   ├── state.py         # EmotionalState class
│       │   └── analyzer.py      # Sentiment analysis
│       ├── models/
│       │   ├── __init__.py
│       │   ├── requests.py      # Pydantic request models
│       │   └── responses.py     # Pydantic response models
│       └── config/
│           ├── __init__.py
│           └── settings.py      # Server settings
└── tests/
    ├── test_api.py
    ├── test_emotions.py
    └── test_llm.py
```

---

### 2. Memory Server Specification

#### Purpose
Transform current monolithic system into continuous inference harness

#### Key Changes

**From**: Request/response orchestrator
**To**: Continuous inference loop

**Current Flow**:
```python
async def process(user_input: str) -> str:
    # Add user message
    # Run ReAct loop (max 10 iterations)
    # Return final response
    # WAIT for next input
```

**New Flow**:
```python
async def continuous_loop():
    while True:
        # Check for injected messages (non-blocking)
        if message := await check_for_messages():
            context.add_message(message)

        # Continue inference
        response = await llm.generate(context.get_window())

        # Stream output
        await output_stream.write(response)

        # Add to context
        context.add_assistant_message(response)

        # Manage context window
        if context.is_full():
            await context.compact()

        # Check for consolidation trigger
        if should_consolidate():
            await consolidate_memories()

        # Brief pause (avoid tight loop)
        await asyncio.sleep(0.1)
```

#### Context Window Management

**Strategies** (to research in Phase 2A):
1. **Sliding Window**: Drop oldest messages (simple, lossy)
2. **Summarization**: LLM summarizes old context (expensive, good)
3. **Importance-Based**: Keep important messages, drop others (complex, best)
4. **Hybrid**: Combine strategies

**Phase 2A Implementation** (Simple - Sliding Window):
```python
class ContextWindow:
    def __init__(self, max_tokens: int = 4096):
        self.messages: List[Message] = []
        self.max_tokens = max_tokens

    def add_message(self, message: Message):
        self.messages.append(message)
        self._trim_if_needed()

    def _trim_if_needed(self):
        """Remove oldest messages if over token limit"""
        while self._count_tokens() > self.max_tokens:
            if len(self.messages) <= 2:  # Keep at least system + last message
                break
            self.messages.pop(1)  # Remove oldest (keep system message)

    def get_window(self) -> List[Dict]:
        """Get messages for LLM"""
        return [msg.to_dict() for msg in self.messages]
```

#### Message Injection System

**Purpose**: Allow async interruption of continuous loop

```python
class MessageQueue:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def inject(self, content: str, role: str = "user"):
        """Inject message into continuous loop"""
        await self.queue.put(Message(role=role, content=content))

    async def get_next(self, timeout: float = 0.1) -> Optional[Message]:
        """Non-blocking get with timeout"""
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
```

#### Output Streaming

**Purpose**: Provide stream of consciousness to clients

```python
class OutputStreamclass OutputStreamManager:
    def __init__(self):
        self.subscribers: List[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to output stream"""
        queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue

    async def write(self, content: str):
        """Write to all subscribers"""
        for queue in self.subscribers:
            await queue.put(content)

    def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from stream"""
        self.subscribers.remove(queue)
```

#### Memory Consolidation (Phase 2C)

**Trigger Conditions**:
- Context window is full
- Emotional intensity exceeds threshold
- Time-based (every N minutes)
- Manual trigger

**Process**:
1. Pause continuous loop
2. Summarize current context window
3. Extract key memories
4. Tag with emotional valence
5. Store to ChromaDB
6. Clear old context
7. Resume with fresh window + summary

#### File Structure Changes

```
src/elpis/
├── cli.py                      # MODIFIED: Launch memory server
├── config/settings.py          # MODIFIED: Add server settings
├── memory_server/              # NEW: Continuous inference
│   ├── __init__.py
│   ├── server.py              # Main server loop
│   ├── context_window.py      # Context management
│   ├── message_queue.py       # Async message injection
│   ├── output_stream.py       # Stream of consciousness
│   └── consolidation.py       # Memory consolidation (Phase 2C)
├── llm/
│   ├── inference_client.py    # NEW: Client for inference server
│   └── prompts.py             # Keep existing
├── tools/                      # KEEP: Still used by memory server
│   └── ...
├── agent/
│   ├── orchestrator.py        # REMOVED/REFACTORED into memory_server/
│   └── repl.py               # MOVED to user_client/
└── storage/                    # NEW: Phase 2C
    ├── __init__.py
    ├── chromadb_client.py     # Long-term storage
    └── models.py              # Memory data models
```

---

### 3. User Input Client Specification

#### Purpose
Async client that displays stream and accepts input

#### Responsibilities
1. Connect to memory server (WebSocket? HTTP SSE?)
2. Display stream of consciousness
3. Accept user input
4. Inject messages into memory server
5. Handle special commands

#### Technology Options

**Option A: WebSocket** (Recommended)
- Real-time bidirectional communication
- Efficient streaming
- Good library support (websockets, aiohttp)

**Option B: HTTP + Server-Sent Events (SSE)**
- HTTP GET for stream (SSE)
- HTTP POST for input injection
- Simpler, more standard

**Option C: gRPC**
- High performance
- Bidirectional streaming
- More complex setup

**Recommendation**: Start with WebSocket for simplicity and real-time feel

#### Client Flow

```python
class ElpisClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.ws = None
        self.output_display = OutputDisplay()

    async def connect(self):
        """Connect to memory server"""
        self.ws = await websockets.connect(f"{self.server_url}/stream")

    async def run(self):
        """Main client loop"""
        # Start two concurrent tasks
        await asyncio.gather(
            self._receive_stream(),
            self._handle_input()
        )

    async def _receive_stream(self):
        """Receive and display stream of consciousness"""
        async for message in self.ws:
            await self.output_display.write(message)

    async def _handle_input(self):
        """Accept user input and send to server"""
        while True:
            user_input = await self._get_user_input()

            if user_input.startswith('/'):
                await self._handle_command(user_input)
            else:
                await self.ws.send(json.dumps({
                    "type": "user_message",
                    "content": user_input
                }))

    async def _get_user_input(self) -> str:
        """Async input (prompt_toolkit)"""
        return await self.session.prompt_async("elpis> ")
```

#### File Structure

```
src/elpis/
└── user_client/                # NEW
    ├── __init__.py
    ├── client.py              # Main client
    ├── display.py             # Output formatting (from repl.py)
    └── commands.py            # Special commands
```

---

## Implementation Phases

### Phase 2A: Extract Inference Server (Week 1-2)

#### Goals
- ✅ Standalone inference server running
- ✅ Basic emotional system (placeholder)
- ✅ OpenAI-compatible API
- ✅ Current system still works

#### Tasks

**Week 1: Server Foundation**
1. Create new `elpis-inference-server/` directory
2. Set up FastAPI project structure
3. Move `llm/inference.py` to server
4. Implement `/v1/chat/completions` endpoint
5. Implement `/v1/function_call` endpoint
6. Add basic emotional state tracking
7. Write API tests

**Week 2: Emotional System & Integration**
8. Implement `EmotionalState` class
9. Add sentiment analysis (simple keyword-based)
10. Implement `/v1/emotions` endpoints
11. Add emotional decay logic
12. Test emotional adjustments
13. Document API (OpenAPI/Swagger)
14. Create inference client in main repo
15. Integration test: main repo → server

#### Success Criteria
- [ ] Server runs independently (`uvicorn elpis_server.main:app`)
- [ ] All API endpoints functional
- [ ] Emotional state persists across requests
- [ ] Current Elpis still works with server
- [ ] Tests pass (unit + integration)
- [ ] API documentation complete

#### Testing
```bash
# Terminal 1: Start server
cd elpis-inference-server
uvicorn elpis_server.main:app --reload

# Terminal 2: Test with curl
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'

# Terminal 3: Run main Elpis (using server)
cd elpis
elpis --inference-server http://localhost:8000
```

---

### Phase 2B: Continuous Inference (Week 3-4)

#### Goals
- ✅ Memory server runs continuous loop
- ✅ Async message injection works
- ✅ Output streaming functional
- ✅ User client can interact

#### Tasks

**Week 3: Memory Server Core**
1. Create `memory_server/` package
2. Implement `ContextWindow` class
3. Implement `MessageQueue` class
4. Implement `OutputStreamManager` class
5. Create continuous inference loop
6. Add WebSocket endpoint for clients
7. Test loop and message injection

**Week 4: User Client**
8. Create `user_client/` package
9. Move REPL display code to client
10. Implement WebSocket client
11. Implement async input handling
12. Test bidirectional communication
13. Add special commands (/stop, /pause, /resume)
14. Integration testing

#### Success Criteria
- [ ] Memory server runs continuously
- [ ] Stream of consciousness visible
- [ ] User can inject messages any time
- [ ] Multiple clients can connect (bonus)
- [ ] Clean shutdown handling
- [ ] Tests pass

#### Testing
```bash
# Terminal 1: Inference server
uvicorn elpis_server.main:app

# Terminal 2: Memory server
python -m elpis.memory_server.server

# Terminal 3: User client
python -m elpis.user_client.client
```

---

### Phase 2C: Memory System (Week 5-6)

#### Goals
- ✅ Context window management working
- ✅ Memory consolidation implemented
- ✅ ChromaDB integration complete
- ✅ Emotional memory tagging functional

#### Tasks

**Week 5: Storage & Consolidation**
1. Set up ChromaDB
2. Design memory data models
3. Implement `MemoryStore` class
4. Create consolidation process
5. Implement compaction strategies
6. Test memory storage/retrieval

**Week 6: Integration & Polish**
7. Integrate consolidation into memory server
8. Add consolidation triggers
9. Implement memory-augmented prompts
10. Test full system
11. Performance optimization
12. Documentation

#### Success Criteria
- [ ] Memories stored to ChromaDB
- [ ] Context compaction works
- [ ] Memory retrieval functional
- [ ] Emotional tagging applied
- [ ] System handles long conversations
- [ ] Tests pass

---

## Migration Path

### For Users

**Option 1: Run All Components (Full Experience)**
```bash
# 1. Start inference server
elpis-server start

# 2. Start memory server
elpis-memory start

# 3. Connect with client
elpis-client connect
```

**Option 2: Run Legacy Mode (Compatibility)**
```bash
# Single command runs all components
elpis --legacy-mode
```

### Configuration

```toml
# config.toml

[inference_server]
host = "localhost"
port = 8000
model_path = "models/llama-3.1-8b-instruct-q5_k_m.gguf"

[inference_server.emotions]
enabled = true
decay_rate = 0.01
homeostasis = { valence = 0.0, arousal = 0.0, dominance = 0.0 }

[memory_server]
host = "localhost"
port = 8001
inference_server_url = "http://localhost:8000"
context_window_tokens = 4096
consolidation_trigger = "auto"  # or "manual", "time", "threshold"

[memory_server.storage]
type = "chromadb"
path = "~/.elpis/memories"

[user_client]
memory_server_url = "ws://localhost:8001/stream"
display_stream = true
```

---

## Testing Strategy

### Unit Tests
- Each component tested in isolation
- Mock external dependencies
- Focus on business logic

### Integration Tests
- Test component interactions
- API contract testing
- End-to-end flows

### Performance Tests
- Continuous loop performance
- Memory usage over time
- Context compaction speed
- Stream throughput

### Test Coverage Goals
- Maintain >80% coverage
- Critical paths: 100% coverage
- Emotional system: Comprehensive behavior tests

---

## Risk Assessment

### High Risk

#### 1. Continuous Loop Stability
**Risk**: Loop crashes or hangs
**Mitigation**:
- Comprehensive error handling
- Health checks
- Auto-restart on failure
- Rate limiting

#### 2. Context Window Management
**Risk**: Poor compaction loses important context
**Mitigation**:
- Start with simple sliding window
- Extensive testing with long conversations
- User feedback mechanism
- Multiple compaction strategies

#### 3. Performance Degradation
**Risk**: Continuous inference too slow
**Mitigation**:
- Profile and optimize early
- Async everywhere
- Consider batch processing
- Hardware acceleration (GPU)

### Medium Risk

#### 4. API Compatibility
**Risk**: Breaking changes to inference API
**Mitigation**:
- Version API endpoints (/v1/, /v2/)
- Maintain backward compatibility
- Clear migration guides

#### 5. Emotional System Complexity
**Risk**: Simple system insufficient
**Mitigation**:
- Phase 2A: Simple keyword-based
- Phase 3: Advanced ML-based
- Iterative improvement
- User testing

### Low Risk

#### 6. Multiple Client Support
**Risk**: Race conditions with multiple clients
**Mitigation**:
- Use asyncio primitives (Lock, Queue)
- Test with multiple clients
- Clear concurrency model

---

## Open Questions

### To Research Before Implementation

1. **Context Compaction**
   - How does Anthropic handle context windows in Claude Code?
   - What compaction strategies exist?
   - Summarization vs importance-based?

2. **LLM API Standards**
   - OpenAI compatibility sufficient?
   - Any emerging standards?
   - Tool/function calling formats?

3. **Project Split Decision**
   - Should inference server be separate repo?
   - Monorepo vs multi-repo?
   - Versioning strategy?

4. **Tool System Location**
   - Tools stay with memory server (current plan)?
   - Or should inference server know about tools?
   - Impact on portability?

5. **Streaming Implementation**
   - Should LLM responses stream token-by-token?
   - WebSocket vs SSE vs HTTP/2?
   - Buffering strategy?

---

## Success Metrics

### Technical Metrics
- ✅ All three components running independently
- ✅ >80% test coverage maintained
- ✅ API response time <100ms (p95)
- ✅ Continuous loop stable for >1 hour
- ✅ Memory consolidation completes <5 seconds
- ✅ Context window management effective (no critical information loss)

### Functional Metrics
- ✅ Emotional state persists and evolves correctly
- ✅ User can interact at any time during continuous inference
- ✅ Memories stored and retrieved accurately
- ✅ System handles long conversations (>1000 messages)
- ✅ Multiple clients can connect simultaneously

### Quality Metrics
- ✅ No regression in existing functionality
- ✅ Clear error messages and logging
- ✅ Complete API documentation
- ✅ Migration guide for users
- ✅ Architecture documentation updated

---

## Next Steps

### Immediate Actions (Before Starting Phase 2A)

1. **Research Tasks** (from phase-2-notes.md TODO):
   - [ ] Research context compaction strategies
   - [ ] Review OpenAI API specification
   - [ ] Investigate ChromaDB best practices
   - [ ] Look into LLM streaming implementations

2. **Planning Tasks**:
   - [ ] Review this refactoring plan
   - [ ] Get user/maintainer approval
   - [ ] Decide on monorepo vs separate repos
   - [ ] Create detailed task breakdown for Phase 2A

3. **Setup Tasks**:
   - [ ] Create feature branch: `feat/phase-2-architecture`
   - [ ] Set up new directory structures
   - [ ] Update project dependencies
   - [ ] Configure development environment

### Suggested Implementation Order

**Option A: Parallel (Faster, Higher Risk)**
- Agent 1: Inference Server (Phase 2A)
- Agent 2: Memory Server Core (Phase 2B)
- Agent 3: User Client (Phase 2B)

**Option B: Sequential (Slower, Lower Risk)**
- Week 1-2: Inference Server (all hands)
- Week 3-4: Memory Server + Client (split teams)
- Week 5-6: Memory System (all hands)

**Recommendation**: Option B (Sequential) for Phase 2A-B, then parallel for 2C

---

## Appendices

### A. Glossary

- **Continuous Inference**: LLM runs continuously, generating stream of thought
- **Context Window**: Active messages sent to LLM for inference
- **Memory Consolidation**: Process of compacting and storing memories
- **Emotional Regulation**: System that adjusts hormone levels based on content
- **Homeostasis**: Target emotional state to return to
- **Stream of Consciousness**: Continuous LLM output visible to user
- **Message Injection**: Asynchronously adding messages to continuous loop

### B. References

- OpenAI API Documentation: https://platform.openai.com/docs/api-reference
- FastAPI Documentation: https://fastapi.tiangolo.com/
- ChromaDB Documentation: https://docs.trychroma.com/
- LLaMA.cpp Python Bindings: https://llama-cpp-python.readthedocs.io/

### C. Related Documents

- `scratchpad/phase-1/PHASE1-COMPLETION.md` - Current system implementation
- `scratchpad/phase-2-notes.md` - Original architecture proposal
- `scratchpad/phase-1/week3-architecture.md` - Current orchestrator design

---

**End of Refactoring Plan**
