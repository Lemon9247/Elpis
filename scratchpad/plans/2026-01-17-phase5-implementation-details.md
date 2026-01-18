# Phase 5 Detailed Implementation Plan

## Current State Summary

Based on codebase exploration, the current architecture has:

| Component | Location | Current Role |
|-----------|----------|--------------|
| **PsycheCore** | `src/psyche/core/server.py` | Library instantiated by Hermes in-process |
| **ReactHandler** | `src/psyche/handlers/react_handler.py` | ReAct loop + tool execution in Hermes |
| **IdleHandler** | `src/psyche/handlers/idle_handler.py` | Workspace exploration when user idle |
| **ToolEngine** | `src/psyche/tools/tool_engine.py` | Tool registry + execution |
| **LocalPsycheClient** | `src/psyche/handlers/psyche_client.py` | Wraps PsycheCore for client interface |
| **Hermes CLI** | `src/hermes/cli.py` | Orchestrates full stack, runs TUI |
| **Elpis Server** | `src/elpis/server.py` | MCP server (stdio) for inference |
| **Mnemosyne Server** | `src/mnemosyne/server.py` | MCP server (stdio) for memory |

**Key Issue:** Hermes directly imports and instantiates PsycheCore as a library. Phase 5 separates them into server-client.

---

## Phase 5A: Psyche Server Infrastructure

### Goal
Create a standalone Psyche server that exposes PsycheCore via HTTP (OpenAI-compatible) and MCP.

### 5A.1: Create Server Package Structure

**New files to create:**
```
src/psyche/server/
├── __init__.py
├── http.py          # FastAPI OpenAI-compatible endpoints
├── mcp.py           # MCP server tools
├── daemon.py        # Server lifecycle management
└── connection.py    # Client connection tracking
```

### 5A.2: HTTP Server (`src/psyche/server/http.py`)

**Purpose:** OpenAI-compatible `/v1/chat/completions` endpoint

**Implementation:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn

# OpenAI-compatible request/response models
class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = "psyche"
    messages: List[ChatMessage]
    tools: Optional[List[Dict]] = None  # Tool definitions from client
    tool_choice: Optional[str] = "auto"
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict

class PsycheHTTPServer:
    def __init__(self, core: "PsycheCore", host: str = "127.0.0.1", port: int = 8741):
        self.core = core
        self.host = host
        self.port = port
        self.app = FastAPI(title="Psyche Server")
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            # 1. Update tool descriptions from request
            if request.tools:
                self.core.set_tool_descriptions_from_openai(request.tools)

            # 2. Process messages (add to context)
            for msg in request.messages:
                if msg.role == "user":
                    await self.core.add_user_message(msg.content)
                elif msg.role == "tool":
                    await self.core.add_tool_result(msg.tool_call_id, msg.content)

            # 3. Generate response (may include tool_calls)
            if request.stream:
                return StreamingResponse(...)  # SSE streaming
            else:
                result = await self.core.generate()
                return self._format_response(result)

        @self.app.get("/v1/models")
        async def list_models():
            return {"data": [{"id": "psyche", "object": "model"}]}

        @self.app.get("/health")
        async def health():
            return {"status": "ok", "connections": self.core.connection_count}
```

**Key behaviors:**
- Accepts tool definitions from client (client owns tools)
- Returns `tool_calls` in response, does NOT execute
- Tracks connections for dream scheduling
- Supports streaming via SSE

### 5A.3: MCP Server (`src/psyche/server/mcp.py`)

**Purpose:** MCP interface for Psyche-aware clients

**Tools to expose:**
```python
@server.tool()
async def chat(message: str) -> str:
    """Send a message and get response (may include tool_calls)"""
    await core.add_user_message(message)
    result = await core.generate()
    return result.to_json()

@server.tool()
async def add_tool_result(tool_call_id: str, result: str) -> str:
    """Add tool execution result"""
    await core.add_tool_result(tool_call_id, result)
    return "ok"

@server.tool()
async def recall_memory(query: str, n: int = 5) -> str:
    """Explicit memory retrieval (optional, auto-retrieval is default)"""
    memories = await core.retrieve_memories(query, n)
    return json.dumps(memories)

@server.tool()
async def store_memory(content: str, importance: float = 0.5) -> str:
    """Explicit memory storage (optional, auto-storage is default)"""
    await core.store_memory(content, importance)
    return "stored"

@server.tool()
async def get_emotion() -> str:
    """Get current emotional state"""
    state = await core.get_emotional_state()
    return state.to_json()

@server.tool()
async def get_context_summary() -> str:
    """Get working memory summary"""
    return core.get_context_summary()
```

### 5A.4: Server Daemon (`src/psyche/server/daemon.py`)

**Purpose:** Server lifecycle, connection tracking, dream scheduling

```python
class PsycheDaemon:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.core: Optional[PsycheCore] = None
        self.http_server: Optional[PsycheHTTPServer] = None
        self.mcp_server: Optional[PsycheMCPServer] = None
        self.connections: Set[str] = set()
        self.dream_handler: Optional[DreamHandler] = None
        self._dream_task: Optional[asyncio.Task] = None

    async def start(self):
        # 1. Create MCP clients for Elpis and Mnemosyne
        self.elpis_client = ElpisClient(...)
        self.mnemosyne_client = MnemosyneClient(...)

        # 2. Create PsycheCore
        self.core = PsycheCore(
            elpis_client=self.elpis_client,
            mnemosyne_client=self.mnemosyne_client,
            config=self.config.core,
        )
        await self.core.initialize()

        # 3. Create servers
        self.http_server = PsycheHTTPServer(self.core, port=self.config.http_port)
        self.mcp_server = PsycheMCPServer(self.core)

        # 4. Create dream handler
        self.dream_handler = DreamHandler(self.core)

        # 5. Start servers
        await asyncio.gather(
            self.http_server.start(),
            self.mcp_server.start(),
        )

    def on_client_connect(self, client_id: str):
        """Called when client connects"""
        self.connections.add(client_id)
        self._cancel_dreaming()

    def on_client_disconnect(self, client_id: str):
        """Called when client disconnects"""
        self.connections.discard(client_id)
        if not self.connections:
            self._schedule_dreaming()

    def _schedule_dreaming(self):
        """Start dreaming after delay if still no clients"""
        async def delayed_dream():
            await asyncio.sleep(self.config.dream_delay_seconds)
            if not self.connections:
                await self.dream_handler.start_dreaming()
        self._dream_task = asyncio.create_task(delayed_dream())

    def _cancel_dreaming(self):
        """Wake from dreaming when client connects"""
        if self._dream_task:
            self._dream_task.cancel()
            self._dream_task = None
        if self.dream_handler:
            self.dream_handler.stop_dreaming()
```

### 5A.5: Update CLI (`src/psyche/cli.py`)

**Current:** Stub or minimal
**New:** Launch server daemon

```python
import click
from psyche.server.daemon import PsycheDaemon, ServerConfig

@click.command()
@click.option("--http-port", default=8741, help="HTTP server port")
@click.option("--mcp-stdio", is_flag=True, help="Run MCP server on stdio")
@click.option("--config", type=click.Path(), help="Config file path")
def main(http_port: int, mcp_stdio: bool, config: Optional[str]):
    """Launch Psyche server daemon"""
    server_config = ServerConfig(http_port=http_port, ...)
    daemon = PsycheDaemon(server_config)

    try:
        asyncio.run(daemon.start())
    except KeyboardInterrupt:
        asyncio.run(daemon.shutdown())

if __name__ == "__main__":
    main()
```

### 5A.6: Modify PsycheCore for Server Mode

**Changes to `src/psyche/core/server.py`:**

1. **Tool handling:** Accept tool definitions from client, format for system prompt
2. **Tool call output:** Return tool_calls in response format, don't execute
3. **Connection awareness:** Track active connections for dream scheduling
4. **Streaming:** Support SSE streaming for HTTP clients

```python
# New method for OpenAI tool format
def set_tool_descriptions_from_openai(self, tools: List[Dict]):
    """Convert OpenAI tool format to internal format"""
    descriptions = []
    for tool in tools:
        func = tool.get("function", {})
        descriptions.append(ToolDescription(
            name=func.get("name"),
            description=func.get("description"),
            parameters=func.get("parameters"),
        ))
    self.set_tool_descriptions(descriptions)

# Modified generate to return tool_calls
async def generate(self) -> GenerationResult:
    """Generate response - may include tool_calls"""
    # ... existing logic ...

    # Parse tool calls from response
    tool_calls = self._parse_tool_calls(response_text)

    return GenerationResult(
        content=response_text,
        tool_calls=tool_calls,  # Client will execute these
        emotion=current_emotion,
    )
```

---

## Phase 5B: Hermes Client Refactor

### Goal
Transform Hermes from a PsycheCore wrapper into a true client that connects to Psyche server.

### 5B.1: Implement RemotePsycheClient

**File:** `src/psyche/handlers/psyche_client.py`

```python
class RemotePsycheClient(PsycheClient):
    """Client that connects to remote Psyche server via HTTP"""

    def __init__(self, base_url: str = "http://127.0.0.1:8741"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.tools: List[Dict] = []  # Tool definitions to send

    async def connect(self):
        self.session = aiohttp.ClientSession()
        # Verify connection
        async with self.session.get(f"{self.base_url}/health") as resp:
            if resp.status != 200:
                raise ConnectionError("Cannot connect to Psyche server")

    async def disconnect(self):
        if self.session:
            await self.session.close()

    def set_tools(self, tools: List[Dict]):
        """Set tool definitions to send with requests"""
        self.tools = tools

    async def chat(self, message: str) -> GenerationResult:
        """Send message and get response"""
        payload = {
            "model": "psyche",
            "messages": [{"role": "user", "content": message}],
            "tools": self.tools,
        }
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload
        ) as resp:
            data = await resp.json()
            return self._parse_response(data)

    async def send_tool_result(self, tool_call_id: str, result: str):
        """Send tool execution result back to server"""
        payload = {
            "model": "psyche",
            "messages": [{"role": "tool", "tool_call_id": tool_call_id, "content": result}],
        }
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload
        ) as resp:
            return await resp.json()

    async def chat_stream(self, message: str) -> AsyncIterator[str]:
        """Streaming chat via SSE"""
        payload = {
            "model": "psyche",
            "messages": [{"role": "user", "content": message}],
            "tools": self.tools,
            "stream": True,
        }
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload
        ) as resp:
            async for line in resp.content:
                if line.startswith(b"data: "):
                    yield json.loads(line[6:])
```

### 5B.2: Refactor ReactHandler for Client-Side Tool Execution

**File:** `src/psyche/handlers/react_handler.py`

The ReactHandler already handles tool execution. Main changes:
1. Use PsycheClient interface instead of direct Elpis calls
2. Send tool results back to server
3. Continue loop until no more tool_calls

```python
class ReactHandler:
    def __init__(
        self,
        client: PsycheClient,  # Changed from elpis_client
        tool_engine: ToolEngine,
        config: ReactConfig = ReactConfig(),
    ):
        self.client = client
        self.tool_engine = tool_engine
        self.config = config

    async def process_input(self, user_input: str) -> AsyncIterator[ReactEvent]:
        """Process user input with ReAct loop"""
        # 1. Send to server, get response
        result = await self.client.chat(user_input)
        yield ReactEvent(type="response", content=result.content)

        # 2. ReAct loop: execute tools, send results, get next response
        iteration = 0
        while result.tool_calls and iteration < self.config.max_tool_iterations:
            for tool_call in result.tool_calls:
                yield ReactEvent(type="tool_start", tool=tool_call)

                # Execute tool locally
                tool_result = await self.tool_engine.execute(
                    tool_call.name,
                    tool_call.arguments,
                )
                yield ReactEvent(type="tool_result", result=tool_result)

                # Send result back to server
                await self.client.send_tool_result(tool_call.id, tool_result)

            # Get next response
            result = await self.client.continue_generation()
            yield ReactEvent(type="response", content=result.content)
            iteration += 1
```

### 5B.3: Refactor Hermes CLI

**File:** `src/hermes/cli.py`

Transform from "create everything" to "connect to server":

```python
@click.command()
@click.option("--server", default="http://127.0.0.1:8741", help="Psyche server URL")
@click.option("--workspace", type=click.Path(), default=".", help="Workspace directory")
@click.option("--idle/--no-idle", default=True, help="Enable idle exploration")
def main(server: str, workspace: str, idle: bool):
    """Launch Hermes TUI client"""

    async def run():
        # 1. Connect to Psyche server
        client = RemotePsycheClient(base_url=server)
        await client.connect()

        # 2. Create local tool engine (client-side execution)
        tool_engine = ToolEngine(workspace_dir=workspace)
        register_all_tools(tool_engine)

        # 3. Tell server about our tools
        client.set_tools(tool_engine.get_openai_tool_definitions())

        # 4. Create handlers (run locally)
        react_handler = ReactHandler(client=client, tool_engine=tool_engine)

        idle_handler = None
        if idle:
            idle_handler = IdleHandler(
                client=client,
                tool_engine=tool_engine,
                workspace=workspace,
            )

        # 5. Run TUI
        app = Hermes(
            client=client,
            react_handler=react_handler,
            idle_handler=idle_handler,
        )
        try:
            await app.run_async()
        finally:
            await client.disconnect()

    asyncio.run(run())
```

### 5B.4: IdleHandler Stays in Hermes

**File:** `src/psyche/handlers/idle_handler.py`

IdleHandler is already workspace-specific. Minor changes:
1. Use PsycheClient interface
2. Clearly document this is CLIENT-side behavior

```python
class IdleHandler:
    """
    Client-side idle behavior when user is inactive but connected.

    This is NOT dreaming (server-side). This is workspace exploration
    specific to this client's environment.
    """

    def __init__(
        self,
        client: PsycheClient,
        tool_engine: ToolEngine,
        workspace: Path,
        config: IdleConfig = IdleConfig(),
    ):
        self.client = client
        self.tool_engine = tool_engine
        self.workspace = workspace
        self.config = config
        self._running = False

    async def start_idle(self):
        """Start idle exploration (user inactive)"""
        self._running = True
        while self._running:
            # Generate idle thought via server
            thought = await self.client.generate_idle_thought()

            # Execute safe tools locally
            if thought.tool_calls:
                for tool_call in thought.tool_calls:
                    if tool_call.name in SAFE_IDLE_TOOLS:
                        await self.tool_engine.execute(...)

            await asyncio.sleep(self.config.idle_tool_cooldown_seconds)

    def stop_idle(self):
        """Stop idle exploration (user active again)"""
        self._running = False
```

---

## Phase 5C: Dream Infrastructure

### Goal
Implement server-side dreaming when no clients are connected.

### 5C.1: Create DreamHandler

**New file:** `src/psyche/handlers/dream_handler.py`

```python
from dataclasses import dataclass
from typing import Optional, List
import asyncio

@dataclass
class DreamConfig:
    dream_interval_seconds: float = 300.0  # 5 minutes between dreams
    max_dream_duration_seconds: float = 60.0  # Max time per dream
    memory_query_count: int = 10  # Memories to load for context

class DreamHandler:
    """
    Server-side dreaming when no clients connected.

    Dreams are memory palace introspection - purely generative
    exploration of stored memories. No tools, no workspace access.

    This is distinct from IdleHandler (client-side workspace exploration).
    """

    def __init__(
        self,
        core: "PsycheCore",
        config: DreamConfig = DreamConfig(),
    ):
        self.core = core
        self.config = config
        self._dreaming = False
        self._dream_task: Optional[asyncio.Task] = None

    async def start_dreaming(self):
        """Begin dream cycle (no clients connected)"""
        self._dreaming = True
        while self._dreaming:
            await self._dream_once()
            await asyncio.sleep(self.config.dream_interval_seconds)

    def stop_dreaming(self):
        """Wake from dreaming (client connected)"""
        self._dreaming = False
        if self._dream_task:
            self._dream_task.cancel()

    async def _dream_once(self):
        """Single dream episode - memory palace exploration"""
        # 1. Load memories for dream context
        memories = await self.core.retrieve_random_memories(
            n=self.config.memory_query_count
        )

        # 2. Build dream prompt
        dream_prompt = self._build_dream_prompt(memories)

        # 3. Generate dream (no tools, pure generation)
        dream_content = await self.core.generate_dream(dream_prompt)

        # 4. Maybe store dream insights as new memories
        if self._is_insightful(dream_content):
            await self.core.store_memory(
                content=dream_content,
                memory_type="semantic",  # Dreams produce semantic memories
                importance=0.6,
            )

        # 5. Log dream for debugging/introspection
        self._log_dream(dream_content)

    def _build_dream_prompt(self, memories: List[Memory]) -> str:
        """Build prompt for dream generation"""
        memory_texts = [m.content for m in memories]
        return f"""You are in a dream state, reflecting on your memories.

Recent memories surfacing:
{chr(10).join(f'- {m}' for m in memory_texts)}

Let your mind wander through these memories. What patterns do you notice?
What connections emerge? What feelings arise?

Dream freely, without the need for action or response."""

    def _is_insightful(self, dream_content: str) -> bool:
        """Determine if dream produced storable insight"""
        # Heuristic: dreams with certain markers are worth storing
        insight_markers = ["realize", "connect", "pattern", "understand", "feel"]
        return any(marker in dream_content.lower() for marker in insight_markers)
```

### 5C.2: Add Dream Support to PsycheCore

**Changes to `src/psyche/core/server.py`:**

```python
class PsycheCore:
    # ... existing code ...

    async def generate_dream(self, dream_prompt: str) -> str:
        """Generate dream content (no tools, pure generation)"""
        # Use Elpis for generation without tool handling
        result = await self.elpis_client.generate(
            prompt=dream_prompt,
            max_tokens=500,
            temperature=0.9,  # Higher creativity for dreams
        )
        return result.text

    async def retrieve_random_memories(self, n: int = 10) -> List[Memory]:
        """Retrieve random memories for dream seeding"""
        # Use Mnemosyne to get diverse memories
        return await self.mnemosyne_client.get_random_memories(n=n)
```

### 5C.3: Connection Tracking in Server

Already covered in 5A.4 (daemon.py). The daemon tracks:
- `on_client_connect()` - cancel dreaming
- `on_client_disconnect()` - schedule dreaming if no other clients

---

## Implementation Order

### Session 1: Server Foundation
1. Create `src/psyche/server/__init__.py`
2. Create `src/psyche/server/http.py` with basic `/v1/chat/completions`
3. Create basic `src/psyche/server/daemon.py`
4. Update `src/psyche/cli.py` to launch server
5. Test: server starts, responds to health check

### Session 2: Server Completeness
1. Complete HTTP server with streaming support
2. Create `src/psyche/server/mcp.py` with tools
3. Add connection tracking to daemon
4. Modify PsycheCore for server mode (tool_calls output)
5. Test: full chat flow via HTTP

### Session 3: Client Refactor
1. Implement `RemotePsycheClient` in `psyche_client.py`
2. Refactor `ReactHandler` to use client interface
3. Test: ReactHandler works with remote client

### Session 4: Hermes Integration
1. Refactor `hermes/cli.py` to use remote client
2. Update Hermes TUI to work with new flow
3. Test: Hermes connects to server, full conversation works

### Session 5: Dream Infrastructure
1. Create `src/psyche/handlers/dream_handler.py`
2. Wire dream scheduling in daemon
3. Add dream methods to PsycheCore
4. Test: server dreams when no clients

### Session 6: Polish & Testing
1. Add configuration options
2. Error handling and reconnection
3. Integration tests
4. Documentation updates

---

## Files Changed Summary

### New Files
- `src/psyche/server/__init__.py`
- `src/psyche/server/http.py`
- `src/psyche/server/mcp.py`
- `src/psyche/server/daemon.py`
- `src/psyche/server/connection.py`
- `src/psyche/handlers/dream_handler.py`

### Modified Files
- `src/psyche/cli.py` - Launch server daemon
- `src/psyche/core/server.py` - Server mode, tool_calls output
- `src/psyche/handlers/psyche_client.py` - Add RemotePsycheClient
- `src/psyche/handlers/react_handler.py` - Use PsycheClient interface
- `src/psyche/handlers/idle_handler.py` - Document as client-side
- `src/hermes/cli.py` - Connect to server instead of creating core

### Unchanged (but important context)
- `src/elpis/` - Stays as-is (MCP server for inference)
- `src/mnemosyne/` - Stays as-is (MCP server for memory)
- `src/psyche/tools/` - Stays as-is (used by clients)
- `src/hermes/app.py` - Minimal changes (receives different deps)

---

## Verification Checklist

After implementation:

- [ ] `psyche-server` launches and listens on HTTP port
- [ ] `curl localhost:8741/health` returns ok
- [ ] `curl localhost:8741/v1/chat/completions` works
- [ ] `hermes --server http://localhost:8741` connects
- [ ] Tool calls returned by server, executed by client
- [ ] Hermes IdleHandler explores workspace (client-side)
- [ ] When Hermes disconnects, server starts dreaming
- [ ] When Hermes reconnects, server wakes
- [ ] External tools (httpie, curl) can chat via HTTP
