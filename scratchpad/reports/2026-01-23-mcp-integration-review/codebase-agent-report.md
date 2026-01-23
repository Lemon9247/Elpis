# Elpis MCP Integration - Codebase Review Report

**Agent**: Codebase Agent
**Date**: 2026-01-23
**Task**: Review Elpis's current MCP integration architecture

---

## Executive Summary

Elpis implements a well-architected MCP integration system with clear separation of concerns. The system comprises two MCP servers (Elpis for inference, Mnemosyne for memory) that Psyche clients connect to via stdio-based MCP clients. The architecture demonstrates sophisticated patterns for lifecycle management, tool discovery, error handling, and asynchronous processing.

---

## 1. How Psyche Connects to MCP Servers

**Location:** `src/psyche/mcp/client.py`

### Architecture

Psyche uses dedicated MCP client classes (`ElpisClient` and `MnemosyneClient`) that wrap the MCP library's `ClientSession`. Connections are established via stdio with `StdioServerParameters` and `stdio_client`.

Both clients follow the **context manager pattern** for lifecycle safety:

```python
async with client.connect() as connected_client:
    result = await connected_client.generate(messages)
```

### Key Features

- **Session locking**: All MCP session operations are protected by `asyncio.Lock` to prevent concurrent access issues
  - Lock initialized early (`self._session_lock = asyncio.Lock()`) for test compatibility
  - Every tool call wraps the session access: `async with self._session_lock:`
- **Connection state tracking**: `is_connected` property validates both session existence and connection status
- **Subprocess environment control**: `ELPIS_QUIET=1` and `MNEMOSYNE_QUIET=1` environment variables suppress server stderr logging

### Tool Calling Pattern

Both clients implement `_call_tool()` as the base method:

```python
async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    async with self._session_lock:
        result = await self._session.call_tool(name, arguments)
        if result.content and len(result.content) > 0:
            return json.loads(result.content[0].text)
        return {}
```

---

## 2. Configuration Mechanisms

### Configuration Files

| File | Purpose |
|------|---------|
| `configs/psyche.toml` | Main client configuration |
| `configs/elpis.toml` | Inference server configuration |
| `configs/mnemosyne.toml` | Memory server configuration |

### Settings Infrastructure

**Psyche Settings** (`src/psyche/config/settings.py`):
- Uses Pydantic `BaseSettings` with TOML file support
- Configuration hierarchy: **Init args > Env vars > TOML file > Defaults**
- Environment variable prefix: `PSYCHE_*` with nested delimiter `__`

**Key Server Configuration:**

```python
@dataclass
class ServerSettings(BaseSettings):
    http_host: str = "127.0.0.1"
    http_port: int = 8741
    elpis_command: str = "elpis-server"
    mnemosyne_command: Optional[str] = "mnemosyne-server"
    dream_enabled: bool = True
    dream_delay_seconds: float = 60.0
    model_name: str = "psyche"
```

**Server Daemon Configuration** (`src/psyche/server/daemon.py`):

```python
@dataclass
class ServerConfig:
    elpis_command: str = "elpis-server"
    mnemosyne_command: Optional[str] = "mnemosyne-server"
    consolidation_enabled: bool = True
    consolidation_interval: float = 300.0
```

CLI allows override of all major settings:

```bash
psyche-server --host 127.0.0.1 --port 8741 --elpis-command "elpis-server" \
              --mnemosyne-command "mnemosyne-server" --no-dream
```

---

## 3. Server Lifecycle Management

### Initialization Flow

The `PsycheDaemon` class manages the full lifecycle (`src/psyche/server/daemon.py`):

1. **Startup (`daemon.start()`)**:
   - Create MCP client objects (no connection yet)
   - Connect to Elpis via `_connect_elpis()` context manager
   - Connect to Mnemosyne via `_connect_mnemosyne()` context manager (gracefully handles failure)
   - Query Elpis capabilities to configure context window
   - Initialize `PsycheCore` with connected clients
   - Initialize HTTP server
   - Initialize dream handler (if enabled)
   - Start consolidation loop (if Mnemosyne available)
   - Run HTTP server via uvicorn

2. **Connection Management** - Nested context managers keep servers alive:

```python
async with self._connect_elpis() as elpis:
    async with self._connect_mnemosyne() as mnemosyne:
        await self._init_core_with_clients(elpis, mnemosyne)
        # ... HTTP server runs while contexts are open
```

3. **Graceful Shutdown (`daemon.shutdown()`)**:
   - Cancel dreaming
   - Cancel consolidation loop
   - Shutdown core (triggers memory consolidation)
   - Set shutdown event

### Mnemosyne Resilience

- If Mnemosyne connection fails, logs warning and continues without persistent memory
- Core functionality works with or without Mnemosyne
- Fallback storage available: `~/.psyche/fallback_memories/`

---

## 4. Tool Registration and Discovery

### Architecture Overview

Tools are registered at MCP server initialization via `@server.list_tools()` decorator. Psyche discovers tools dynamically and manages two types:
1. **Memory tools** - Execute server-side (recall_memory, store_memory)
2. **Client tools** - Returned to client for execution

### Elpis Tool Registration

**Location:** `src/elpis/server.py`

Tools exposed by Elpis (14 total):
- `generate` - Text generation with emotional modulation
- `function_call` - Tool call generation
- `generate_stream_start/read/cancel` - Streaming generation protocol
- `update_emotion` - Trigger emotional events
- `reset_emotion`, `get_emotion` - Emotional state queries
- `get_capabilities` - Server metadata

### Mnemosyne Tool Registration

**Location:** `src/mnemosyne/server.py`

Tools exposed by Mnemosyne:
- `store_memory` - Store episodic/semantic/procedural/emotional memories
- `search_memories` - Semantic search via embeddings
- `consolidate_memories` - Clustering and long-term promotion
- `should_consolidate` - Check if consolidation needed
- `get_memory_stats` - Database statistics
- `get_memory_context` - Formatted memories for injection
- `delete_memory` - Remove memories by ID
- `get_recent_memories` - Time-based retrieval

### Tool Discovery at Runtime

Psyche discovers available tools during request handling:

```python
# From HTTP server (http.py)
all_tools = list(request.tools or [])
all_tools.extend([Tool(**t) for t in MEMORY_TOOL_DEFINITIONS])
tool_desc = self._format_tool_descriptions(all_tools)
self.core.set_tool_descriptions(tool_desc)
```

### Tool Execution Patterns

1. **Elpis tools** - Called via MCP protocol:
```python
async def generate(self, messages, max_tokens, emotional_modulation):
    arguments = {
        "messages": messages,
        "max_tokens": max_tokens,
        "emotional_modulation": emotional_modulation,
    }
    result = await self._call_tool("generate", arguments)
    return GenerationResult(...)
```

2. **Memory tools** - Server-side execution in HTTP endpoint:
```python
async def _execute_memory_tool(self, tool_call: Dict) -> str:
    if name == "recall_memory":
        memories = await self.core.retrieve_memories(query, n_results)
        return json.dumps({"memories": memories})
```

3. **Client tools** - Returned for local execution:
```python
if client_calls:
    message = ChatMessage(
        role="assistant",
        content=accumulated_content,
        tool_calls=client_calls,
    )
    return ChatCompletionResponse(...)
```

### Tool Call Parsing

Psyche parses tool calls from LLM output using regex pattern:
```python
pattern = r"```tool_call\s*\n?(.*?)\n?```"
matches = re.findall(pattern, content, re.DOTALL)
```

---

## 5. Error Handling

### MCP Library Patch

**Location:** `src/shared/mcp_patch.py`

Fixes a race condition in MCP library where dictionary iteration fails during concurrent modification:

```python
class SafeIterDict(dict):
    def items(self):
        try:
            return list(super().items())
        except RuntimeError:
            return []  # Safely handle concurrent modification
```

Applied in `psyche-server` before imports: `apply_mcp_patch()`

### Connection Error Handling

1. **Elpis Connection Failure:**
```python
try:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            self._connected = True
except Exception as e:
    logger.error(f"Failed to connect to Elpis: {e}")
    raise  # Propagate to daemon
```

2. **Mnemosyne Connection Failure (Graceful Fallback):**
```python
try:
    async with self.mnemosyne_client.connect() as client:
        yield client
except Exception as e:
    logger.warning(f"Failed to connect to Mnemosyne: {e}")
    logger.warning("Continuing without persistent memory")
    yield None  # Continue without memory
```

3. **Tool Call Errors:**
```python
@server.call_tool()
async def call_tool(name, arguments):
    try:
        if name == "generate":
            result = await _handle_generate(ctx, arguments)
        return [TextContent(type="text", text=json.dumps(result))]
    except Exception as e:
        logger.exception(f"Tool call failed: {name}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
```

### HTTP Error Handling

```python
async def chat_completions(request, raw_request):
    try:
        return await self._generate_response(request, connection_id)
    except RuntimeError as e:
        error_msg = str(e)
        if "exceed context" in error_msg.lower():
            accumulated_content += "\n\n[Context limit reached...]"
        else:
            raise
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Timeout Handling

- Streaming: Stream task waits up to 1.0 seconds for completion
- Stale stream cleanup: Streams TTL is 10 minutes (`STREAM_TTL_SECONDS = 600`)
- Max active streams: 100

### Resource Cleanup

Stream cancellation on error:
```python
except Exception:
    try:
        await self._call_tool("generate_stream_cancel", {"stream_id": stream_id})
    except Exception:
        pass
    raise
```

---

## 6. Advanced Architectural Patterns

### Memory Tool Loop with Iteration Limits

The HTTP server implements a controlled loop for memory tool execution (max 5 iterations):

```python
MAX_MEMORY_ITERATIONS = 5
for iteration in range(MAX_MEMORY_ITERATIONS):
    result = await self.core.generate(...)
    tool_calls = self._parse_tool_calls(content)
    memory_calls, client_calls = self._separate_tool_calls(tool_calls)

    if memory_calls:
        # Execute memory tools internally

    if client_calls:
        return ChatCompletionResponse(...)  # Return for client execution
```

### Emotional Modulation Integration

MCP tools automatically apply emotional state to LLM behavior:

```python
# Get modulated parameters based on emotion
if emotional_modulation and temperature is None:
    params = ctx.emotion_state.get_modulated_params()
    temperature = params["temperature"]
    top_p = params["top_p"]

# Get steering coefficients
emotion_coefficients = ctx.emotion_state.get_steering_coefficients()

# Pass to LLM
content = await ctx.llm.chat_completion(
    messages=messages,
    emotion_coefficients=emotion_coefficients,
)
```

### Server-side Memory Consolidation Loop

Runs independently every 300s:

```python
async def _consolidation_loop(self):
    while self._running:
        await asyncio.sleep(self.config.consolidation_interval)
        await self._maybe_consolidate()
```

### Dream Scheduling Based on Connection State

```python
def on_client_disconnect(self, client_id):
    if not self._connections and self.config.dream_enabled:
        self._schedule_dreaming()

def _schedule_dreaming(self):
    async def delayed_dream():
        await asyncio.sleep(self.config.dream_delay_seconds)
        if not self._connections and self.dream_handler:
            await self.dream_handler.start_dreaming()
```

---

## 7. Key Strengths

1. **Clean Separation**: MCP servers are independent processes; Psyche is a client orchestrating them
2. **Resilience**: Graceful degradation when Mnemosyne unavailable
3. **Thread Safety**: Session locking prevents race conditions in async context
4. **Error Recovery**: Specific handling for different error types
5. **Lifecycle Safety**: Context managers ensure clean resource cleanup
6. **Configuration Flexibility**: TOML + environment variables + CLI args with clear precedence
7. **Tool Abstraction**: Unified interface for diverse server tools via MCP protocol

---

## 8. Potential Improvements

1. **Reconnection Logic**: No retry mechanism if Elpis connection fails mid-session
2. **Tool Call Timeout**: Some tool calls could benefit from explicit timeouts
3. **Resource Limits**: While streaming has per-stream limits, no global resource quotas
4. **Circuit Breaker**: Could improve resilience with circuit breaker pattern
5. **Tool Versioning**: No mechanism to handle MCP server upgrades/version mismatches
6. **External Server Configuration**: No JSON/TOML config file for arbitrary external MCP servers

---

## 9. Files Reviewed

| File | Lines | Purpose |
|------|-------|---------|
| `src/psyche/mcp/client.py` | ~350 | MCP client implementations |
| `src/psyche/server/daemon.py` | ~450 | Server lifecycle management |
| `src/psyche/server/http.py` | ~800 | HTTP API with tool execution |
| `src/psyche/config/settings.py` | ~200 | Settings infrastructure |
| `src/elpis/server.py` | 753 | Inference MCP server |
| `src/mnemosyne/server.py` | ~400 | Memory MCP server |
| `src/shared/mcp_patch.py` | ~50 | MCP library patches |
| `configs/*.toml` | various | Configuration templates |
