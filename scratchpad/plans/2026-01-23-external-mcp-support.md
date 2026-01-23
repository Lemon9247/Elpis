# External MCP Server Support for Psyche and Hermes

## Overview

Add support for connecting to arbitrary external MCP servers in both Psyche (server-side) and Hermes (client-side), enabling extensibility through the MCP ecosystem.

**Architecture:**
- Psyche connects to "core" MCP servers (always available, used during dreaming)
- Hermes connects to "client-specific" MCP servers (user-controlled, local execution)

**Transport Types:**
- **STDIO**: Local subprocess servers (spawn command, communicate via stdin/stdout)
- **SSE/HTTP**: Remote servers running independently (connect to URL)

**Important: Existing Connections Unchanged**

Elpis and Mnemosyne connections remain exactly as they are:

```
PsycheDaemon (current - unchanged)
├── ElpisClient (STDIO) ──────► elpis-server        [specialized methods: generate, emotion]
├── MnemosyneClient (STDIO) ──► mnemosyne-server    [specialized methods: store, search]

PsycheDaemon (with external MCP - additive)
├── ElpisClient (STDIO) ──────► elpis-server        [unchanged]
├── MnemosyneClient (STDIO) ──► mnemosyne-server    [unchanged]
└── MCPServerManager ─────────► external servers    [NEW - generic call_tool only]
    ├── MCPClient (STDIO) ────► github-mcp-server
    ├── MCPClient (SSE) ──────► http://remote-tools/sse
    └── ...
```

The new `MCPClient` is for **external servers only** - servers discovered at runtime via configuration. Elpis/Mnemosyne have specialized client classes with domain-specific methods.

---

## Phase 1: Shared Infrastructure

### New Files

**`src/shared/mcp/__init__.py`**
```python
from .config import MCPServerConfig, TransportType
from .base_client import MCPClient
from .manager import MCPServerManager
```

**`src/shared/mcp/config.py`** - Server configuration dataclass
```python
from enum import Enum

class TransportType(str, Enum):
    STDIO = "stdio"      # Local subprocess
    SSE = "sse"          # HTTP/SSE remote server

@dataclass
class MCPServerConfig:
    name: str                           # Identifier (used for tool namespacing)
    transport: TransportType = TransportType.STDIO

    # For STDIO transport (local subprocess)
    command: Optional[str] = None       # Command to launch server
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)

    # For SSE transport (remote server)
    url: Optional[str] = None           # Server URL (e.g., "http://localhost:3001/sse")

    # Common settings
    timeout: float = 30.0
    enabled: bool = True

    def validate(self) -> None:
        if self.transport == TransportType.STDIO and not self.command:
            raise ValueError(f"STDIO transport requires 'command' for server {self.name}")
        if self.transport == TransportType.SSE and not self.url:
            raise ValueError(f"SSE transport requires 'url' for server {self.name}")
```

**`src/shared/mcp/base_client.py`** - Generic MCP client supporting both transports
```python
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client

class MCPClient:
    """Generic MCP client supporting STDIO and SSE transports."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.namespace = config.name
        self._session: Optional[ClientSession] = None
        self._session_lock: asyncio.Lock = asyncio.Lock()
        self._tools: List[Tool] = []

    @asynccontextmanager
    async def connect(self) -> AsyncIterator["MCPClient"]:
        if self.config.transport == TransportType.STDIO:
            async with self._connect_stdio() as client:
                yield client
        else:
            async with self._connect_sse() as client:
                yield client

    @asynccontextmanager
    async def _connect_stdio(self) -> AsyncIterator["MCPClient"]:
        """Connect via STDIO (local subprocess)."""
        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env={**os.environ, **self.config.env},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                self._session = session
                await session.initialize()
                self._tools = (await session.list_tools()).tools
                yield self

    @asynccontextmanager
    async def _connect_sse(self) -> AsyncIterator["MCPClient"]:
        """Connect via SSE (remote HTTP server)."""
        async with sse_client(self.config.url) as (read, write):
            async with ClientSession(read, write) as session:
                self._session = session
                await session.initialize()
                self._tools = (await session.list_tools()).tools
                yield self
```

- `discover_tools()` -> List[Tool] via `list_tools()`
- `call_tool(name, arguments)` with session lock
- Tool namespacing support (e.g., `github.list_repos`)

**`src/shared/mcp/manager.py`** - Multi-server manager
- `start_all()` - Connect to all configured servers, discover tools
- `stop_all()` - Graceful disconnect
- `get_all_tools()` - All tools in OpenAI format
- `execute_tool(name, args)` - Route to correct server

### Tests
- `tests/shared/mcp/test_base_client.py`
- `tests/shared/mcp/test_manager.py`

---

## Phase 2: Psyche Integration

### Modified Files

**`src/psyche/config/settings.py`** (lines 136-176)
```python
class MCPServerSettings(BaseSettings):
    name: str
    command: str
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

class MCPSettings(BaseSettings):
    enabled: bool = False
    servers: List[MCPServerSettings] = Field(default_factory=list)

class ServerSettings(BaseSettings):
    # ... existing ...
    mcp: MCPSettings = Field(default_factory=MCPSettings)
```

**`src/psyche/server/daemon.py`** (lines 103-125)
- Initialize `MCPServerManager` with configs from settings
- Call `await mcp_manager.start_all()` after Elpis/Mnemosyne connect
- Pass manager to HTTP server
- Call `await mcp_manager.stop_all()` in shutdown

**`src/psyche/server/http.py`** (lines 39, 690-702, 657-688)
- Change `MEMORY_TOOLS` to `SERVER_SIDE_TOOLS` (dynamic set)
- Add external MCP tool names to server-side set on init
- Update `_separate_tool_calls()` to check `SERVER_SIDE_TOOLS`
- Add `_execute_external_tool()` method routing to MCPServerManager

### Configuration

**`configs/psyche.toml`**
```toml
[server.mcp]
enabled = false

# STDIO transport: Local subprocess server
[[server.mcp.servers]]
name = "websearch"
transport = "stdio"
command = "uvx"
args = ["mcp-server-brave-search"]
enabled = true
env = { BRAVE_API_KEY = "${BRAVE_API_KEY}" }

# SSE transport: Remote HTTP server
[[server.mcp.servers]]
name = "knowledge-base"
transport = "sse"
url = "http://localhost:3001/sse"
enabled = true
```

### Tests
- `tests/psyche/integration/test_external_mcp.py` - Mock MCP server test

---

## Phase 3: Hermes Integration

### New Files

**`src/hermes/mcp/__init__.py`**

**`src/hermes/mcp/tool_adapter.py`** - Convert MCP tools to ToolDefinition
```python
class MCPToolAdapter:
    def __init__(self, manager: MCPServerManager): ...

    def create_tool_definition(self, tool: Tool, namespace: str) -> ToolDefinition:
        # Create Pydantic model from inputSchema
        # Create async handler that calls manager.execute_tool()
        # Return ToolDefinition compatible with ToolEngine
```

### Modified Files

**`src/hermes/config/settings.py`** (lines 51-87)
```python
class MCPSettings(BaseSettings):
    enabled: bool = False
    servers: List[MCPServerSettings] = Field(default_factory=list)

class Settings(BaseSettings):
    # ... existing ...
    mcp: MCPSettings = Field(default_factory=MCPSettings)
```

**`src/hermes/cli.py`** (lines 129-141)
```python
# After line 136 (ToolEngine creation):
if settings.mcp.enabled:
    mcp_configs = [MCPServerConfig(**s.model_dump()) for s in settings.mcp.servers]
    mcp_manager = MCPServerManager(mcp_configs)
    await mcp_manager.start_all()  # Need to handle in async context

    adapter = MCPToolAdapter(mcp_manager)
    for server_name, tools in mcp_manager.get_tools_by_server().items():
        for tool in tools:
            tool_def = adapter.create_tool_definition(tool, server_name)
            tool_engine.register_tool(tool_def)

# Line 139: Now includes MCP tools
client.set_tools(tool_engine.get_openai_tool_definitions())
```

### Configuration

**`configs/hermes.toml`**
```toml
[mcp]
enabled = false

# STDIO transport: Local subprocess server
[[mcp.servers]]
name = "github"
transport = "stdio"
command = "uvx"
args = ["mcp-server-github"]
enabled = true
env = { GITHUB_TOKEN = "${GITHUB_TOKEN}" }

# STDIO transport: npm package server
[[mcp.servers]]
name = "filesystem"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"]
enabled = true

# SSE transport: Remote server
[[mcp.servers]]
name = "remote-tools"
transport = "sse"
url = "http://tools.example.com/mcp/sse"
enabled = false
```

### Tests
- `tests/hermes/unit/test_mcp_tool_adapter.py`
- `tests/hermes/integration/test_external_mcp.py`

---

## Phase 4: Polish

- Graceful error handling for server crashes
- Tool name collision detection and warnings
- Logging for MCP operations
- Documentation updates

---

## Phase 5 (Optional): Refactor Elpis/Mnemosyne Clients

**Goal:** Reduce code duplication by having specialized clients inherit from `MCPClient`.

### Current State (duplicated pattern)

Both `ElpisClient` and `MnemosyneClient` in `src/psyche/mcp/client.py` share ~150 lines of identical code:

```python
# Duplicated in both classes:
- __init__ with server_command, server_args, quiet, _session_lock
- connect() async context manager with stdio_client
- _ensure_connected() guard
- _call_tool(name, arguments) with lock and JSON parsing
```

### Refactored State

**`src/shared/mcp/base_client.py`** becomes the base:
```python
class MCPClient:
    """Base MCP client with common infrastructure."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._session: Optional[ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._connected = False

    @asynccontextmanager
    async def connect(self) -> AsyncIterator["MCPClient"]:
        # STDIO or SSE connection based on config.transport
        ...

    async def call_tool(self, name: str, args: Dict) -> Dict:
        async with self._session_lock:
            result = await self._session.call_tool(name, args)
            return json.loads(result.content[0].text) if result.content else {}
```

**`src/psyche/mcp/client.py`** - Specialized clients extend base:
```python
from shared.mcp import MCPClient, MCPServerConfig, TransportType

class ElpisClient(MCPClient):
    """Elpis inference server client with specialized methods."""

    def __init__(self, server_command: str = "elpis-server", ...):
        config = MCPServerConfig(
            name="elpis",
            transport=TransportType.STDIO,
            command=server_command,
            args=server_args or [],
        )
        super().__init__(config)

    # Specialized methods (not in base class)
    async def generate(self, messages, max_tokens, ...) -> GenerationResult:
        result = await self.call_tool("generate", {...})
        return GenerationResult(**result)

    async def update_emotion(self, event_type, intensity) -> Dict:
        return await self.call_tool("update_emotion", {...})

    async def get_capabilities(self) -> Dict:
        return await self.call_tool("get_capabilities", {})


class MnemosyneClient(MCPClient):
    """Mnemosyne memory server client with specialized methods."""

    def __init__(self, server_command: str = "mnemosyne-server", ...):
        config = MCPServerConfig(
            name="mnemosyne",
            transport=TransportType.STDIO,
            command=server_command,
            args=server_args or [],
        )
        super().__init__(config)

    # Specialized methods
    async def store_memory(self, content, memory_type, ...) -> Dict:
        return await self.call_tool("store_memory", {...})

    async def search_memories(self, query, n_results, ...) -> Dict:
        return await self.call_tool("search_memories", {...})
```

### Benefits

| Aspect | Before | After |
|--------|--------|-------|
| Lines of code | ~600 (duplicated) | ~350 (shared base) |
| Error handling | Duplicated in each | Single place in base |
| New transport support | Edit both clients | Edit base only |
| Testing | Test each client | Test base + specialized methods |

### Migration Steps

1. Create `MCPClient` base class with common infrastructure
2. Update `ElpisClient` to extend `MCPClient`, keep specialized methods
3. Update `MnemosyneClient` to extend `MCPClient`, keep specialized methods
4. Update tests to verify inheritance works correctly
5. Verify daemon initialization still works

### Backward Compatibility

The public API remains identical:
```python
# Before and after - same usage
client = ElpisClient(server_command="elpis-server")
async with client.connect() as c:
    result = await c.generate(messages, max_tokens=1024)
```

---

## Critical Files Summary

| File | Purpose |
|------|---------|
| `src/shared/mcp/base_client.py` | NEW: Generic MCP client |
| `src/shared/mcp/manager.py` | NEW: Multi-server manager |
| `src/psyche/config/settings.py` | ADD: MCPSettings |
| `src/psyche/server/daemon.py` | ADD: MCP initialization |
| `src/psyche/server/http.py:39` | CHANGE: SERVER_SIDE_TOOLS dynamic |
| `src/hermes/config/settings.py` | ADD: MCPSettings |
| `src/hermes/cli.py:136` | ADD: MCP initialization |
| `src/hermes/mcp/tool_adapter.py` | NEW: MCP to ToolDefinition adapter |

---

## Verification

### Unit Tests
```bash
./venv/bin/pytest tests/shared/mcp/ -v
./venv/bin/pytest tests/psyche/unit/test_settings.py -v
./venv/bin/pytest tests/hermes/unit/test_mcp_tool_adapter.py -v
```

### Integration Test (Psyche)
```bash
# Start psyche-server with MCP config
PSYCHE_MCP__ENABLED=true ./venv/bin/psyche-server

# In another terminal, test tool discovery
curl http://localhost:8741/health
```

### Integration Test (Hermes)
```bash
# Start psyche-server first
./venv/bin/psyche-server

# Start hermes with MCP config
HERMES_MCP__ENABLED=true ./venv/bin/hermes

# Use /status to see available tools
```

### End-to-End Test
1. Configure filesystem MCP server in hermes.toml
2. Start psyche-server and hermes
3. Ask "List files in /tmp" - should invoke `filesystem.list_directory`

---

## Session Estimates

| Phase | Sessions |
|-------|----------|
| Phase 1: Shared Infrastructure | 1-2 |
| Phase 2: Psyche Integration | 1-2 |
| Phase 3: Hermes Integration | 1-2 |
| Phase 4: Polish | 1 |
| **Core Total** | **4-7** |
| Phase 5: Refactor Elpis/Mnemosyne (optional) | 1 |
| **Full Total** | **5-8** |
