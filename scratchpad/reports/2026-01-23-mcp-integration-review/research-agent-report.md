# MCP Integration Research Report

**Agent**: Research Agent
**Date**: 2026-01-23
**Task**: Research MCP client/harness integration patterns for potential adoption in Elpis

---

## Executive Summary

The Model Context Protocol (MCP) is an open standard by Anthropic (now under the Linux Foundation) for connecting LLMs to external data sources and tools. This report documents how major MCP clients configure and integrate external servers, with focus on patterns applicable to Elpis.

Key findings:
1. **Configuration is JSON-based** across all major implementations
2. **Stdio transport dominates** for local servers; Streamable HTTP for remote
3. **Capabilities negotiation** is critical to the protocol handshake
4. **Security via sandboxing** is increasingly important
5. **Elpis already follows best practices** in its MCP client implementation

---

## 1. MCP Protocol Overview

### Core Architecture

MCP uses a client-server model with three participant types:
- **Hosts**: LLM applications that initiate connections (e.g., Claude Desktop, VS Code)
- **Clients**: Connectors within the host that manage server connections
- **Servers**: Services exposing context and capabilities

Communication uses JSON-RPC 2.0 over two primary transports:
- **Stdio**: Client spawns server as subprocess, communicates via stdin/stdout
- **Streamable HTTP**: Server runs independently, client uses HTTP POST/GET

### Server Features

Servers can expose three types of capabilities:
| Feature | Description | Discovery Method |
|---------|-------------|------------------|
| **Tools** | Functions the LLM can invoke | `tools/list` |
| **Resources** | Data/content to read | `resources/list` |
| **Prompts** | Template messages | `prompts/list` |

### Protocol Lifecycle

1. **Initialization**: Capability negotiation via `initialize` request/response
2. **Operation**: Normal JSON-RPC communication
3. **Shutdown**: Graceful disconnection

---

## 2. Claude Desktop Configuration

### Configuration File Location

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

### Configuration Format

```json
{
  "mcpServers": {
    "<server-name>": {
      "command": "<executable>",
      "args": ["<arg1>", "<arg2>"],
      "env": {
        "API_KEY": "value",
        "OTHER_VAR": "value"
      }
    }
  }
}
```

### Real-World Examples

**Filesystem Server:**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/username/Desktop",
        "/Users/username/Downloads"
      ]
    }
  }
}
```

**GitHub Server with Docker:**
```json
{
  "mcpServers": {
    "github": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxxxxxxxxxx"
      }
    }
  }
}
```

### Key Patterns

1. **npx for Node.js servers** - No installation required
2. **Docker for isolation** - Sandboxed execution
3. **Environment variables** - Scoped per server process
4. **Restart required** - Config changes need app restart

---

## 3. Claude Code Configuration

### Configuration Locations

| Scope | File | Purpose |
|-------|------|---------|
| User | `~/.claude.json` | Available across all projects |
| Project | `.mcp.json` (project root) | Shared with team |
| Local | `.claude.json` (project dir) | Personal, project-specific |

### CLI Management

```bash
# Add server
claude mcp add github --scope user

# List servers
claude mcp list

# Get details
claude mcp get github

# Remove server
claude mcp remove github

# Check status (in Claude Code)
/mcp
```

### Configuration Format

**User-level (`~/.claude.json`):**
```json
{
  "mcpServers": {
    "mcp-omnisearch": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "mcp-omnisearch"],
      "env": {
        "TAVILY_API_KEY": "",
        "BRAVE_API_KEY": ""
      }
    }
  }
}
```

**Project-specific (nested structure):**
```json
{
  "projects": {
    "/path/to/your/project": {
      "mcpServers": {
        "brave-search": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-brave-search"]
        }
      }
    }
  }
}
```

### Notable Features

1. **Scope hierarchy** - Local > Project > User
2. **Type field** - Explicit `"type": "stdio"` or `"type": "http"`
3. **Timeout configuration** - `MCP_TIMEOUT` environment variable
4. **Import from Claude Desktop** - Can reuse existing configs

---

## 4. VS Code / GitHub Copilot Configuration

### Configuration Locations

| Scope | File |
|-------|------|
| Workspace | `.vscode/mcp.json` |
| User | VS Code `settings.json` |

### Configuration Format

**settings.json:**
```json
{
  "mcp": {
    "servers": {
      "your-mcp-server": {
        "command": "node",
        "args": ["<path>/index.js"],
        "env": {
          "YOUR_API_KEY": "<key>"
        }
      }
    }
  }
}
```

**SSE/HTTP server:**
```json
{
  "mcpServers": {
    "Pieces": {
      "url": "http://localhost:39300/model_context_protocol/2024-11-05/sse"
    }
  }
}
```

### Notable Features

1. **Agent Mode required** - Enable `chat.agent.enabled`
2. **Tool limit** - 128 tools max per request
3. **Import from Claude Desktop** - Automatic detection
4. **Enterprise policy** - MCP can be disabled organization-wide

---

## 5. Open Source MCP Clients

### Notable Implementations

| Client | Description | Configuration |
|--------|-------------|---------------|
| **Dive** | Desktop MCP host | JSON config similar to Claude Desktop |
| **HyperChat** | Multi-LLM chat client | Per-server JSON config |
| **5ire** | Cross-platform assistant | JSON + local knowledge base |
| **BeeAI Framework** (IBM) | Production agent framework | Programmatic + config file |
| **Langflow** | Visual AI builder | GUI + JSON export |
| **MCPHost** | Multi-server web app | JSON config with persistence |
| **CopilotKit open-mcp-client** | Reference implementation | TypeScript SDK patterns |

### Common Patterns

1. **JSON configuration** is universal
2. **Server name as key** in mcpServers object
3. **Stdio default** for local, HTTP/SSE for remote
4. **Environment injection** via `env` object

---

## 6. Transport Mechanisms

### Stdio Transport (Recommended for Local)

**Characteristics:**
- Client spawns server as subprocess
- Communication via stdin/stdout (JSON-RPC messages)
- Messages newline-delimited
- stderr for logging (optional capture)
- Microsecond latency
- Most interoperable

**Message Format:**
```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}
```

### Streamable HTTP Transport (Modern Remote)

**Characteristics:**
- Server runs independently
- Single endpoint for POST/GET
- Optional SSE for streaming
- Session management via headers
- Connection recovery supported

**Request:**
```http
POST /mcp HTTP/1.1
Content-Type: application/json
Accept: application/json, text/event-stream
MCP-Protocol-Version: 2025-11-25

{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{...}}
```

### SSE Transport (Deprecated)

Legacy transport for remote servers. Two separate endpoints required. Use Streamable HTTP for new implementations.

---

## 7. Initialization Handshake

### Sequence

```
Client                          Server
  |                               |
  |--- initialize request ------->|
  |                               |
  |<-- initialize response -------|
  |                               |
  |--- initialized notification ->|
  |                               |
  |    [Operation Phase]          |
```

### Initialize Request

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-11-25",
    "capabilities": {
      "roots": { "listChanged": true },
      "sampling": {},
      "elicitation": { "form": {}, "url": {} }
    },
    "clientInfo": {
      "name": "ExampleClient",
      "version": "1.0.0"
    }
  }
}
```

### Initialize Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-11-25",
    "capabilities": {
      "logging": {},
      "prompts": { "listChanged": true },
      "resources": { "subscribe": true, "listChanged": true },
      "tools": { "listChanged": true }
    },
    "serverInfo": {
      "name": "ExampleServer",
      "version": "1.0.0"
    },
    "instructions": "Optional server instructions"
  }
}
```

### Initialized Notification

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

### Version Negotiation

1. Client sends supported version (preferably latest)
2. Server responds with same version if supported, else its latest
3. Client disconnects if versions incompatible

---

## 8. Tools, Resources, and Prompts

### Tool Definition

```json
{
  "name": "get_weather",
  "title": "Weather Information Provider",
  "description": "Get current weather for a location",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or zip code"
      }
    },
    "required": ["location"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "temperature": { "type": "number" },
      "conditions": { "type": "string" }
    }
  }
}
```

### Tool Call Flow

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": { "location": "New York" }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Temperature: 72F, Partly cloudy"
      }
    ],
    "isError": false
  }
}
```

### Content Types

- **text**: Plain text content
- **image**: Base64-encoded image data
- **audio**: Base64-encoded audio data
- **resource_link**: URI to a resource
- **resource**: Embedded resource content

---

## 9. Environment Variable Handling

### Architecture Constraint

MCP servers run as subprocesses spawned by the client. The server's working directory may not match the development directory, making `.env` files unreliable.

### Best Practices

1. **Host Injection**: Define env vars in client config, injected at spawn
2. **Variable Isolation**: Each server has isolated environment
3. **Secrets in Config**: Use config file `env` object (not ideal for sharing)
4. **External Secrets Manager**: HashiCorp Vault, AWS Secrets Manager for production

### Configuration Patterns

**Direct values (not recommended for secrets):**
```json
{
  "env": {
    "API_KEY": "sk-actual-secret-value"
  }
}
```

**Environment variable reference (some clients):**
```json
{
  "env_vars": ["OPENAI_API_KEY", "DATABASE_URL"]
}
```

**Bearer token for HTTP:**
```json
{
  "bearer_token_env_var": "API_TOKEN"
}
```

### Common Issues

1. **JSON syntax errors** - Trailing commas, missing braces
2. **Process caching** - Client caches config, restart required
3. **Variable not found** - Env var not in host process environment

---

## 10. Security Considerations

### Sandboxing Recommendations

From the MCP specification:

> Implementations should execute MCP server commands in a sandboxed environment with minimal default privileges, launch MCP servers with restricted access to the file system, network, and other system resources.

### Sandboxing Technologies

1. **Docker containers** - Full process isolation
2. **chroot jails** - Filesystem restriction
3. **Application sandboxes** - OS-level sandboxing (macOS sandbox, Linux seccomp)
4. **Read-only mounts** - Prevent filesystem modification

### Security Best Practices

1. **Principle of Least Privilege**
   - Grant only necessary permissions
   - Scope tokens narrowly
   - Time-limit access tokens

2. **User Consent**
   - Confirm before tool invocation
   - Show tool inputs before execution
   - Log all tool usage

3. **Credential Management**
   - Never hardcode secrets
   - Use secrets managers
   - Rotate credentials regularly

4. **Network Isolation**
   - Prevent unauthorized exfiltration
   - Use allowlists for network access
   - Monitor outbound connections

5. **Input Validation**
   - Validate all tool inputs
   - Sanitize tool outputs
   - Implement rate limiting

---

## 11. Current Elpis Implementation Analysis

### What Elpis Does Well

Looking at `/home/lemoneater/Projects/Elpis/src/psyche/mcp/client.py`:

1. **Uses official Python MCP SDK** - `from mcp import ClientSession, StdioServerParameters`
2. **Proper async context managers** - Connection lifecycle handled correctly
3. **Session locking** - `asyncio.Lock()` prevents concurrent access
4. **Environment injection** - `env` parameter passed to `StdioServerParameters`
5. **Initialization sequence** - `await session.initialize()` called properly
6. **Error handling** - Exceptions logged with traceback
7. **Clean disconnection** - Context manager ensures cleanup

### Code Example (Current Implementation)

```python
server_params = StdioServerParameters(
    command=self.server_command,
    args=self.server_args,
    env=env,
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        self._session = session
        await session.initialize()
        # ... operations ...
```

This follows the MCP SDK best practices exactly.

### Potential Enhancements

1. **External Server Configuration**
   - Add JSON config file support (e.g., `.psyche.json` or `psyche.yaml`)
   - Support multiple server connections
   - Enable hot-reload of server configs

2. **HTTP Transport Support**
   - Add Streamable HTTP client for remote servers
   - Enable cloud-hosted MCP servers

3. **Capability-Based Feature Gates**
   - Check server capabilities before calling features
   - Graceful degradation when features unavailable

4. **Tool Discovery UI**
   - List available tools from all connected servers
   - Show tool schemas and descriptions

---

## 12. Recommendations for Elpis

### Short-term (Configuration File)

Add a configuration file format for external MCP servers:

**Proposed `~/.psyche.json`:**
```json
{
  "mcpServers": {
    "elpis": {
      "command": "elpis-server",
      "args": ["--model", "mistral"],
      "env": {
        "ELPIS_QUIET": "1"
      }
    },
    "mnemosyne": {
      "command": "mnemosyne-server",
      "args": [],
      "env": {
        "MNEMOSYNE_QUIET": "1"
      }
    },
    "external-tools": {
      "command": "npx",
      "args": ["-y", "@some/mcp-tool-server"]
    }
  }
}
```

### Medium-term (Multi-Server Support)

1. **ServerRegistry class** - Manage multiple MCP server connections
2. **Tool namespace prefixing** - `server_name.tool_name` for disambiguation
3. **Lazy connection** - Connect to servers on-demand
4. **Health monitoring** - Reconnect on server failure

### Long-term (Remote Servers)

1. **Streamable HTTP transport** - Connect to remote MCP servers
2. **Authentication** - OAuth, API keys, bearer tokens
3. **Server discovery** - Registry of available servers
4. **Distributed tooling** - Tools across network boundary

---

## 13. Sources

### Official Documentation
- [MCP Specification (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25)
- [MCP GitHub Organization](https://github.com/modelcontextprotocol)
- [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk)

### Client Documentation
- [Claude Desktop MCP Setup](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop)
- [Claude Code MCP Docs](https://code.claude.com/docs/en/mcp)
- [VS Code MCP Servers](https://code.visualstudio.com/docs/copilot/customization/mcp-servers)
- [GitHub Copilot MCP](https://docs.github.com/copilot/customizing-copilot/using-model-context-protocol)

### Open Source Clients
- [awesome-mcp-clients](https://github.com/punkpeye/awesome-mcp-clients)
- [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers)
- [IBM MCP Collection](https://github.com/IBM/mcp)

### Security Resources
- [MCP Security Best Practices](https://modelcontextprotocol.io/specification/draft/basic/security_best_practices)
- [Claude Code Sandboxing](https://code.claude.com/docs/en/sandboxing)
- [MCP Security Overview (Datadog)](https://www.datadoghq.com/blog/monitor-mcp-servers/)

### Technical Guides
- [MCP Transports Comparison (MCPcat)](https://mcpcat.io/guides/comparing-stdio-sse-streamablehttp/)
- [JSON-RPC in MCP (MCPcat)](https://mcpcat.io/guides/understanding-json-rpc-protocol-mcp/)
- [Building MCP Clients (Real Python)](https://realpython.com/python-mcp-client/)

---

## Appendix A: JSON-RPC Message Reference

### Core Methods

| Method | Direction | Purpose |
|--------|-----------|---------|
| `initialize` | Client -> Server | Start session, negotiate capabilities |
| `notifications/initialized` | Client -> Server | Signal ready for operations |
| `tools/list` | Client -> Server | Discover available tools |
| `tools/call` | Client -> Server | Invoke a tool |
| `resources/list` | Client -> Server | Discover available resources |
| `resources/read` | Client -> Server | Read resource content |
| `prompts/list` | Client -> Server | Discover available prompts |
| `prompts/get` | Client -> Server | Get prompt template |

### Notification Methods

| Method | Direction | Purpose |
|--------|-----------|---------|
| `notifications/tools/list_changed` | Server -> Client | Tool list updated |
| `notifications/resources/list_changed` | Server -> Client | Resource list updated |
| `notifications/progress` | Either | Report progress on request |
| `notifications/cancelled` | Either | Request was cancelled |

---

## Appendix B: Capability Reference

### Client Capabilities

| Capability | Sub-capabilities | Description |
|------------|------------------|-------------|
| `roots` | `listChanged` | Filesystem root boundaries |
| `sampling` | - | LLM sampling requests |
| `elicitation` | `form`, `url` | Request info from user |
| `tasks` | `requests` | Task-augmented requests |

### Server Capabilities

| Capability | Sub-capabilities | Description |
|------------|------------------|-------------|
| `prompts` | `listChanged` | Prompt templates |
| `resources` | `subscribe`, `listChanged` | Data resources |
| `tools` | `listChanged` | Callable tools |
| `logging` | - | Structured logging |
| `completions` | - | Argument autocompletion |
| `tasks` | `list`, `cancel`, `requests` | Task management |
