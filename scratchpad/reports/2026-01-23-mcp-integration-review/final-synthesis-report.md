# MCP Integration Review - Final Synthesis Report

**Date**: 2026-01-23
**Task**: Review external MCP server integration in Elpis and compare with industry patterns

---

## Executive Summary

This multi-agent review examined Elpis's MCP integration architecture and compared it against patterns used by Claude Desktop, Claude Code, VS Code, and other MCP clients. The key finding is that **Elpis already implements MCP best practices** for its internal servers (Elpis, Mnemosyne), but lacks a configuration mechanism for **arbitrary external MCP servers**.

### Key Conclusions

| Aspect | Elpis Status | Industry Standard | Gap |
|--------|--------------|-------------------|-----|
| MCP SDK usage | Official Python SDK | Official SDKs | None |
| Transport | Stdio | Stdio (local) / HTTP (remote) | Missing HTTP |
| Error handling | Comprehensive | Varies | None |
| Configuration | TOML + CLI | JSON (`mcpServers` object) | Different format |
| External servers | Hardcoded Elpis/Mnemosyne | Dynamic via config file | **Primary gap** |
| Multi-server | Two built-in | Unlimited | Limited |

---

## Part 1: Current Elpis Architecture

### Strengths (from Codebase Agent)

1. **Clean Separation of Concerns**
   - Three independent components: Elpis (inference), Mnemosyne (memory), Psyche (orchestration)
   - Each runs as a separate process, communicating via MCP stdio

2. **Robust Connection Management**
   - `asyncio.Lock` protects all MCP session operations
   - Context managers ensure proper lifecycle cleanup
   - Graceful degradation if Mnemosyne unavailable

3. **Sophisticated Tool Handling**
   - Server-side memory tools (executed internally)
   - Client tools (returned to caller)
   - Tool call parsing from LLM output

4. **Production-Ready Error Handling**
   - MCP library race condition patch (`SafeIterDict`)
   - Timeout handling for streams (TTL: 10 min)
   - Context overflow detection and recovery

### Current Limitations

1. **No external server configuration**
   - Only Elpis and Mnemosyne are supported
   - Server commands hardcoded in `ServerConfig`

2. **No reconnection logic**
   - If server dies mid-session, no retry

3. **No HTTP transport**
   - Cannot connect to remote MCP servers

4. **No tool versioning**
   - No capability checking for version mismatches

---

## Part 2: Industry Patterns

### Universal Configuration Format

All major MCP clients use the same JSON structure:

```json
{
  "mcpServers": {
    "<server-name>": {
      "command": "<executable>",
      "args": ["<arg1>", "<arg2>"],
      "env": {
        "API_KEY": "value"
      }
    }
  }
}
```

### Configuration File Locations

| Client | User Config | Project Config |
|--------|-------------|----------------|
| Claude Desktop | `~/.config/Claude/claude_desktop_config.json` | N/A |
| Claude Code | `~/.claude.json` | `.mcp.json` or `.claude.json` |
| VS Code | `settings.json` | `.vscode/mcp.json` |

### Transport Selection

| Use Case | Transport | Rationale |
|----------|-----------|-----------|
| Local servers | Stdio | Lowest latency, simplest setup |
| Remote servers | Streamable HTTP | Independent lifecycle, scalable |
| Legacy remote | SSE | Deprecated, avoid for new work |

### Security Patterns

1. **Sandboxing** - Docker, chroot, seccomp
2. **Environment isolation** - Per-server env vars
3. **Least privilege** - Scoped API tokens
4. **User consent** - Confirm before tool execution

---

## Part 3: Gap Analysis

### What Elpis is Missing

| Feature | Priority | Complexity | Value |
|---------|----------|------------|-------|
| External server config file | High | Low | Enables extensibility |
| Multi-server registry | High | Medium | Manage N servers |
| Tool namespacing | Medium | Low | Avoid collisions |
| HTTP transport | Medium | Medium | Remote servers |
| Server health checks | Low | Low | Better resilience |
| Hot-reload config | Low | Medium | Developer experience |

### What Elpis Has That Others Don't

1. **Emotional modulation integration** - Unique to Elpis
2. **Memory consolidation loop** - Automatic background processing
3. **Dream scheduling** - Idle-time processing
4. **Graceful memory fallback** - Works without Mnemosyne

---

## Part 4: Recommendations

### Short-Term: Add Configuration File Support

Create `~/.psyche.json` (or `~/.config/psyche/servers.json`):

```json
{
  "mcpServers": {
    "elpis": {
      "command": "elpis-server",
      "args": ["--model", "mistral"],
      "env": { "ELPIS_QUIET": "1" },
      "required": true
    },
    "mnemosyne": {
      "command": "mnemosyne-server",
      "args": [],
      "env": { "MNEMOSYNE_QUIET": "1" },
      "required": false
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"],
      "required": false
    }
  }
}
```

**Implementation**:
- Add `ServerRegistry` class to manage multiple connections
- Parse config at daemon startup
- Connect to each server as configured
- Mark `required: true` servers as critical (fail if unavailable)

### Medium-Term: Multi-Server Architecture

1. **ServerRegistry Class**
   ```python
   class ServerRegistry:
       servers: Dict[str, MCPClient]

       async def connect_all(self, config: Dict) -> None: ...
       async def get_server(self, name: str) -> MCPClient: ...
       def list_tools(self) -> List[Tool]: ...
   ```

2. **Tool Namespacing**
   - Prefix tools with server name: `filesystem.read_file`
   - Maintain backward compatibility for Elpis/Mnemosyne tools

3. **Lazy Connection**
   - Connect to servers on first use
   - Reduce startup time

### Long-Term: HTTP Transport and Discovery

1. **Streamable HTTP Support**
   - Add `HTTPClient` alongside `StdioClient`
   - Support `"type": "http"` in config

2. **Server Discovery**
   - Query server capabilities on connect
   - Enable/disable features based on capabilities

3. **Authentication**
   - Bearer tokens for HTTP servers
   - OAuth integration for cloud services

---

## Part 5: Implementation Priority

### Phase 1: Configuration (1 session)
- [ ] Define config file schema
- [ ] Add config loader to daemon
- [ ] Migrate hardcoded server settings to config

### Phase 2: ServerRegistry (1-2 sessions)
- [ ] Create `ServerRegistry` class
- [ ] Support multiple simultaneous connections
- [ ] Add tool namespacing

### Phase 3: Resilience (1 session)
- [ ] Add reconnection logic
- [ ] Implement circuit breaker pattern
- [ ] Add health checks

### Phase 4: HTTP Transport (2 sessions)
- [ ] Add `StreamableHTTPClient`
- [ ] Support `"type": "http"` config
- [ ] Authentication support

---

---

## Part 6: Hermes as MCP Gateway

Based on discussion with Willow, there's a preferred architecture for external MCP integration: **Hermes connects to external MCP servers and routes tools through to Psyche**.

### Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Psyche Server                        │
│  ┌─────────────┐    ┌───────────────┐    ┌───────────────┐ │
│  │ PsycheCore  │───>│ Elpis MCP     │───>│ LLM Inference │ │
│  │             │    │ Client        │    │               │ │
│  │             │───>│ Mnemosyne MCP │───>│ ChromaDB      │ │
│  │             │    │ Client        │    │               │ │
│  └─────────────┘    └───────────────┘    └───────────────┘ │
└──────────────────────────▲──────────────────────────────────┘
                           │ HTTP
┌──────────────────────────│──────────────────────────────────┐
│                     Hermes TUI                              │
│  ┌───────────────┐   ┌───────────────┐                     │
│  │ RemotePsyche  │   │  ToolEngine   │                     │
│  │ Client        │   │  (local tools)│                     │
│  └───────────────┘   └───────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### Proposed Architecture: Hermes as MCP Gateway

```
┌─────────────────────────────────────────────────────────────┐
│                         Psyche Server                        │
│  ┌─────────────┐    ┌───────────────┐    ┌───────────────┐ │
│  │ PsycheCore  │───>│ Elpis MCP     │───>│ LLM Inference │ │
│  │             │    │ Client        │    │               │ │
│  │             │───>│ Mnemosyne MCP │───>│ ChromaDB      │ │
│  │             │    │ Client        │    │               │ │
│  └─────────────┘    └───────────────┘    └───────────────┘ │
└──────────────────────────▲──────────────────────────────────┘
                           │ HTTP (tools defined by Hermes)
┌──────────────────────────│──────────────────────────────────┐
│                     Hermes TUI                              │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐ │
│  │ RemotePsyche  │   │  ToolEngine   │   │ MCPRegistry   │ │
│  │ Client        │   │  (local tools)│   │ (ext servers) │ │
│  └───────────────┘   └───────────────┘   └───────────────┘ │
│                              │                    │         │
│                              ▼                    ▼         │
│                      ┌───────────────┐   ┌───────────────┐ │
│                      │ bash, file,   │   │ github, fs,   │ │
│                      │ search        │   │ browser, etc  │ │
│                      └───────────────┘   └───────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

1. **Client-Side Tool Execution**
   - Tools already execute on the client (Hermes) side
   - External MCP tools fit naturally into this model
   - Psyche server stays lightweight

2. **Tool Configuration Per-Client**
   - Different Hermes instances can have different MCP servers
   - Work laptop has GitHub MCP, personal has different tools
   - No server restart needed to add new tools

3. **Security**
   - MCP servers run in Hermes's security context
   - No need to expose external tools through the server
   - Client controls what tools are available

4. **Already Supported Pattern**
   - `ToolEngine.register_tool()` exists for runtime tool registration
   - `RemotePsycheClient.set_tools()` sends tool definitions to server
   - Just need to wire up MCP tool discovery

### Implementation Approach

1. **Add `MCPServerRegistry` to Hermes**
   ```python
   # src/hermes/mcp/registry.py
   class MCPServerRegistry:
       """Manage connections to external MCP servers."""

       servers: Dict[str, MCPClient]

       async def connect_all(self, config: Dict) -> None: ...
       async def discover_tools(self) -> List[ToolDefinition]: ...
       async def execute_tool(self, server: str, name: str, args: Dict) -> Any: ...
   ```

2. **Integrate with ToolEngine**
   ```python
   # At startup
   mcp_registry = MCPServerRegistry()
   await mcp_registry.connect_all(config.mcp_servers)

   # Register MCP tools with ToolEngine
   for tool in await mcp_registry.discover_tools():
       tool_engine.register_tool(tool)
   ```

3. **Route Tool Calls**
   - Local tools (bash, file, search) → ToolEngine handles directly
   - MCP tools (namespaced: `github.list_repos`) → Route to MCPServerRegistry

4. **Configuration File**
   ```json
   // ~/.hermes.json
   {
     "mcpServers": {
       "github": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-github"]
       },
       "filesystem": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
       }
     }
   }
   ```

### Session Estimate

| Task | Sessions |
|------|----------|
| MCPServerRegistry class | 1 |
| Integration with ToolEngine | 0.5 |
| Configuration file support | 0.5 |
| Tool namespacing & routing | 0.5 |
| Testing & edge cases | 0.5 |
| **Total** | **3 sessions** |

---

## Appendix: Configuration Schema

Proposed JSON Schema for `~/.psyche.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "mcpServers": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "command": { "type": "string" },
          "args": { "type": "array", "items": { "type": "string" } },
          "env": { "type": "object", "additionalProperties": { "type": "string" } },
          "type": { "enum": ["stdio", "http"], "default": "stdio" },
          "url": { "type": "string", "format": "uri" },
          "required": { "type": "boolean", "default": false },
          "timeout": { "type": "integer", "minimum": 1000 }
        },
        "required": ["command"]
      }
    }
  }
}
```

---

## Sources

### Agent Reports
- [Codebase Agent Report](./codebase-agent-report.md) - Internal architecture review
- [Research Agent Report](./research-agent-report.md) - External patterns research

### External References
- MCP Specification: https://modelcontextprotocol.io/specification/2025-11-25
- Python MCP SDK: https://github.com/modelcontextprotocol/python-sdk
- Claude Code MCP Docs: https://code.claude.com/docs/en/mcp
