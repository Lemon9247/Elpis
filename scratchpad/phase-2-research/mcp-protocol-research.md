# MCP (Model Context Protocol) Research

**Date**: 2026-01-13
**Purpose**: Inform Elpis Phase 2 inference server design

## Overview

The Model Context Protocol is an open standard developed by Anthropic (now donated to the Agentic AI Foundation under the Linux Foundation) that solves the "N×M integration problem" for AI applications and external tools/data sources.

## 1. What is MCP and How Does It Work?

**Core Philosophy**: MCP is inspired by the Language Server Protocol (LSP) and emphasizes:
- Security and user consent (explicit authorization required)
- Stateful connections maintaining context across interactions
- A client-server architecture where AI applications act as hosts
- Support for multiple transport mechanisms (local and remote)

## 2. MCP Architecture and Structure

### Three Key Participants

- **MCP Host**: The AI application (e.g., Claude Desktop, Claude Code, VS Code) that coordinates and manages connections
- **MCP Client**: A component within the host that maintains connections to servers and translates between host requirements and the protocol
- **MCP Server**: Standalone programs that provide context, tools, and capabilities to clients

### Two-Layer Architecture

**Data Layer** (JSON-RPC 2.0 based):
- Protocol version negotiation and capability discovery
- Lifecycle management (initialization, capability negotiation, termination)
- Server-exposed primitives (tools, resources, prompts)
- Client-exposed primitives (sampling, elicitation, logging)
- Notifications for real-time updates

**Transport Layer** (abstraction for communication):
- **STDIO (Standard Input/Output)**: For local processes, optimal for single-user scenarios, sequential processing
- **HTTP + Server-Sent Events (SSE)**: For remote connections, supports OAuth and API keys, enables true concurrency
- **Streamable HTTP**: For serverless/auto-scaling environments with stateless operation

## 3. Core Primitives: Tools, Resources, and Prompts

### Tools (Model-Controlled Capabilities)
- Executable functions that LLMs can invoke with user approval
- Defined with unique names, descriptions, and JSON Schema input specifications
- Examples: API calls, code execution, database queries, calculations
- The model decides when and how to call them based on task requirements

### Resources (Application-Controlled Data)
- File-like data sources providing context to the LLM
- Exposed via unique URIs with structured naming
- Can contain text or binary data
- Client applications must explicitly fetch them (not automatically provided)
- Think of them as "GET endpoints" for LLMs
- Examples: API responses, configuration files, knowledge bases

### Prompts (User-Controlled Interaction Templates)
- Reusable interaction templates with variables for dynamic filling
- Versioned centrally without requiring client code changes
- Explicitly invoked by users or clients (not automatic)
- Guide how users interact with available tools and resources
- Provide structured patterns for common workflows

### How They Work Together
- Prompts structure user intent
- Tools execute operations
- Resources provide data
- Creates modular interaction loops for complex workflows

## 4. Message Format and Protocol

### JSON-RPC 2.0 Based Communication

Message types include:
- **Requests**: Expect a response, contain method and optional params
- **Responses**: Include result or error with status code and message
- **Notifications**: One-way messages without expecting responses (for real-time updates)
- **Errors**: Include code, message, and optional data field

### Connection Lifecycle
1. Client connects to Server
2. Exchange protocol versions and capabilities (initialization)
3. Server advertises supported tools, resources, prompts
4. Client dynamically discovers capabilities via:
   - `tools/list` - Advertise available tools with JSON Schema contracts
   - `resources/list` - Advertise available resources
   - `prompts/list` - Advertise available prompts
5. Client calls tools with `tools/call` request
6. Notifications enable real-time capability changes

### Stateful Connections
- MCP maintains long-lived connections that preserve state
- Whether transport is STDIO, HTTP stream, or WebSocket, connection stays open
- Enables server to carry context from one request to the next
- Critical for complex workflows requiring multi-step interactions

### Multiple Clients
- Each client maintains one-to-one stateful connection with a single server
- MCP host can manage multiple clients simultaneously
- Session state must be isolated (each client has own context)
- For concurrent connections, state management can use external storage (Redis, DynamoDB)

## 5. How LLM Inference Servers Can Implement MCP

### As MCP Servers (most common)

1. **As Tool Providers**:
   - Expose inference/completion as a callable tool
   - Define JSON Schema for input prompts and sampling parameters
   - Return structured outputs with completions

2. **As Resource Providers**:
   - Expose model information as resources (capabilities, limits, pricing)
   - Provide access to model state or cached information
   - Return model documentation and configuration options

3. **With Sampling Capability** (bidirectional):
   - Servers can implement the `sampling` capability (currently experimental)
   - Allows servers to request LLM completions from the client's LLM
   - Enables recursive/agentic behaviors within server logic
   - Note: Not yet supported in Claude Desktop or Claude Code as clients

### As MCP Clients

An inference server wanting to use MCP servers:
1. Connect to external MCP servers for memory, knowledge, or tools
2. Use server's tools/resources to augment inference prompts
3. Call MCP sampling to recursively invoke LLM for complex reasoning
4. Maintain stateful connections to preserve context across requests

### Implementation Options

**Python SDKs**:
- **Official MCP Python SDK**: Core protocol implementation with `mcp.server.fastmcp` FastMCP class
- **FastMCP 2.0** (standalone, actively maintained): Simplifies development with decorators, includes enterprise auth, deployment tools, testing frameworks
- **Installation**: `pip install fastmcp<3` (pins to v2 to avoid breaking changes in v3)

**Example minimal Python MCP server**:
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Inference Server")

@mcp.tool()
def infer(prompt: str, model: str = "default") -> str:
    """Run inference on the provided prompt"""
    # Call inference backend
    return result

@mcp.resource("model://{name}")
def get_model_info(name: str) -> str:
    """Get information about a model"""
    return model_details

if __name__ == "__main__":
    mcp.run(transport="stdio")  # or "streamable-http" for remote
```

### Key Implementation Guidelines
- STDIO transport: Never write to stdout (corrupts JSON-RPC), logging goes elsewhere
- HTTP transport: Standard logging to stdout is fine
- Validate all incoming MCP messages against protocol spec to prevent attacks
- Handle connection lifecycle properly (initialization, capability negotiation, cleanup)
- For stateful inference contexts, maintain session-specific state carefully
- SDKs handle much of the complexity; focus on business logic

## 6. Application to Elpis Phase 2

### For Portable Inference Server with Emotional Regulation

1. **As MCP Server**: Expose the inference server's capabilities via MCP
   - Tool: `infer(prompt, system_prompt, emotional_context)` for regulated inference
   - Resources: Model capabilities, emotional state profiles, regulation parameters
   - Prompts: Templates for different emotional contexts or use cases

2. **With Stateful Connections**: Preserve emotional state across multiple requests
   - Connection stays open throughout a conversation or session
   - Server maintains conversation history and emotional progression
   - Sampling capability could enable recursive emotional reasoning (if implemented)

3. **Multiple Harnesses**: MCP enables this perfectly
   - Claude Desktop can connect as a client
   - Custom applications can connect as clients
   - IDE plugins can connect as clients
   - All use the same standardized interface
   - No vendor lock-in

4. **Standard API Format**: JSON-RPC 2.0 with MCP protocol
   - Language-agnostic
   - Well-documented specification
   - Strong typing via JSON Schema
   - Built-in discovery mechanism

### For Memory Server

1. **As MCP Server**: Expose memory operations
   - Tools: `create_entity()`, `add_observation()`, `link_entities()`, `query_graph()`
   - Resources: Knowledge graph snapshots, entity profiles, relationship maps
   - Prompts: Query templates for common memory patterns

2. **Integration with Inference Server**:
   - Inference server (as MCP client) connects to memory server
   - Before inference: Query memory server for relevant context
   - During inference: Reference emotional history, past patterns, learned preferences
   - After inference: Update memory with new observations and relationships
   - Enables continuous learning and emotional adaptation

3. **Existing Precedent**:
   - Anthropic's Knowledge Graph Memory MCP server demonstrates this pattern
   - Supports cross-application persistence (Claude, Cursor, Raycast)
   - Graph structure (entities, observations, relations) enables semantic retrieval
   - Can track emotional journey and relationship evolution

### Architecture Benefits for Elpis

- **Portability**: MCP clients (harnesses) can be swapped without changing inference logic
- **Interoperability**: Memory servers, inference servers, and other services communicate via standard protocol
- **Scalability**: Servers can run locally or remotely via different transports
- **Security**: Explicit authorization for all operations, user approval required
- **Extensibility**: New MCP servers can be added without modifying existing ones
- **State Management**: Stateful connections enable complex, multi-turn interactions with preserved context
- **Standardization**: Follows industry-standard protocol (JSON-RPC 2.0), inspired by LSP

### Key Considerations for Implementation

1. **Connection Lifecycle**: Properly initialize, negotiate capabilities, and terminate connections
2. **Error Handling**: Implement robust error handling with proper JSON-RPC error responses
3. **Logging Strategy**: Be mindful of stdout/logging depending on transport type
4. **Session State**: If supporting multiple concurrent clients, isolate state per session
5. **Stateful Inference**: Maintain emotional state and context properly across requests
6. **Performance**: STDIO transport is optimal for local single-user scenarios
7. **Remote Deployment**: Use HTTP+SSE for remote inference servers with external state storage
8. **Testing**: FastMCP 2.0 includes testing frameworks for easier server validation

## Sources

- [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/specification/2025-11-25)
- [Build an MCP server - Model Context Protocol](https://modelcontextprotocol.io/docs/develop/build-server)
- [Architecture overview - Model Context Protocol](https://modelcontextprotocol.io/docs/learn/architecture)
- [Model Context Protocol - Wikipedia](https://en.wikipedia.org/wiki/Model_Context_Protocol)
- [What Is the Model Context Protocol (MCP) and How It Works](https://www.descope.com/learn/post/mcp)
- [How to MCP - The Complete Guide](https://simplescraper.io/blog/how-to-mcp)
- [Model Context Protocol (MCP) Tutorial](https://towardsdatascience.com/model-context-protocol-mcp-tutorial-build-your-first-mcp-server-in-6-steps/)
- [Knowledge Graph Memory MCP Server by Anthropic](https://www.pulsemcp.com/servers/modelcontextprotocol-knowledge-graph-memory)
- [Sampling - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/client/sampling)
- [What is MCP sampling?](https://www.speakeasy.com/mcp/core-concepts/sampling)
- [LLM Sampling with FastMCP](https://www.pondhouse-data.com/blog/llm-sampling-with-fastmcp)
- [Welcome to FastMCP 2.0!](https://gofastmcp.com/)
- [GitHub - modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)
- [GitHub - jlowin/fastmcp](https://github.com/jlowin/fastmcp)
- [Composio - How to effectively use prompts, resources, and tools in MCP](https://composio.dev/blog/how-to-effectively-use-prompts-resources-and-tools-in-mcp)
- [Managing Stateful MCP Server Sessions](https://codesignal.com/learn/courses/developing-and-integrating-an-mcp-server-in-typescript/lessons/stateful-mcp-server-sessions)
- [Configure MCP Servers for Multiple Connections - Guide](https://mcpcat.io/guides/configuring-mcp-servers-multiple-simultaneous-connections/)
- [The Communication Protocol - Hugging Face MCP Course](https://huggingface.co/learn/mcp-course/en/unit1/communication-protocol)
