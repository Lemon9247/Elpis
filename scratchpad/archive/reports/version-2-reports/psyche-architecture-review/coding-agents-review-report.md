# Coding Agents Architecture Review Report

**Agent**: Coding Agents Review Agent
**Date**: 2026-01-16
**Status**: Complete

---

## Executive Summary

This report reviews three coding agent frameworks - OpenCode, Crush, and Letta - to inform Psyche's architecture improvements. Key findings:

1. **OpenCode** demonstrates excellent multi-backend support and clean UI patterns through provider abstraction
2. **Crush** shows how to build lightweight, focused CLI tools with streamlined tool display
3. **Letta** provides the most sophisticated memory and state management architecture

The primary recommendations for Psyche are:
- Adopt a provider abstraction layer for LLM backends
- Implement clean tool display UI (action summaries instead of raw JSON)
- Consider Letta-style memory blocks for context management
- Use MCP as the standard tool interface

---

## 1. OpenCode Architecture Review

### Overview

OpenCode is a Go-based TUI coding agent that communicates with a JavaScript/Bun backend via HTTP. It's designed for provider-agnostic LLM integration.

### Architecture Approach

```
+------------------+       HTTP/SSE       +------------------+
|   Go TUI Client  |  <--------------->   |  Bun/JS Backend  |
|  (Bubble Tea)    |                      |  (Hono Server)   |
+------------------+                      +------------------+
                                                   |
                                                   v
                                          +------------------+
                                          |   AI SDK Layer   |
                                          | (Provider Agnostic)|
                                          +------------------+
                                                   |
                    +------------------------------+------------------------------+
                    |              |               |              |               |
                    v              v               v              v               v
               +--------+    +--------+      +--------+    +--------+      +--------+
               | OpenAI |    |Anthropic|     | Google |    | Ollama |      |  vLLM  |
               +--------+    +--------+      +--------+    +--------+      +--------+
```

### LLM Backend Handling

**Provider Abstraction via Vercel AI SDK**:
- Uses `@ai-sdk/` packages for each provider (openai, anthropic, google, etc.)
- Single unified interface regardless of backend
- Supports 75+ LLM providers including local models via Ollama and vLLM
- Custom system prompts per provider to handle quirks

**Configuration Pattern**:
```typescript
// Provider configuration is runtime-switchable
const provider = getProvider(config.provider);  // "openai", "anthropic", "ollama", etc.
const model = provider.model(config.model);

// Unified streaming interface
const stream = await streamText({
  model,
  messages,
  tools: registeredTools,
  onChunk: handleChunk,
});
```

### Tool Calling and Display

**Tool Definition**:
- Uses OpenAI-compatible function calling format
- Tools defined with JSON Schema parameters
- Model decides when to invoke tools based on context

**Clean Tool UI Pattern**:
```
Instead of:
{"name": "edit_file", "arguments": {"path": "/foo/bar.ts", "content": "..."}}

OpenCode displays:
Editing /foo/bar.ts
  + Added: 5 lines
  - Removed: 2 lines
```

**Implementation Approach**:
1. Tool execution happens in backend
2. Backend emits structured events via SSE
3. TUI renders human-friendly summaries based on event type
4. Full details available in expandable/debug mode

### Streaming

- Server-Sent Events (SSE) for server-to-client streaming
- Token-by-token delivery for responsive UX
- Structured event types: `text`, `tool_call`, `tool_result`, `error`
- Automatic reconnection handled by SSE protocol

### MCP/Plugin System

- Tools registered via plugin system
- Each tool is a module with schema + handler
- No native MCP integration (uses custom tool protocol)
- Plugin discovery at startup

### Pros for Psyche

1. **Clean separation of concerns**: UI completely separate from inference
2. **Provider agnostic**: Easy to swap LLM backends
3. **Mature UI patterns**: Good reference for tool display
4. **SSE streaming**: Proven pattern that Psyche already uses

### Cons for Psyche

1. **Two-language architecture**: Go + JS adds complexity
2. **No MCP support**: Would need custom integration
3. **Stateless backend**: Limited memory/context management
4. **Heavyweight dependencies**: Requires Bun runtime

---

## 2. Crush Architecture Review

### Overview

Crush is a lightweight CLI coding agent focused on simplicity and speed. It prioritizes developer experience with minimal configuration.

### Architecture Approach

```
+------------------+
|   CLI Interface  |
|   (Rust/Go)      |
+------------------+
         |
         v
+------------------+
|  Agent Runtime   |
|  - Tool Registry |
|  - Context Mgmt  |
+------------------+
         |
         v
+------------------+
|   LLM Client     |
| (Direct API)     |
+------------------+
```

### LLM Backend Handling

**Direct Provider Integration**:
- Primarily targets OpenAI and Anthropic APIs
- Configuration via environment variables and config files
- Supports model switching via CLI flags

**Local Model Support**:
- Ollama integration for local inference
- OpenAI-compatible API endpoints for custom servers
- Limited compared to OpenCode's breadth

**Configuration**:
```yaml
# .crush.yaml
provider: anthropic
model: claude-sonnet-4-20250514
api_key: ${ANTHROPIC_API_KEY}

# Alternative
provider: ollama
model: llama3.1:70b
base_url: http://localhost:11434
```

### Tool Calling and Display

**Streamlined Tool Display**:
Crush excels at presenting tool actions clearly:

```
Reading file: src/main.py
  Lines: 1-50

Writing file: src/utils.py
  Created new file (23 lines)

Running command: pytest tests/
  Exit code: 0
  Duration: 1.2s
```

**Action-Based UI Pattern**:
1. Each tool has a "display name" and "action verb"
2. Arguments are rendered contextually (paths shortened, content summarized)
3. Results shown inline with collapsible details
4. Progress indicators for long-running operations

### Streaming

- Streaming text output as tokens arrive
- Tool calls shown as "thinking" state
- Results appear when tool execution completes
- Interruptible via Ctrl+C (graceful cancellation)

### MCP/Plugin System

- Limited plugin architecture
- Core tools built-in (file operations, shell, search)
- No external tool discovery
- No MCP support

### Pros for Psyche

1. **Excellent tool display patterns**: Clear, contextual summaries
2. **Lightweight**: Fast startup, low resource usage
3. **Interruptible**: Graceful cancellation of operations
4. **Simple configuration**: Easy to understand and modify

### Cons for Psyche

1. **Limited backend support**: Few providers compared to OpenCode
2. **No memory system**: Ephemeral context only
3. **No MCP**: Custom tool interface
4. **Less extensible**: Focused on core use case

---

## 3. Letta Architecture Review

### Overview

Letta (formerly MemGPT) is a sophisticated agent framework focused on long-term memory, stateful agents, and multi-session persistence.

### Architecture Approach

```
+------------------+     REST/WS      +------------------+
|    Client SDK    | <------------->  |   Letta Server   |
| (Python/TS/ADE)  |                  | (Agent Runtime)  |
+------------------+                  +------------------+
                                              |
                    +-------------------------+-------------------------+
                    |                         |                         |
                    v                         v                         v
            +---------------+        +----------------+         +---------------+
            | Memory System |        | Tool Registry  |         | LLM Providers |
            | - Core Memory |        | - Built-in     |         | - OpenAI      |
            | - Archival    |        | - Custom       |         | - Anthropic   |
            | - Recall      |        | - MCP          |         | - Local       |
            +---------------+        +----------------+         +---------------+
                    |
                    v
            +---------------+
            |   Database    |
            | (PostgreSQL/  |
            |  SQLite)      |
            +---------------+
```

### LLM Backend Handling

**Provider Abstraction Layer**:
- Native support for OpenAI, Anthropic, Google, Cohere
- Local model support via Ollama, vLLM, llama.cpp
- Unified interface through provider adapters

**Configuration**:
```python
from letta import LettaClient

# Cloud provider
client = LettaClient(
    provider="anthropic",
    model="claude-sonnet-4-20250514"
)

# Local model
client = LettaClient(
    provider="ollama",
    model="llama3.1:8b",
    base_url="http://localhost:11434"
)
```

**Key Features**:
- Automatic function calling format conversion per provider
- Model-specific system prompt templates
- Token counting and context management per model

### Tool Calling and Display

**Tool Architecture**:
- Tools defined as Python functions with decorators
- JSON Schema auto-generated from type hints
- Built-in tools for memory management
- Custom tool registration API

**Memory Management Tools** (unique to Letta):
```python
# Agent has tools to manage its own memory
@tool
def core_memory_append(self, label: str, content: str):
    """Add to a core memory block"""

@tool
def core_memory_replace(self, label: str, old_content: str, new_content: str):
    """Replace content in core memory"""

@tool
def archival_memory_insert(self, content: str):
    """Store in long-term archival memory"""

@tool
def archival_memory_search(self, query: str, page: int = 0):
    """Search archival memory"""
```

**Tool Display**:
- Uses structured logging for tool invocations
- Agent Development Environment (ADE) provides visual tool execution
- Streaming shows tool calls as structured events

### Streaming

**Multi-Channel Streaming**:
- WebSocket connections for real-time updates
- Separate channels for: text, tool_calls, tool_results, memory_updates
- Event-driven architecture with async handlers

```python
async for event in client.send_message_async(agent_id, message):
    if event.type == "text":
        print(event.content, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\n[Tool: {event.tool_name}]")
    elif event.type == "memory_update":
        print(f"\n[Memory: {event.block_label} updated]")
```

### MCP/Plugin System

**Native MCP Support**:
- Letta 2.0+ includes MCP client capabilities
- Can connect to external MCP servers for tools
- Tool discovery via MCP protocol

**Tool Registration**:
```python
# Register MCP server
client.add_mcp_server(
    name="github",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"]
)

# Tools from MCP server become available to agent
agent = client.create_agent(
    tools=["mcp:github:*"]  # All tools from github MCP server
)
```

### Memory System (Unique Differentiator)

**Hierarchical Memory Architecture**:

1. **Core Memory** (In-Context):
   - Fixed-size memory blocks in the context window
   - Agent-editable via memory tools
   - Examples: `human` (user info), `persona` (agent personality)
   - Default 2,000 characters per block

2. **Archival Memory** (Vector DB):
   - Long-term storage backed by vector database
   - Searchable via semantic similarity
   - Unlimited capacity
   - Used for: facts, knowledge, historical data

3. **Recall Memory** (Conversation History):
   - Complete message history
   - Searchable by date and content
   - Supports conversation compaction

**Memory Block Pattern**:
```python
# Memory blocks are structured units
memory_block = {
    "label": "human",
    "value": "Name: Alice\nPreferences: Detailed explanations, code examples",
    "limit": 2000,  # Character limit
    "description": "Information about the user"
}

# Agent sees this in context and can edit it
```

**Sleep-Time Compute** (Background Processing):
- Agents can consolidate memories during idle time
- Identifies contradictions in stored memories
- Abstracts patterns from specific experiences
- Pre-computes associations for faster reasoning

### Pros for Psyche

1. **Sophisticated memory system**: Most advanced of the three
2. **MCP support**: Standard tool interface
3. **Perpetual agents**: Long-running stateful agents
4. **Multi-provider support**: Good backend flexibility
5. **Server-side state**: Database-backed persistence

### Cons for Psyche

1. **Complexity**: Heavy framework, steep learning curve
2. **Server-centric**: Requires running Letta server
3. **Overhead**: More resources than simpler agents
4. **Learning curve**: Complex concepts to master

---

## 4. Comparison Table

| Feature | OpenCode | Crush | Letta |
|---------|----------|-------|-------|
| **Primary Language** | Go + JS | Rust/Go | Python |
| **UI Type** | TUI (Bubble Tea) | CLI | CLI + Web ADE |
| **Architecture** | Client-Server | Monolithic | Client-Server |
| **LLM Providers** | 75+ via AI SDK | ~5 direct | 10+ via adapters |
| **Local Model Support** | Ollama, vLLM | Ollama | Ollama, vLLM, llama.cpp |
| **Tool Definition** | JSON Schema | JSON Schema | Python decorators |
| **Tool Display** | Clean summaries | Clean summaries | Structured events |
| **MCP Support** | No | No | Yes (native) |
| **Memory System** | None | None | Hierarchical (Core/Archival/Recall) |
| **Streaming** | SSE | Direct | WebSocket |
| **State Persistence** | None | None | Database-backed |
| **Context Management** | Basic | Basic | Advanced (compaction, blocks) |
| **Multi-Agent** | No | No | Yes |
| **Complexity** | Medium | Low | High |

---

## 5. Key Patterns to Adopt

### 5.1 Provider Abstraction Layer

**Pattern**: Create a unified interface for all LLM backends

```python
# Recommended pattern for Psyche
from abc import ABC, abstractmethod
from typing import AsyncIterator

class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, messages: list, tools: list = None) -> AsyncIterator:
        """Stream completion events from the LLM"""
        pass

    @abstractmethod
    def get_tool_format(self) -> str:
        """Return 'openai', 'anthropic', 'llama' for tool call formatting"""
        pass

    @abstractmethod
    def format_tools(self, tools: list) -> list:
        """Convert tools to provider-specific format"""
        pass

class AnthropicProvider(LLMProvider):
    async def complete(self, messages, tools=None):
        # Anthropic-specific implementation
        pass

class OllamaProvider(LLMProvider):
    async def complete(self, messages, tools=None):
        # Ollama-specific implementation
        pass

class ElpisProvider(LLMProvider):
    """Our local Elpis inference server via MCP"""
    async def complete(self, messages, tools=None):
        # Connect to Elpis MCP server
        pass
```

**Benefits**:
- Swap backends without changing agent code
- Test with cheap/fast models, deploy with powerful ones
- Support both cloud and local inference
- Handle provider-specific quirks in one place

### 5.2 Clean Tool Display

**Pattern**: Transform raw tool calls into human-readable summaries

```python
class ToolDisplayFormatter:
    """Format tool calls for human-readable display"""

    DISPLAY_TEMPLATES = {
        "read_file": "Reading {path}",
        "write_file": "Writing {path} ({lines} lines)",
        "edit_file": "Editing {path}",
        "execute_bash": "Running: {command}",
        "search": "Searching for: {query}",
        "memory_store": "Storing memory: {key}",
        "memory_recall": "Recalling: {query}",
    }

    RESULT_TEMPLATES = {
        "read_file": "  {line_count} lines read",
        "write_file": "  File saved ({bytes} bytes)",
        "execute_bash": "  Exit code: {returncode}",
        "search": "  Found {count} results",
    }

    def format_tool_call(self, tool_name: str, args: dict) -> str:
        template = self.DISPLAY_TEMPLATES.get(tool_name, f"Using {tool_name}")
        try:
            return template.format(**args)
        except KeyError:
            return f"Using {tool_name}"

    def format_tool_result(self, tool_name: str, result: dict) -> str:
        template = self.RESULT_TEMPLATES.get(tool_name)
        if template:
            try:
                return template.format(**result)
            except KeyError:
                pass
        return ""
```

**UI Implementation Example**:
```
User: Read the config file and update the port
Assistant: Let me read the config file first...
[Reading config.yaml]
  42 lines read

I found the port setting. Let me update it...
[Editing config.yaml]
  Changed port from 8080 to 3000

Done! The port has been updated.
```

### 5.3 MCP as Tool Interface Standard

**Pattern**: Use MCP protocol for all tool communication

**Benefits**:
- Standard interface works across different agents
- Tools discoverable at runtime
- No custom protocol maintenance
- Works with external MCP servers

**Implementation for Psyche**:
```python
# Psyche should connect to Mnemosyne (memory) via MCP
# Psyche should connect to Elpis (inference) via MCP
# External tools should also use MCP

class MCPToolRegistry:
    """Manage tools from multiple MCP servers"""

    def __init__(self):
        self.servers = {}  # name -> MCPClient

    async def add_server(self, name: str, transport: str, config: dict):
        """Add an MCP server"""
        client = MCPClient(transport, config)
        await client.initialize()
        self.servers[name] = client

    def list_tools(self) -> list:
        """Get all available tools from all servers"""
        tools = []
        for name, client in self.servers.items():
            for tool in client.list_tools():
                tools.append({
                    "server": name,
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.input_schema
                })
        return tools

    async def call_tool(self, server: str, tool: str, args: dict):
        """Call a tool on a specific server"""
        return await self.servers[server].call_tool(tool, args)
```

### 5.4 Letta-Style Memory Blocks

**Pattern**: Structure context as discrete, editable memory blocks

**Implementation for Psyche**:
```python
@dataclass
class MemoryBlock:
    label: str
    value: str
    limit: int = 2000  # Character limit
    description: str = ""

class ContextManager:
    """Manage memory blocks in context window"""

    def __init__(self):
        self.blocks = {
            "system": MemoryBlock(
                label="system",
                value="You are Psyche, an emotional coding assistant...",
                limit=1000,
                description="Core system personality"
            ),
            "human": MemoryBlock(
                label="human",
                value="",
                limit=2000,
                description="Information about the current user"
            ),
            "emotional_state": MemoryBlock(
                label="emotional_state",
                value="valence: 0.0, arousal: 0.0",
                limit=500,
                description="Current emotional state"
            ),
            "working_memory": MemoryBlock(
                label="working_memory",
                value="",
                limit=4000,
                description="Task-relevant context"
            )
        }

    def update_block(self, label: str, value: str):
        """Update a memory block, enforcing limits"""
        if label not in self.blocks:
            raise KeyError(f"Unknown block: {label}")
        block = self.blocks[label]
        if len(value) > block.limit:
            raise ValueError(f"Content exceeds {block.limit} char limit")
        block.value = value

    def get_context(self) -> str:
        """Assemble context window from blocks"""
        context_parts = []
        for block in self.blocks.values():
            if block.value:
                context_parts.append(f"[{block.label}]\n{block.value}")
        return "\n\n".join(context_parts)
```

---

## 6. Anti-Patterns to Avoid

### 6.1 Raw JSON Tool Display

**Anti-Pattern**: Showing raw tool call JSON to users
```
{"name": "write_file", "arguments": {"path": "/home/user/file.txt", "content": "..."}}
```

**Problem**: Poor UX, information overload, technical debt

**Solution**: Always transform to human-readable summary

### 6.2 Tight Backend Coupling

**Anti-Pattern**: Hardcoding LLM provider calls throughout the codebase
```python
# Bad: Direct API calls scattered everywhere
response = openai.chat.completions.create(...)
```

**Problem**: Difficult to switch providers, test, or add new backends

**Solution**: Use provider abstraction layer

### 6.3 Unbounded Context Growth

**Anti-Pattern**: Appending to context indefinitely until overflow
```python
# Bad: Just keep adding
context += new_message
```

**Problem**: Eventually hits context limit, no control over what's kept

**Solution**: Use memory blocks with explicit limits, implement compaction

### 6.4 Custom Tool Protocols

**Anti-Pattern**: Inventing a new tool calling format
```python
# Bad: Custom format
{"tool": "my_tool", "params": {...}}
```

**Problem**: Incompatible with ecosystem, maintenance burden

**Solution**: Use OpenAI-compatible or MCP standard

### 6.5 Blocking Tool Execution

**Anti-Pattern**: Running tools synchronously without feedback
```python
# Bad: User sees nothing during long operation
result = execute_long_running_tool()
```

**Problem**: Poor UX, no ability to cancel

**Solution**: Stream progress, support cancellation

---

## 7. Specific Recommendations for Psyche

### 7.1 Priority 1: Clean Tool Display

**Current Issue**: Tool JSON dumped to chat window
**Recommendation**: Implement ToolDisplayFormatter
**Effort**: Low (few days)
**Impact**: High (immediate UX improvement)

### 7.2 Priority 2: Provider Abstraction

**Current Issue**: Tight coupling to Elpis inference
**Recommendation**: Create LLMProvider interface
**Effort**: Medium (1-2 weeks)
**Impact**: High (enables Ollama, cloud providers)

### 7.3 Priority 3: Interruption Support

**Current Issue**: Cannot interrupt LLM or tool execution
**Recommendation**: Add cancellation tokens, async patterns
**Effort**: Medium (1-2 weeks)
**Impact**: High (critical for UX)

### 7.4 Priority 4: MCP Tool Standardization

**Current Issue**: Psyche has internal tools, Mnemosyne exposes MCP
**Recommendation**: All tools via MCP, including internal ones
**Effort**: Medium (2-3 weeks)
**Impact**: Medium (better modularity)

### 7.5 Priority 5: Memory Block System

**Current Issue**: Flawed compaction and storage
**Recommendation**: Adopt Letta-style memory blocks
**Effort**: High (3-4 weeks)
**Impact**: High (enables long-term memory)

---

## 8. Proposed Architecture

Based on this review, here is a proposed target architecture for Psyche:

```
+------------------+
|   Psyche TUI     |
| (Textual Python) |
+--------+---------+
         |
         | Tool Display Layer
         | (human-readable summaries)
         |
+--------v---------+
|  Agent Runtime   |
| - Context Manager|
| - Memory Blocks  |
| - Tool Registry  |
+--------+---------+
         |
         | Provider Abstraction Layer
         |
+--------v---------+
|  LLM Provider    |
| - Elpis (local)  |
| - Ollama         |
| - Anthropic      |
| - OpenAI         |
+--------+---------+
         |
         | MCP Protocol
         |
+--------v---------+        +------------------+
|  Tool Servers    | <----> |   Mnemosyne      |
| - File Ops       |        |  (Memory MCP)    |
| - Shell          |        +------------------+
| - Search         |
+------------------+        +------------------+
                    <----> |    Elpis         |
                           | (Inference MCP)  |
                           +------------------+
```

**Key Properties**:
1. **UI Layer**: Only handles display, receives structured events
2. **Agent Runtime**: Core logic, context management, tool orchestration
3. **Provider Layer**: Abstracts LLM backends, handles format conversion
4. **Tool Layer**: All tools via MCP, including internal and external

---

## 9. Conclusion

This review of OpenCode, Crush, and Letta reveals clear patterns for improving Psyche:

1. **OpenCode** shows us how to build provider-agnostic systems with clean UI
2. **Crush** demonstrates streamlined tool display and lightweight design
3. **Letta** provides a roadmap for sophisticated memory management

The key takeaways are:
- Use abstraction layers for LLM backends
- Transform tool calls into human-readable summaries
- Adopt structured memory blocks instead of naive appending
- Standardize on MCP for tool communication

By implementing these patterns, Psyche can evolve from a working prototype into a modular, extensible, and user-friendly coding agent.

---

## References

### Existing Project Research
- `/home/lemoneater/Projects/Personal/Elpis/scratchpad/archive/phase-2/agent-frameworks-research.md`
- `/home/lemoneater/Projects/Personal/Elpis/scratchpad/archive/phase-2/letta-architecture-research.md`
- `/home/lemoneater/Projects/Personal/Elpis/scratchpad/archive/phase-2/mcp-protocol-research.md`
- `/home/lemoneater/Projects/Personal/Elpis/scratchpad/archive/phase-2/streaming-research.md`
- `/home/lemoneater/Projects/Personal/Elpis/scratchpad/archive/initial-research/tool-system-report.md`

### External Documentation
- OpenCode: https://opencode.ai/docs/
- Letta: https://docs.letta.com/
- MCP Specification: https://modelcontextprotocol.io/specification/

---

**Report Status**: Complete
**Agent**: Coding Agents Review Agent
**Date**: 2026-01-16
