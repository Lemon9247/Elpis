# Modern Agentic Coding Frameworks - Architectural Patterns Research

**Date**: 2026-01-13
**Purpose**: Inform Elpis Phase 2 architecture design

## Overview

Research on OpenCode, LangGraph, AutoGen, CrewAI, Google's ADK, Microsoft Agent Framework, and industry standards to identify key architectural patterns for modern agent frameworks.

## 1. Client/Server Architecture Separation

### OpenCode Pattern (Most Relevant to Elpis)
- Go TUI frontend communicates with JavaScript backend via HTTP/Hono server
- Backend exposes functionality through REST endpoints with Server-Sent Events for streaming
- Enables provider-agnostic inference (OpenAI, Anthropic, Google, AWS Bedrock, local models)
- Separates concerns: Inference (external LLM APIs), Agent Logic (Bun backend), Execution (local tools), UI (Go TUI)

**Applicability to Elpis Phase 2**: This model directly supports the distributed architecture goal of separating the Inference Server, Memory Server, and User Input Client into distinct services.

## 2. Distributed Runtime Patterns

### AutoGen's Host-Worker Architecture
- Host service maintains connections to all worker runtimes
- Facilitates message delivery and session management
- Workers advertise supported agents so host can route messages correctly
- Uses gRPC for cross-process/cross-machine communication with protobuf schemas
- Enables graceful startup/shutdown and lifecycle management

**Key for Elpis**: The host-worker pattern maps cleanly to a Memory Server (host) coordinating between Inference Server and User Input Client (workers).

### AutoGen's Event-Driven Model
- Agents publish messages to topics that other agents subscribe to
- Enables asynchronous, decoupled communication
- Suitable for long-running autonomous agents across organizational boundaries

## 3. State Management Approaches

### LangGraph's Channel-Based State Model
- Uses Pregel/BSP algorithm for deterministic state management
- State flows through named channels with versioning
- Parallel node executions are ordered deterministically to prevent data races
- Supports checkpointing and human-in-the-loop without rerunning completed work
- Architecture is independent from developer SDKs, enabling streaming, checkpointing, and tracing as optional features

### Google ADK's Tiered Context Model
- Context is "a compiled view over a richer stateful system" (not a mutable string buffer)
- Separates storage (Session logs) from presentation (working context)
- Uses ordered processor pipeline: Selection -> Transformation -> Injection
- Enables asynchronous context compaction: summarizes older events, stores summaries, prunes raw events
- Prevents "context explosion" in multi-agent systems

### Letta's Memory Blocks Pattern
- Breaks context window into discrete, functional units (label, value, size limit, description)
- Individually stored in database with unique IDs
- Multiple agents can access same memory blocks (shared knowledge base)
- Agents can edit their own memory through specialized tools
- Supports persistent, long-term coherence across interactions

## 4. Tool Execution and Isolation Patterns

### Claude Code Sandboxing Approach
- OS-level isolation enforces filesystem and network restrictions
- Filesystem isolation: read/write to current directory and subdirectories
- Network access through proxy server with domain restrictions
- Prevents sensitive file exfiltration and system resource backdooring

### Google ADK Agent Engine Pattern
- Single persistent sandbox throughout task execution
- Sandbox state persists across all operations within a session
- Maintains context between multiple tool calls
- Process-level isolation for safety while remaining lightweight

### Production Recommendations
- Firecracker microVMs: 125ms startup, 150 microVMs/second creation rate, defense-in-depth with jailer process
- gVisor: good middle ground for Kubernetes deployments
- WebAssembly/isolate: for capability-scoped operations when shell access not needed

## 5. Inference Server Separation

### OpenCode Pattern
- External LLM APIs (provider-agnostic via AI SDK)
- Backend orchestration handles tool integration independently of model choice
- Each provider has custom system prompts
- Tool results feed back into context window automatically

### Multi-Provider Support
- 75+ LLM providers supported (OpenAI, Anthropic, Google, AWS Bedrock, Groq, Azure, OpenRouter)
- Local model support via vLLM in Docker for zero-latency localhost connections
- Provider-agnostic through standardization layer

**Implication for Elpis**: The Inference Server can be completely decoupled from agent logic, supporting multiple LLM providers and even local models.

## 6. Async User Input Handling Patterns

### Azure App Service Pattern (Production)
- API immediately returns 202 Accepted with task ID (async request-reply)
- Background worker processes Agent Framework workflow
- Client polls for status with real-time progress updates
- Durable state storage (Cosmos DB) maintains task status and results
- Handles long-running workflows (30s-minutes) that would timeout in sync patterns

### LiveKit Agents Pattern
- AgentSession container manages interactions with end users
- Entrypoint defines starting point (like request handler in web server)
- Worker coordinates job scheduling and launches agents for user sessions
- Enables real-time multimodal interactions

### Smolagents Integration Pattern
- Use `anyio.to_thread.run_sync()` to run blocking agent logic in background threads
- Keeps async event loop responsive
- Prevents single long-running inference from blocking other requests

**For Elpis**: The User Input Client should use async request-reply with polling or websockets, not blocking on inference.

## 7. Context Window Management Patterns

### Progressive Disclosure
- Large data appears as lightweight references; agents load content via tools when needed
- Memory retrieval: agents dynamically query long-term knowledge
- Scoped handoffs: multi-agent transfers restrict context visibility to essential information
- Prevents "context explosion" problem

### Context Offloading to Filesystem
- Old tool results written to files with summarization only when diminishing returns
- Cursor Agent pattern: offloads tool results and agent trajectories to filesystem
- Agents can read back into context when needed

### Plan-and-Execute Pattern
- Rather than loading entire dataset, agent plans steps first
- Each step remains manageable in context
- Example: "Analyze sales data" -> "Summarize data" -> "Extract insights" -> "Draft email"

### Structured Prompting with Memory Blocks
- System prompt + tool schemas + memory blocks form complete context
- Memory blocks are individual, purposeful units with size limits
- Each labeled for its function (e.g., 'human', 'persona', 'knowledge')

## 8. Multi-Agent Coordination Patterns

### LangGraph Patterns
- Graph-based representation: agents as nodes, connections as edges
- Shared scratchpad: all agents see intermediate steps (transparent but verbose)
- Supervisor pattern: dedicated supervisor routes work between specialists
- Nested architecture: LangGraph objects themselves as nodes (hierarchical teams)
- Advantages: explicit routing, dynamic transitions, human-in-the-loop, modularity

### CrewAI Flows and Crews Architecture
- Flows: deterministic backbone with almost no abstractions (thin decorator layer)
- Crews: intelligence layer deployed strategically at workflow steps
- Philosophy: "Structure where you need it, intelligence where it matters"
- Control always returns to Flow backbone, preventing unbounded autonomous behavior
- Enables auditability, cost control, observability

### Pipeline, Hub-and-Spoke, Hierarchical Patterns
- Sequential handoffs for simple workflows
- Central coordinator dispatching to specialists
- Nested multi-level delegation

## 9. Tool Abstraction and Standardization

### Model Context Protocol (MCP)
- Standardized "tool directory" replacing custom integrations
- Servers advertise tools with JSON Schema definitions
- Clients call tools using standard `call_tool` requests
- Build once, use everywhere across different AI platforms
- Adopted by OpenAI, integrated into ChatGPT, IDEs, coding platforms
- Enables safe, predictable interactions and reduces integration complexity

**For Elpis Phase 2**: MCP provides a standard way to expose tools that can be used by the agent, external systems, and multiple LLM providers.

## 10. Emotional Regulation and State Tracking

### Foundation Agent Framework (Academic)
- Agents decomposed into perception, cognition, action subsystems
- Cognition includes: memory, world model, emotional state, goals, reward, reasoning
- Emotional state tracked alongside goal state and reward signals
- Important for long-running agents that maintain relationships and context

### Practical Application Patterns
- BDI model (Beliefs-Desires-Intentions) for behavior modeling
- Closed-loop emotional state inference from feedback
- Dynamic planner that adjusts strategies based on emotional equilibrium
- Temporal tracking: emotions evolve with conversation history and relationship context

**Relevance to Elpis**: The "emotional regulation" aspect could be implemented as a memory block tracking conversation tone, user frustration level, task complexity, and agent uncertainty - feeding into decision-making in the Memory Server.

## 11. Key Architectural Principles Across Frameworks

1. **Separation of Concerns**: Inference (external), orchestration (agent logic), execution (tools), UI (client)
2. **Determinism Over Magic**: Explicit control flow beats unbounded autonomous behavior
3. **Persistent State**: Store interaction logs, memories, and checkpoints for long-running agents
4. **Async by Default**: Non-blocking inference with polling/websockets for user interaction
5. **Sandboxed Execution**: OS-level isolation for tool execution (filesystem, network)
6. **Protocol Standardization**: MCP-like patterns for tool abstraction across frameworks
7. **Progressive Disclosure**: Load context on-demand rather than upfront
8. **Testable Transformations**: Observable, debuggable state management instead of ad-hoc concatenation

## Recommended Architecture for Elpis Phase 2

Based on research, the target architecture should include:

### 1. Inference Server (Anthropic/OpenAI/local LLM)
- Handles LLM calls and tool invocation decisions
- Separate from agent logic
- Supports multiple providers via standardization

### 2. Memory Server (continuous loop)
- Manages state channels (tiered: working context + session log + long-term memory)
- Coordinates between Inference Server and tools
- Implements context compaction and progressive disclosure
- Stores memory blocks for persistent state
- Tracks emotional/conversational state

### 3. User Input Client (async)
- Non-blocking request-reply with task IDs
- Polling or websocket for updates
- Interrupts execution via event messages

### 4. Tool Execution Sandbox (isolated)
- Firecracker microVMs or gVisor for safety
- Persistent state within session
- Returns results to Memory Server

### 5. Tool Abstraction Layer (MCP-compatible)
- Standard tool interface for discovery and invocation
- Works across inference providers and agents

## Sources

- [How Coding Agents Actually Work: Inside OpenCode](https://cefboud.com/posts/coding-agents-internals-opencode-deepdive/)
- [Agents - OpenCode Documentation](https://opencode.ai/docs/agents/)
- [Building LangGraph: Designing an Agent Runtime from first principles](https://blog.langchain.com/building-langgraph/)
- [LangGraph Multi-Agent Workflows](https://blog.langchain.com/langgraph-multi-agent-workflows/)
- [AutoGen Distributed Agent Runtime](https://microsoft.github.io/autogen/stable//user-guide/core-user-guide/framework/distributed-agent-runtime.html)
- [CrewAI - Building Agentic Systems](https://blog.crewai.com/agentic-systems-with-crewai/)
- [Architecting efficient context-aware multi-agent framework for production (Google ADK)](https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/)
- [Memory Blocks: The Key to Agentic Context Management](https://www.letta.com/blog/memory-blocks)
- [Build Long-Running AI Agents on Azure App Service](https://azure.github.io/AppService/2025/10/21/app-service-agent-framework.html)
- [Claude Code Sandboxing](https://code.claude.com/docs/en/sandboxing)
- [A field guide to sandboxes for AI](https://www.luiscardoso.dev/blog/sandboxes-for-ai/)
- [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
- [What is Model Context Protocol (MCP)?](https://cloud.google.com/discover/what-is-model-context-protocol)
- [Async Applications with Agents (HuggingFace Smolagents)](https://huggingface.co/docs/smolagents/en/examples/async_agent)
- [LiveKit Agents Framework](https://docs.livekit.io/reference/agents-js/)
