# Letta/MemGPT Architecture Research

**Date**: 2026-01-13
**Purpose**: Inform Elpis Phase 2 memory system design

## Overview

Letta (formerly MemGPT) implements a hierarchical memory system inspired by operating system virtual memory, dividing the LLM's memory into tiers.

## 1. Long-Term Memory and Context Window Management

Letta implements a **hierarchical memory system**:

### In-Context Memory (Core Memory)
- Reserved, editable section of the LLM's context window
- Functions like an editable system prompt
- Structured memory blocks with character limits (default 2,000 characters per block)
- Prevents token bloat

### External Storage Tiers
- **Archival Memory**: Vector database-backed storage for long-running agent memories and external data sources
- **Recall Memory**: Complete conversation history searchable by date and text

**Key Innovation**: The LLM itself manages memory transitions between tiers using designated tools rather than relying on external systems. Agents actively decide what information to load into context at any given time.

## 2. Memory Consolidation and Compaction

### Memory Block Architecture
- Each memory block has a size limit (default 2,000 characters)
- Forces deliberate consolidation of information
- Agents must decide what stays in context versus what moves to external storage

### Self-Editing Memory Tools
Agents have three primary tools for memory management:
- `memory_replace`: Search and replace for precise edits
- `memory_insert`: Add new information
- `memory_rethink`: Rewrite and consolidate memory sections

### Sleep-Time Compute
A background process paradigm where specialized "sleep-time agents" run asynchronously during idle periods to:
- Identify contradictions in stored memories
- Abstract patterns from specific experiences
- Pre-compute associations that speed up future reasoning
- Consolidate important information without blocking user-facing responses

This solves a key limitation: unlike MemGPT where memory operations could slow responses, Letta handles consolidation asynchronously.

## 3. Architecture for Continuous Inference and Memory Storage

### Server-Centric, Stateful Architecture
Designed for perpetual agent operation:

**Database-Backed Agent State**:
- Agent state is checkpointed and persisted to a database at each agent step
- Unlike stateless APIs (e.g., ChatCompletions), Letta maintains server-side state for:
  - Agent configuration
  - Memory blocks
  - Complete message history
  - Tool definitions

**Perpetual Agent Model**:
- No sessions or threads
- Agents maintain a single perpetual message history
- Enables continuous learning and reasoning across unlimited interactions

**Message Buffer + Memory Stack**:
- Recent conversation messages stay in a message buffer for immediate context
- Core memory blocks are pinned to the context window
- Historical context can be summarized and stored in archival/recall memory

**Token-Space Learning**:
- Rather than fine-tuning weights, agents maintain and refine learned memories stored as tokens
- System prompts, tool definitions, context history transfer across model generations
- Allows infinite-horizon learning without catastrophic forgetting

## 4. The "Nap" Concept for Memory Consolidation

### Background Processing
Between active user sessions, dedicated agents can run memory consolidation tasks asynchronously.

### Memory Refinement Activities
- Identifying and resolving contradictions in stored memories
- Abstracting patterns from specific experiences into generalizable knowledge
- Pre-computing associations for faster future reasoning
- Recursive summarization of older conversation messages

### Non-Blocking Design
Unlike MemGPT where memory operations could delay user responses, sleep-time compute happens in the background, maintaining quick response times while enabling sophisticated memory management.

### Human Memory Analogy
Inspired by human sleep consolidation, agents restructure and refine learned context when not actively processing user requests.

## 5. API Server Design Patterns

### Stateful Server Pattern
- Centralized server (deployable locally or self-hosted) is the single source of truth
- All agent state and memory persists server-side, not in client applications
- Clients only send messages; server handles all state management

### Multi-Interface Consistency
- REST API, Python SDK, TypeScript SDK, and visual ADE all connect to the same underlying agent instance
- Developers can switch between programmatic and visual interfaces without changing agent implementation
- All interfaces use the same internal API, ensuring consistency

### Agent-as-Service Pattern
- Agents are persistent services, not ephemeral library objects
- Agents continue existing when client applications stop
- Multiple clients can connect to the same agent instance
- Enables patterns like "one agent per user" for personalization

### Memory Block Isolation Pattern
- Each agent has isolated memory block instances
- Multiple agents can optionally share read-only blocks
- Blocks are individually stored in the database with unique identifiers
- Enables multi-agent collaboration while maintaining isolation

### Database Persistence Model
- At each agent reasoning step, the full `AgentState` object is checkpointed to the database
- This includes memory blocks, message history, tool definitions, and execution state
- Enables agent resumption and recovery across server restarts

## Key Takeaways for Elpis Phase 2

1. **Hierarchical Memory Over Naive Appending**: Instead of appending until context overflows, actively manage information tiers with size constraints forcing consolidation

2. **Agent-Driven Memory Management**: Agents use tools to control their own memory rather than passive RAG retrieval

3. **Asynchronous Consolidation**: Sleep-time compute decouples memory operations from user-facing inference

4. **Token-Space Over Weight-Space Learning**: Learning happens by updating context tokens, not model weights - portable across model generations

5. **Server-Side Statefulness**: All state lives server-side with database persistence - enables multi-client access and recovery

6. **Perpetual Agents**: Single continuous message history rather than sessions - naturally supports long-term memory

## Sources

- [MemGPT | Letta Docs](https://docs.letta.com/concepts/memgpt/)
- [Memory Blocks: The Key to Agentic Context Management | Letta](https://www.letta.com/blog/memory-blocks)
- [Understanding memory management | Letta Docs](https://docs.letta.com/advanced/memory-management/)
- [Memory overview | Letta Docs](https://docs.letta.com/guides/agents/memory/)
- [Agent Memory: How to Build Agents that Learn and Remember | Letta](https://www.letta.com/blog/agent-memory)
- [Continual Learning in Token Space | Letta](https://www.letta.com/blog/continual-learning)
- [Core concepts | Letta Docs](https://docs.letta.com/core-concepts/)
- [The Letta API | Letta Docs](https://docs.letta.com/api-reference/overview/)
- [Building stateful agents | Letta Docs](https://docs.letta.com/guides/agents/overview)
- [Letta Code Memory Architecture Deep Dive - GitHub Gist](https://gist.github.com/monotykamary/89226f685c17841d4d910c30b6b88442)
- [GitHub - letta-ai/letta](https://github.com/letta-ai/letta)
