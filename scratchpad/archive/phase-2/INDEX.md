# Phase 2 Research Index

**Date**: 2026-01-13
**Purpose**: Research to inform Elpis Phase 2 architecture redesign

## Research Documents

1. **[letta-architecture-research.md](./letta-architecture-research.md)**
   - Letta/MemGPT memory architecture
   - Hierarchical memory (core/archival/recall)
   - Memory blocks and consolidation
   - Sleep-time compute for background processing
   - Server-centric stateful design

2. **[mcp-protocol-research.md](./mcp-protocol-research.md)**
   - Model Context Protocol specification
   - Tools, Resources, and Prompts primitives
   - JSON-RPC 2.0 communication
   - STDIO vs HTTP transports
   - Implementation with FastMCP

3. **[agent-frameworks-research.md](./agent-frameworks-research.md)**
   - OpenCode architecture patterns
   - LangGraph state management
   - AutoGen distributed runtime
   - CrewAI flows pattern
   - Google ADK context management
   - Async user input handling

## Key Takeaways

### For Inference Server (Elpis)
- Use MCP protocol for portability
- Implement valence-arousal emotional model
- Stateful connections preserve emotional state
- FastMCP provides simple Python implementation

### For Memory/Harness Project
- Adopt Letta's hierarchical memory model
- Memory blocks with size limits force consolidation
- Sleep-time compute for async consolidation
- Async user input with polling/websockets

### Architecture Principles
1. Separation of concerns (inference vs orchestration vs tools)
2. Protocol standardization (MCP)
3. Persistent state with database backing
4. Async by default
5. Progressive context disclosure
