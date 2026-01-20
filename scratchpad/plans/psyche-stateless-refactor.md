# Psyche Stateless Refactoring - Discussion Summary

## Context

Psyche should appear stateless to clients. Currently, idle handling and client orchestration live in Psyche but are actually client-side concerns.

## Current Architecture

```
src/psyche/handlers/
├── react_handler.py    # ReAct loop - client orchestration
├── idle_handler.py     # Idle thinking - client behavior (workspace exploration)
├── dream_handler.py    # Dreaming - server behavior (memory processing)
└── psyche_client.py    # Client interface abstraction
```

**Problem**: ReactHandler, IdleHandler, and PsycheClient are client concerns living in Psyche.

## Research: How Other Harnesses Handle This

### OpenCode (https://opencode.ai/docs/server/)
- Clear client/server split
- **Server**: Sessions (stateful containers), tool execution, provider communication
- **Client (TUI)**: UI, user input, event subscription
- Sessions are the stateful unit - API is stateless per-request
- Multiple clients can connect to same session

### Claude Code
- "Stateless architecture" - commands execute independently
- State persisted in files (JSON, Markdown)
- Can be both MCP server and client
- Uses subagents for task delegation

### Aider
- More monolithic - runs locally
- MCP server mode for external integration

## Key Insight from OpenCode

Sessions are the stateful container. The server doesn't assume state between requests - the session holds:
- Conversation history
- Memory context
- Emotional state

Any client can connect and interact with a session via session ID.

---

## Working Plan (Current)

**Move to Hermes:**
- ReactHandler (ReAct loop orchestration)
- IdleHandler (idle workspace exploration)
- PsycheClient (interface abstraction)
- IdleSettings

**Stay in Psyche:**
- DreamHandler (server-side dreaming)
- PsycheCore (memory coordination)
- Memory tools (recall_memory, store_memory)
- Consolidation (move from IdleHandler to PsycheDaemon)

**Remote mode idle**: Use existing `/v1/chat/completions` endpoint with reflection prompts.

---

## Open Questions

### 1. Architecture Model
Which approach?
- **Option A**: Move orchestration to Hermes, Psyche is pure inference+memory (current plan)
- **Option B**: Session-based (OpenCode-like) - Psyche manages sessions as stateful containers
- **Option C**: Hybrid - Sessions for memory/emotion tracking, Hermes handles orchestration

### 2. Session Concept
Should Psyche have explicit sessions?
- Pro: Cleaner multi-client support, matches OpenCode pattern
- Con: More complex, current HTTP server already rebuilds from messages

### 3. Memory Tools Location
Memory tools stay in Psyche (confirmed by user). But how do they route?
- Local mode: Direct PsycheCore access
- Remote mode: HTTP calls to Psyche server

### 4. Tool Execution Boundary
- **Workspace tools** (read_file, bash, etc.): Execute client-side (Hermes)
- **Memory tools** (recall, store): Execute server-side (Psyche)

This matches OpenCode's model where file/system tools run on client.

---

## Files to Modify (Current Plan)

| File | Action |
|------|--------|
| `src/hermes/handlers/__init__.py` | Create |
| `src/hermes/handlers/psyche_client.py` | Move from psyche |
| `src/hermes/handlers/react_handler.py` | Move from psyche |
| `src/hermes/handlers/idle_handler.py` | Move from psyche |
| `src/hermes/config/settings.py` | Add IdleSettings |
| `src/psyche/handlers/__init__.py` | Remove moved exports |
| `src/psyche/handlers/*.py` | Delete moved files |
| `src/psyche/config/settings.py` | Remove IdleSettings |
| `src/psyche/server/daemon.py` | Add consolidation loop |
| `configs/psyche.toml` | Remove [idle] section |
| `configs/hermes.toml` | Add [idle] section |

---

## Next Steps

1. Decide on architecture model (A, B, or C)
2. If sessions needed, design session API
3. Implement handler moves
4. Add consolidation to PsycheDaemon
5. Update configs and tests

---

## Sources

- [OpenCode Server Docs](https://opencode.ai/docs/server/)
- [OpenCode Internals Deep Dive](https://cefboud.com/posts/coding-agents-internals-opencode-deepdive/)
- [Claude Code MCP Integration](https://www.ksred.com/claude-code-as-an-mcp-server-an-interesting-capability-worth-understanding/)
- [Aider Docs](https://aider.chat/docs/)
