# Bug Investigation Coordination

## Task
Identify critical bugs that could cause task group errors or other runtime failures.

## Agents
1. **Elpis Agent** - Investigates `src/elpis/` (inference MCP server)
2. **Mnemosyne Agent** - Investigates `src/mnemosyne/` (memory MCP server)
3. **Psyche Agent** - Investigates `src/psyche/` (TUI client)

## Focus Areas
- Exception handling gaps
- Async/await issues (missing awaits, unhandled task exceptions)
- Connection/disconnection handling
- Resource cleanup
- Type mismatches
- Uninitialized variables

## Questions / Findings

(Agents: post your findings and questions here)

---
