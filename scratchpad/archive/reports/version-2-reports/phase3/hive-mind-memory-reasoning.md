# Phase 3: Memory & Reasoning - Hive Mind Coordination

**Branch**: `phase3/memory-reasoning`
**Started**: 2026-01-16
**Status**: In Progress

## Active Agents

| Agent | Session | Focus | Status |
|-------|---------|-------|--------|
| Summarization Agent | 10 (B2.3) | Verify existing summarization, add tests | **Complete** |
| Importance Agent | 11 (B2.4) | Implement importance scoring module | **Complete** |
| Reasoning Agent | 12 (D1) | Implement `<thinking>` tag workflow | **Complete** |

## Dependency Graph

```
Session 10 (B2.3) ─────────────────────────────────────────┐
                                                           │
Session 11 (B2.4) ──────────────────────────┬──────────────┤
                                            │              │
Session 12 (D1) ────────────────────────────┴──────────────┤
                                                           │
                                 Session 13 (Integration) ─┘
```

Sessions 10, 11, 12 are **independent** and can run in parallel.
Session 13 (Integration Testing) requires all three to complete first.

## File Ownership

To avoid conflicts, each agent owns specific files:

### Summarization Agent (Session 10)
- `tests/psyche/integration/test_summarization.py` (new)
- `src/psyche/memory/server.py` - ONLY lines ~1303-1356 (summary prompt)

### Importance Agent (Session 11)
- `src/psyche/memory/importance.py` (new)
- `src/psyche/memory/config.py` - add auto_storage settings
- `src/psyche/memory/server.py` - ONLY add `_after_response()` method

### Reasoning Agent (Session 12)
- `src/psyche/memory/reasoning.py` (new)
- `src/psyche/client/widgets/thought_panel.py` - add "reasoning" type
- `src/psyche/client/commands.py` - add `/thinking` command
- `src/psyche/client/app.py` - add keybinding and handler
- `src/psyche/memory/server.py` - ONLY reasoning mode toggle and system prompt addition

## Coordination Notes

- **Do not modify files owned by other agents** without coordination
- Write your report to `scratchpad/reports/phase3/session-{N}-report.md`
- If you need to ask a question, add it to the Questions section below
- Check this file before making changes to shared files (server.py)

## Questions

(Add questions here for other agents or the coordinator)

## Completion Checklist

- [x] Session 10: Summarization verification complete (20 tests, all pass)
- [x] Session 11: Importance scoring implemented (32 tests, all pass)
- [x] Session 12: Reasoning workflow implemented (22 tests, all pass)
- [x] All unit tests pass
- [ ] Session 13: Integration testing (after above complete)
