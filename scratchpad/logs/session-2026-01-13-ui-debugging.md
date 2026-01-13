# Session Log: Psyche UI Debugging and Improvements

**Date:** 2026-01-13
**Branch:** `psyche-ui-improvements`
**Focus:** Bug fixes and improvements to the Textual TUI

## Summary

Fixed multiple issues with the Psyche Textual UI implementation from the previous session, plus added idle tool rate limiting.

## Issues Fixed

### 1. TUI Crash on Submit
**Problem:** App crashed when user submitted input - `UserInput.Submitted` had different signature than Textual's base `Input.Submitted`.

**Fix:** Renamed custom message class to `UserSubmitted` to avoid conflict.
- `src/psyche/client/widgets/user_input.py` - Renamed class, moved outside widget
- `src/psyche/client/app.py` - Updated handler to `on_user_submitted()`

### 2. Logging Breaking TUI Display
**Problem:** Logging output to stderr broke the Textual display.

**Fixes (multiple iterations):**
1. Configure loguru early in cli.py before imports
2. Redirect `sys.stderr` to `~/.psyche/stderr.log` during TUI run
3. Added `ELPIS_QUIET` env var for server subprocess logging control
   - Elpis server checks this var and logs to file when set
   - ElpisClient sets `ELPIS_QUIET=1` when spawning subprocess

**Log locations:**
- `~/.psyche/psyche.log` - Psyche client logs
- `~/.psyche/stderr.log` - Native library stderr output
- `~/.elpis/elpis-server.log` - Elpis server logs (when quiet mode)

### 3. Streaming Display Not Working
**Problem:** `RichLog.write()` doesn't support `end=""` - each token appeared on new line.

**Fix:** Rewrote `ChatView` to use composite widget structure:
- `VerticalScroll` container
- `RichLog` for completed message history
- `StreamingMessage` (Static widget) for current streaming response
- Streaming message updates in place with cursor indicator (▌)
- On completion, content moves to history log

### 4. ThoughtPanel Hidden by Default
**Problem:** Internal thoughts panel was hidden, user wanted it visible.

**Fix:**
- Changed `visible: reactive[bool] = reactive(True)`
- Updated CSS to use `.hidden` class instead of `.visible`

### 5. Excessive Tool Use in Dream State
**Problem:** Psyche used tools too frequently during idle/dream state.

**Fix:** Added rate limiting in `ServerConfig`:
```python
startup_warmup_seconds: float = 120.0  # No tools for first 2 minutes
idle_tool_cooldown_seconds: float = 300.0  # 5 min between idle tool uses
```

New method `_can_use_idle_tools()` checks:
1. Startup warmup period elapsed
2. Cooldown since last idle tool use

### 6. User Input Ignored During Idle Tool Use
**Problem:** Async inference loop blocked while `_generate_idle_thought()` ran with tools.

**Fix:** Rewrote `_inference_loop()` to run idle thinking as cancellable background task:
- Uses `asyncio.wait()` to monitor both input queue and idle task
- User input cancels idle task immediately
- Ensures responsive interaction during dream state

## Files Modified

| File | Changes |
|------|---------|
| `src/psyche/client/widgets/user_input.py` | Renamed Submitted → UserSubmitted |
| `src/psyche/client/widgets/__init__.py` | Export UserSubmitted |
| `src/psyche/client/app.py` | Updated message handler |
| `src/psyche/client/widgets/chat_view.py` | Rewrote with StreamingMessage |
| `src/psyche/client/widgets/thought_panel.py` | Default visible, use hidden class |
| `src/psyche/client/app.tcss` | Updated CSS for new structure |
| `src/psyche/cli.py` | Early logging setup, stderr redirect |
| `src/psyche/mcp/client.py` | Added quiet param, set ELPIS_QUIET env |
| `src/elpis/server.py` | Check ELPIS_QUIET for logging destination |
| `src/psyche/memory/server.py` | Rate limiting + async loop fix |

## Commits

1. `8cf0fdb` - Fix Textual TUI crash and logging interference
2. `aec62b4` - Fix TUI logging interference and streaming display
3. `d062b18` - Fix ThoughtPanel visibility and logging interference
4. `5afa1a9` - Add ELPIS_QUIET env var to suppress server logging in TUI mode
5. `801fc44` - Add rate limiting for tool use during idle/dream state
6. `525ab54` - Fix async inference loop to not block on idle thinking

## Test Results

All 267 tests pass after all changes.

## Architecture Notes

### Idle Tool Rate Limiting Flow
```
User starts Psyche
    │
    ▼
startup_warmup_seconds (120s)
    │ No idle tools allowed
    ▼
After warmup: idle tools enabled
    │
    ▼
Tool used during idle
    │
    ▼
idle_tool_cooldown_seconds (300s)
    │ No idle tools allowed
    ▼
Cooldown elapsed: idle tools re-enabled
```

### Async Inference Loop
```
┌─────────────────────────────────────────┐
│          _inference_loop()              │
├─────────────────────────────────────────┤
│  idle_task running?                     │
│     YES → asyncio.wait([input, idle])   │
│           ├─ input arrives → cancel idle│
│           └─ idle done → process result │
│     NO  → wait_for(input, timeout)      │
│           ├─ input arrives → process    │
│           └─ timeout → start idle task  │
└─────────────────────────────────────────┘
```

## Next Steps

- Test full flow with actual LLM to verify all fixes work end-to-end
- Consider merging branch to main when stable
- Fine-tune rate limiting values based on usage patterns
