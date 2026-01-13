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

### 7. edit_file Tool Confusion
**Problem:** Psyche tried to use `edit_file` on non-existent files, causing failures when trying to create new files (e.g., petting zoo project).

**Fix:** Improved tool description in `tool_engine.py`:
- Clearly states file MUST already exist
- Directs LLM to use `create_file` for new files
- Emphasizes `old_string` must match EXACTLY and be unique

### 8. Streaming Never Ending During Tool Iterations
**Problem:** During ReAct tool iterations, tokens kept streaming without end. `on_response` was only called after final response, so `end_stream()` never triggered between tool calls.

**Fix:** Call `on_response` after EACH LLM generation:
- Stream properly ends before tool execution begins
- New stream starts automatically for next generation
- User sees distinct responses instead of endless stream

### 9. Psyche Continues Speaking Without Accepting Input
**Problem:** After responding to user, Psyche would start idle thinking almost immediately (30s default), appearing to "continue speaking".

**Fix:** Added post-interaction delay in `ServerConfig`:
```python
post_interaction_delay: float = 60.0  # Wait 60s after user speaks
```

New method `_can_start_idle_thinking()` checks time since last user interaction before allowing idle thoughts to begin.

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
| `src/psyche/memory/server.py` | Rate limiting, async loop, post-interaction delay, stream fix |
| `src/psyche/tools/tool_engine.py` | Improved edit_file description |

## Commits

1. `8cf0fdb` - Fix Textual TUI crash and logging interference
2. `aec62b4` - Fix TUI logging interference and streaming display
3. `d062b18` - Fix ThoughtPanel visibility and logging interference
4. `5afa1a9` - Add ELPIS_QUIET env var to suppress server logging in TUI mode
5. `801fc44` - Add rate limiting for tool use during idle/dream state
6. `525ab54` - Fix async inference loop to not block on idle thinking
7. `fc80995` - Add session log for UI debugging work
8. `9286bd4` - Fix streaming and tool use issues

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
│           └─ timeout → check delays     │
│                 ├─ post_interaction OK? │
│                 │   YES → start idle    │
│                 │   NO  → loop again    │
└─────────────────────────────────────────┘
```

### ReAct Loop with Stream Completion
```
User sends message
    │
    ▼
_process_user_input()
    │
    ├─► Generate response (stream tokens via on_token)
    │         │
    │         ▼
    │   Tool call found?
    │         │
    │    YES  │  NO
    │    ▼    │  ▼
    │   on_response() ◄──┘
    │   (ends stream)
    │         │
    │    YES  │
    │    ▼    │
    │   Execute tool
    │         │
    │    Loop back to generate
    │         │
    └─────────┘
```

## Outcome

Psyche successfully created her petting zoo project (`petting_zoo.py`, `sheep_class.py`) after the fixes were applied, demonstrating that the file tools now work correctly.

## Next Steps

- Continue testing with actual LLM usage
- Consider merging branch to main when stable
- Fine-tune timing values based on user experience:
  - `post_interaction_delay` (currently 60s)
  - `idle_tool_cooldown_seconds` (currently 300s)
  - `startup_warmup_seconds` (currently 120s)
