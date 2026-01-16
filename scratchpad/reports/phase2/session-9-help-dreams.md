# Session 9: Help System & Dream Streaming

**Date:** 2026-01-16
**Task:** C3 (Help System) + C4 (Dream State Streaming)
**Status:** Completed

## Summary

This session implemented the remaining Phase 2 UX improvements: the help system (C3) and dream state streaming (C4). The help system was already implemented in a previous session, so this session focused on the dream streaming feature.

## C3: Help System (Already Implemented)

Upon review, the help system was found to be fully implemented in a previous session:

### Files Already in Place
- `/src/psyche/client/commands.py` - Complete command registry with:
  - `Command` dataclass with name, aliases, description, and shortcut
  - `COMMANDS` dictionary with all commands (help, quit, clear, status, thoughts, emotion)
  - `get_command()` function for lookup by name or alias
  - `format_help_text()` for displaying all commands
  - `format_shortcut_help()` for keyboard shortcuts
  - `format_startup_hint()` for the startup message

- `/src/psyche/client/app.py` - Already integrated:
  - Imports from commands.py
  - Shows startup hint on mount
  - Uses `get_command()` for alias support in `_handle_command()`

### Existing Functionality
- `/help`, `/h`, `/?" - Show available commands
- `/quit`, `/q`, `/exit` - Exit the application (Ctrl+Q)
- `/clear`, `/c`, `/cls` - Clear chat history (Ctrl+L)
- `/status`, `/s` - Show server status
- `/thoughts`, `/t` - Toggle thought panel (Ctrl+T)
- `/emotion`, `/e` - Show emotional state

## C4: Dream State Streaming (Implemented This Session)

### Changes Made

#### 1. `src/psyche/memory/server.py`
Added `on_thinking` callback to MemoryServer:
```python
def __init__(
    ...
    on_thinking: Optional[Callable[[str], None]] = None,
):
    ...
    self.on_thinking = on_thinking
```

Modified `_generate_idle_thought()` to use streaming:
- Changed from `self.client.generate()` to `self.client.generate_stream()`
- Added interrupt checking during streaming (checks `_interrupt_event`)
- Calls `on_thinking(token)` for each token during generation
- Properly handles `asyncio.CancelledError` for task cancellation

Key changes:
- Streaming tokens are collected while generating
- Each token triggers `on_thinking` callback
- Interrupt event is checked between tokens
- Task cancellation is handled gracefully

#### 2. `src/psyche/client/widgets/thought_panel.py`
Added streaming indicator support:
- New `is_thinking` reactive property
- `_thinking_token_count` to track tokens received
- `start_thinking()` - Shows "[thinking...]" indicator
- `on_thinking_token(token)` - Updates token count in border title
- `stop_thinking()` - Resets thinking state
- `watch_is_thinking()` - Adds/removes "thinking" CSS class
- `add_thought()` now calls `stop_thinking()` to clear indicator

Visual feedback:
- Border title shows: "Internal Thoughts [thinking... N tokens]"
- CSS class "thinking" added during generation for styling

#### 3. `src/psyche/client/app.py`
Registered `on_thinking` callback:
```python
self.memory_server.on_thinking = self._on_thinking
```

Added `_on_thinking()` method:
```python
def _on_thinking(self, token: str) -> None:
    """Handle streaming token during idle thought generation."""
    def update():
        try:
            thoughts = self.query_one("#thoughts", ThoughtPanel)
            thoughts.on_thinking_token(token)
        except Exception:
            pass
    self.call_later(update)
```

## Expected Behavior

1. **Startup**: Shows "Type /help or /h for available commands"
2. **Commands**: `/h`, `/q`, `/c` and other aliases work
3. **Dream State**:
   - When idle thinking starts, thought panel shows "[thinking...]"
   - Border title updates with token count during generation
   - Final thought displayed when generation completes
   - User input can interrupt idle thinking mid-stream

## Files Modified

1. `/src/psyche/memory/server.py` - Added `on_thinking` callback, streaming in `_generate_idle_thought()`
2. `/src/psyche/client/widgets/thought_panel.py` - Added streaming indicator support
3. `/src/psyche/client/app.py` - Registered `on_thinking` callback

## Test Results

All 205 psyche tests pass:
```
============================= 205 passed in 2.32s ==============================
```

## Notes

- The RichLog widget doesn't support editing existing content, so the streaming indicator uses the border title for activity feedback
- The `_interrupt_event` from Session 7 is used to allow cancellation of idle thought generation
- The existing `generate_stream()` method in `ElpisClient` was already available and works well for this use case

## Next Steps

All Phase 2 UX improvements are now complete:
- C1: Streaming output (Session 7)
- C2: Interrupt/cancellation (Session 7)
- C3: Help system (previously implemented)
- C4: Dream streaming (this session)
- C5: Tool timeout (Session 8)
- C6: Tool display (Session 6)
