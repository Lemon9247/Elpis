# Session 7: Streaming Interruption (C2.1+C2.3)

**Date**: 2026-01-16
**Branch**: phase2/ux-improvements
**Session Duration**: ~1 session

## Summary

Implemented Ctrl+C interrupt handling for the Psyche TUI client. Previously, Ctrl+C would immediately quit the application. Now it provides context-aware behavior:

- During generation: Interrupts the stream and shows "[Interrupted]"
- When idle (first tap): Shows "Press Ctrl+C again to quit, or Ctrl+Q"
- When idle (second tap within 1.5s): Quits the application
- Ctrl+Q: Always quits immediately

## Changes Made

### 1. `src/psyche/memory/server.py`

Added interrupt mechanism to the MemoryServer class:

1. **New instance variable** (line 181-182):
   ```python
   # Interrupt event for stopping generation mid-stream
   self._interrupt_event: asyncio.Event = asyncio.Event()
   ```

2. **New `interrupt()` method** (lines 317-329):
   ```python
   def interrupt(self) -> bool:
       """Request interruption of current generation."""
       if self._state == ServerState.THINKING:
           self._interrupt_event.set()
           logger.debug("Generation interrupt requested")
           return True
       return False
   ```

3. **Interrupt check in streaming loop** (lines 642-686):
   - Added `interrupted` flag before the streaming try block
   - Added interrupt check inside the token streaming loop:
     ```python
     if self._interrupt_event.is_set():
         self._interrupt_event.clear()
         logger.info("Generation interrupted by user")
         interrupted = True
         break
     ```
   - After the streaming completes, handles the interrupted state by:
     - Appending "\n\n[Interrupted]" marker to response
     - Adding partial response to context (so conversation history is preserved)
     - Notifying the response callback
     - Returning early from the ReAct loop

### 2. `src/psyche/client/app.py`

Updated keybindings and added interrupt/quit action:

1. **New imports** (lines 4, 23):
   ```python
   import time
   from psyche.memory.server import MemoryServer, ServerState, ThoughtEvent
   ```

2. **Updated BINDINGS** (lines 38-44):
   ```python
   BINDINGS = [
       Binding("ctrl+c", "interrupt_or_quit", "Stop/Quit"),
       Binding("ctrl+q", "quit", "Quit", show=False),
       # ... rest unchanged
   ]
   ```

3. **New class constant** (lines 46-47):
   ```python
   DOUBLE_TAP_THRESHOLD = 1.5  # seconds
   ```

4. **New instance variable** (lines 60-61):
   ```python
   self._last_ctrl_c: float = 0.0
   ```

5. **New `action_interrupt_or_quit()` method** (lines 282-308):
   - Checks if server is in THINKING state; if so, calls `interrupt()` and shows notification
   - If not generating, implements double-tap logic:
     - If second tap within threshold: calls `action_quit()`
     - Otherwise: records timestamp and shows help notification

## Testing

All tests passed:

1. **Syntax validation**: Both files compile without errors
2. **Unit test for interrupt method**: Custom test verified:
   - `interrupt()` returns False when server is idle
   - `interrupt()` returns True when server is thinking
   - `_interrupt_event` is properly set when interrupted
3. **Full test suite**: All 205 tests in `tests/psyche/` passed

## Testing Checklist

- [x] Syntax check passed for both modified files
- [x] Server parsing unit tests pass (8/8)
- [x] Memory server integration tests pass (17/17)
- [x] Full psyche test suite passes (205/205)
- [x] Manual test of interrupt method behavior

## Manual Testing Notes

The implementation should be manually tested with:
1. Start psyche with a connected elpis server
2. Submit a prompt that generates a long response
3. Press Ctrl+C mid-stream -> should see "[Interrupted]" marker
4. When idle, press Ctrl+C once -> should see notification
5. Press Ctrl+C again within 1.5s -> should quit
6. Press Ctrl+Q at any time -> should quit immediately

## Files Modified

- `/home/lemoneater/Projects/Personal/Elpis/src/psyche/memory/server.py`
- `/home/lemoneater/Projects/Personal/Elpis/src/psyche/client/app.py`

## Notes

- The interrupt mechanism uses asyncio.Event which is checked on each token during streaming
- Partial responses are still saved to context to maintain conversation history
- The double-tap threshold of 1.5 seconds provides a good balance between preventing accidental quits and being responsive
- The notification system (using Textual's `notify()`) provides clear user feedback
