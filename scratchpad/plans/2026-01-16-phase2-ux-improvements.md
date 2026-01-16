# Phase 2: UX Improvements Implementation Plan

**Branch**: `phase2/ux-improvements`
**Sessions**: 4 (Sessions 6-9 of master workplan)

## Summary

| Session | Track | Focus | Key Changes |
|---------|-------|-------|-------------|
| 6 | C1 | Tool Display | Formatter + widget enhancement |
| 7 | C2.1+C2.3 | Interruption | Streaming interrupt + keybindings |
| 8 | C2.2 | Tool Timeout | Tool execution timeout/cancel |
| 9 | C3+C4 | Help + Dreams | Command registry + streaming thoughts |

---

## Session 6: C1 - Tool Display Enhancement

### Problem
Tool activity shows only name + status icon. Full args and result details available but unused.

### Changes

**1. New file: `src/psyche/client/formatters/tool_formatter.py`**
```python
class ToolDisplayFormatter:
    TEMPLATES = {
        "read_file": "Reading {file_path}",
        "execute_bash": "$ {command}",
        ...
    }

    @classmethod
    def format_start(cls, tool_name: str, args: dict) -> str

    @classmethod
    def format_result(cls, tool_name: str, result: dict) -> str
```

**2. Modify: `src/psyche/memory/server.py`**
- Change callback signature: `on_tool_call(name, args, result)`
- Pass args at start (currently only passes `None`)

**3. Modify: `src/psyche/client/widgets/tool_activity.py`**
- Store args in `ToolExecution` dataclass
- Use formatter for display text
- Show result summary on completion

**4. Modify: `src/psyche/client/app.py`**
- Update `_on_tool_call` to handle new signature

### Result
Display shows: `[OK] Reading src/main.py (150 lines)` instead of `[OK] read_file`

---

## Session 7: C2.1+C2.3 - Streaming Interruption

### Problem
Ctrl+C quits entire app. No way to interrupt generation mid-stream.

### Changes

**1. Modify: `src/psyche/memory/server.py`**
```python
# Add to __init__
self._interrupt_event: asyncio.Event = asyncio.Event()

# Add method
def interrupt(self) -> bool:
    """Request interruption of current generation."""
    if self._state == ServerState.THINKING:
        self._interrupt_event.set()
        return True
    return False

# In streaming loop, check for interrupt
if self._interrupt_event.is_set():
    self._interrupt_event.clear()
    response_tokens.append("\n\n[Interrupted]")
    break
```

**2. Modify: `src/psyche/client/app.py`**
```python
BINDINGS = [
    Binding("ctrl+c", "interrupt_or_quit", "Stop/Quit"),  # Changed
    Binding("ctrl+q", "quit", "Quit", show=False),        # New explicit quit
    ...
]

# Track last Ctrl+C time for double-tap quit
_last_ctrl_c: float = 0.0
DOUBLE_TAP_THRESHOLD = 1.5  # seconds

async def action_interrupt_or_quit(self) -> None:
    import time
    now = time.time()

    # If generating, interrupt
    if self.memory_server.state == ServerState.THINKING:
        self.memory_server.interrupt()
        return

    # Double-tap to quit when idle
    if now - self._last_ctrl_c < DOUBLE_TAP_THRESHOLD:
        await self.action_quit()
    else:
        self._last_ctrl_c = now
        chat.add_system_message("[dim]Press Ctrl+C again to quit, or Ctrl+Q[/]")
```

### Result
- Ctrl+C during generation: interrupts and shows "[Interrupted]"
- Ctrl+C when idle: first shows "Press again to quit", second quits
- Ctrl+Q: always quits immediately

---

## Session 8: C2.2 - Tool Execution Timeout

### Problem
Long-running tools (e.g., slow bash commands) can't be cancelled.

### Changes

**1. Modify: `src/psyche/tools/tool_engine.py`**
```python
async def execute_tool_call(self, tool_call: dict, timeout: float = None):
    try:
        async with asyncio.timeout(timeout or self.settings.bash_timeout):
            result = await tool_def.handler(**args)
    except asyncio.TimeoutError:
        return {"success": False, "error": f"Timed out after {timeout}s"}
```

**2. Modify: `src/psyche/memory/server.py`**
- Check `_interrupt_event` at start of ReAct loop iterations
- Skip tool execution if interrupted

### Result
- Tools have configurable timeout
- Ctrl+C can interrupt tool execution

---

## Session 9: C3+C4 - Help & Dream Streaming

### C3: Help System

**1. New file: `src/psyche/client/commands.py`**
```python
@dataclass
class Command:
    name: str
    aliases: List[str]  # e.g., ["h", "?"] for help
    description: str
    shortcut: Optional[str]

COMMANDS = {
    "help": Command("help", ["h", "?"], "Show help", None),
    "quit": Command("quit", ["q", "exit"], "Exit", "Ctrl+Q"),
    ...
}

def get_command(name: str) -> Optional[Command]
def format_help_text() -> str
def format_startup_hint() -> str
```

**2. Modify: `src/psyche/client/app.py`**
- Show startup hint on mount
- Use registry in `_handle_command` for alias support

### C4: Dream State Streaming

**1. Modify: `src/psyche/memory/server.py`**
- Add `on_thinking: Callable[[str], None]` callback
- Change `_generate_idle_thought` to use `generate_stream()`
- Call `on_thinking(token)` for each token

**2. Modify: `src/psyche/client/app.py`**
- Register `on_thinking` callback
- Show "[Thinking...]" indicator during idle thought generation

**3. Modify: `src/psyche/client/widgets/thought_panel.py`**
- Add streaming indicator support (RichLog limitation: show indicator, then final thought)

### Result
- Startup shows hint message
- `/h`, `/q`, `/c` work as aliases
- Thought panel shows activity during idle generation

---

## Critical Files

| File | Sessions | Changes |
|------|----------|---------|
| `src/psyche/memory/server.py` | 6,7,8,9 | Callback signatures, interrupt event, streaming idle |
| `src/psyche/client/app.py` | 6,7,9 | Callbacks, keybindings, commands |
| `src/psyche/client/widgets/tool_activity.py` | 6 | Enhanced display |
| `src/psyche/tools/tool_engine.py` | 8 | Timeout support |
| `src/psyche/client/commands.py` | 9 | New: command registry |
| `src/psyche/client/formatters/tool_formatter.py` | 6 | New: display formatter |

---

## Dependencies

```
Session 6 (C1) ─────────────────────────────────────────────┐
                                                            │
Session 7 (C2.1+C2.3) ──┬── Session 8 (C2.2) ───────────────┤
                        │                                   │
                        └── Session 9 (C3+C4) ──────────────┘
```

- Session 8 depends on Session 7 (uses interrupt event)
- Session 9 C4 depends on Session 7 (uses interrupt event for idle cancellation)
- Sessions 6 and 7 can run in parallel if needed

---

## Testing Checklist

### C1 (Tool Display)
- [ ] `read_file` shows "Reading path/to/file.py"
- [ ] Completion shows "(150 lines)"
- [ ] Errors show formatted error message
- [ ] Unknown tools fall back gracefully

### C2 (Interruption)
- [ ] Ctrl+C during streaming shows "[Interrupted]"
- [ ] Ctrl+C when idle shows "Press again to quit"
- [ ] Double Ctrl+C when idle quits app
- [ ] Ctrl+Q always quits immediately
- [ ] Long bash command times out
- [ ] Rapid Ctrl+C doesn't crash

### C3 (Help)
- [ ] Startup shows hint message
- [ ] `/h` shows help (alias)
- [ ] `/q` quits (alias)
- [ ] Unknown command shows friendly error

### C4 (Dream Streaming)
- [ ] Thought panel shows activity during idle
- [ ] User input interrupts idle thinking
- [ ] Final thought appears correctly
