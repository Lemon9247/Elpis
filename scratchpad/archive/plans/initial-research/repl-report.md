# CLI/REPL Interface Patterns Research Report
## Elpis Interactive Coding Agent

**Date:** January 2026
**Agent:** REPL-Agent
**Status:** Complete

---

## Executive Summary

This report documents comprehensive research into Python REPL/CLI interface libraries and patterns for the Elpis emotional coding agent. We evaluated four primary approaches and identified prompt_toolkit as the optimal foundation for building a sophisticated, user-friendly interactive interface.

**Key Recommendation:** Use **prompt_toolkit** as the core REPL framework, combined with **Rich** for output formatting, complemented by **asyncio** for real-time streaming and background task management.

---

## 1. Python REPL Libraries Comparison

### 1.1 cmd Module (Python Standard Library)

**Overview:** The `cmd` module provides a simple framework for writing line-oriented command interpreters.

**Strengths:**
- Part of Python standard library (no external dependencies)
- Simple, lightweight implementation
- Automatic help system from docstrings
- Built-in command completion via readline
- Excellent for simple prototypes and test harnesses

**Weaknesses:**
- No syntax highlighting
- Limited visual features (no colors, rich formatting)
- Single-line input only (no multiline editing)
- No history persistence mechanism built-in
- Readline-dependent (platform-specific behavior)
- Minimal customization for modern CLI experiences

**Architecture:**
```python
class cmd.Cmd:
    - cmdloop()          # Main command loop
    - onecmd(str)        # Process single command
    - precmd()           # Hook before command
    - postcmd()          # Hook after command
    - do_<command>(arg)  # Command methods
```

**Best For:** Simple administrative tools, test harnesses, prototypes without UI requirements.

**Not Suitable For:** Elpis (requires rich output formatting, multiline editing, streaming capabilities).

---

### 1.2 prompt_toolkit (Python Package)

**Overview:** Advanced library for building powerful interactive command-line applications and REPLs with modern features.

**Strengths:**
- Syntax highlighting during input (via Pygments lexers)
- Multiline editing and navigation
- Advanced code completion (fuzzy matching)
- History with persistent file storage (InMemoryHistory, FileHistory)
- Mouse support for cursor positioning
- Emacs and Vi key bindings
- Auto-suggestions and pager support
- Built-in prompt customization
- Cross-platform (Linux, macOS, Windows, OpenBSD)
- Python 3.6+ support
- Actively maintained (3.0.52+ as of 2026)

**Weaknesses:**
- More complex API (more boilerplate than cmd)
- Requires deeper understanding of async patterns
- Steeper learning curve for beginners

**Core Components:**
```python
# Session Management
PromptSession(history=FileHistory(), completer=Completer)
session.prompt(message, multiline=True)

# Input Handling
from prompt_toolkit.input import create_input
from prompt_toolkit.output import create_output

# Completion
from prompt_toolkit.completion import Completer, Completion

# Validators
from prompt_toolkit.validation import Validator
```

**Error Handling Pattern:**
```python
try:
    text = await session.prompt()
except KeyboardInterrupt:
    # Handle Ctrl+C
except EOFError:
    # Handle Ctrl+D (exit)
```

**Implementation Complexity:** ~20-50 lines for a functional REPL with all standard features.

**Best For:** Elpis primary REPL interface (recommended).

**Real-World Examples:**
- ptpython: Enhanced Python REPL built on prompt_toolkit
- IPython: Uses prompt_toolkit for interactive shell
- litecli, pgcli: Database CLI tools

---

### 1.3 Rich Library (Python Package)

**Overview:** Terminal output formatting and visualization library (not a full REPL, complementary tool).

**Strengths:**
- Colored and styled text output
- Tables, panels, trees, progress bars, markdown rendering
- Syntax highlighting for code
- Live display updates (animated content)
- JSON pretty-printing
- Traceback formatting
- Layout and column management
- Logging integration
- Simple "Console" API

**Weaknesses:**
- Not a complete REPL solution (requires integration with input handling)
- Primarily output-focused (limited input capabilities)
- Interactive prompts are basic (not advanced completion/history)

**Usage Pattern:**
```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
console.print("[bold]Title[/bold]")
console.print(Panel("content"))
```

**Best For:** Output formatting within a prompt_toolkit-based REPL.

**Integration:** Pair Rich with prompt_toolkit for complete solution.

---

### 1.4 cmd2 (Enhanced cmd Module)

**Overview:** Drop-in replacement for cmd with enhanced features.

**Status:** Less recommended than prompt_toolkit but better than cmd.

**Key Differences from cmd:**
- More built-in features (less boilerplate than prompt_toolkit)
- Better out-of-the-box experience
- Still limited visual features compared to prompt_toolkit

**Use Case:** Middle ground between simplicity and features (not needed for Elpis).

---

## 2. Command Loop Structure and Input Handling

### 2.1 Event Loop Architecture

**Pattern: Async/Await with Readline Integration**

```python
import asyncio
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit import PromptSession

async def main():
    session = PromptSession()

    while True:
        try:
            with patch_stdout():
                user_input = await session.prompt_async('> ')

            # Process input
            result = await process_command(user_input)

        except KeyboardInterrupt:
            print("\nInterrupt received. Continuing...")
            continue
        except EOFError:
            print("\nExiting...")
            break

asyncio.run(main())
```

**Key Patterns:**
- `prompt_async()` for non-blocking input (allows concurrent tasks)
- `KeyboardInterrupt` handling (Ctrl+C)
- `EOFError` handling (Ctrl+D)
- `patch_stdout()` context manager (prevents output corruption during async operations)

### 2.2 Input Processing Pipeline

```
User Input
    ↓
[Parsing] → Command tokenization
    ↓
[Validation] → Check syntax, command existence
    ↓
[Execution] → Run command, collect results
    ↓
[Formatting] → Prepare output (Rich formatting)
    ↓
[Display] → Print to console
    ↓
[History] → Store in session history
```

### 2.3 Readline Integration

**Python Standard Library:**
- `readline` module provides history and line editing
- Automatic `.python_history` file in home directory
- Bash-like editing capabilities (arrow keys, Ctrl+A/E, etc.)

**prompt_toolkit Approach:**
- Implements readline-like functionality without readline dependency
- `FileHistory()` for persistent storage
- `InMemoryHistory()` for session-only storage
- Customizable history behavior

---

## 3. Session Management and History

### 3.1 Three-Tier History System (Recommended for Elpis)

```python
from prompt_toolkit.history import FileHistory, InMemoryHistory
import json
from datetime import datetime

class SessionManager:
    def __init__(self, history_dir="./history"):
        # Tier 1: Current session (in-memory, fast)
        self.in_memory = InMemoryHistory()

        # Tier 2: File history (current session backup)
        self.file_history = FileHistory(
            f"{history_dir}/session_{datetime.now().isoformat()}.txt"
        )

        # Tier 3: Metadata database (for emotional context)
        self.metadata = {}

    async def record_interaction(self, command, result, emotional_state):
        """Record interaction with emotional context"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "result": result,
            "emotional_state": emotional_state,
            "status": "success" if result else "failed"
        }
        self.metadata[command] = entry
```

### 3.2 Session Lifecycle

```
Session Start
    ↓
[Initialize] → Load history from file
    ↓
[Main Loop] → Accept commands (REPL loop)
    ↓
[Record] → Store each interaction
    ↓
[Consolidate] → Periodically compress old history
    ↓
[Persist] → Save on exit or at checkpoints
    ↓
Session End
```

### 3.3 History Features for Elpis

**Basic Features:**
- Per-session history file
- Command line history traversal (arrow keys)
- Substring search in history (Ctrl+R)

**Advanced Features:**
- Emotional tagging (success/failure/novel)
- Command success rate tracking
- Pattern detection (repeated commands)
- Context-aware history retrieval

**Implementation:**
```python
# Search history
matching_commands = [
    cmd for cmd in history
    if cmd.startswith("def ")
    and emotional_state["dopamine"] > 0.7
]
```

---

## 4. Output Formatting and Streaming

### 4.1 Streaming Output Pattern

**Using Rich + asyncio:**

```python
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
import asyncio

async def stream_response(agent_output):
    """Stream agent response in real-time"""
    console = Console()

    with Live(Panel("Starting..."), refresh_per_second=4) as live:
        buffer = ""

        async for chunk in agent_output:
            buffer += chunk

            # Update display with formatted content
            live.update(
                Panel(buffer, title="[bold blue]Agent Thinking[/bold blue]")
            )
            await asyncio.sleep(0)  # Yield control
```

### 4.2 Output Categories and Formatting

```python
from enum import Enum
from rich.syntax import Syntax
from rich.table import Table

class OutputType(Enum):
    AGENT_THINKING = "Agent Reasoning"
    TOOL_CALL = "Tool Execution"
    TOOL_RESULT = "Tool Result"
    ERROR = "Error"
    SUCCESS = "Success"
    EMOTION_STATE = "Emotional State"

async def format_output(message, output_type):
    """Format output based on type"""

    if output_type == OutputType.AGENT_THINKING:
        console.print(Panel(message, style="blue"))

    elif output_type == OutputType.TOOL_CALL:
        console.print(Panel(f"→ {message}", style="yellow"))

    elif output_type == OutputType.TOOL_RESULT:
        console.print(Panel(message, style="green"))

    elif output_type == OutputType.ERROR:
        console.print(Panel(message, style="bold red"))

    elif output_type == OutputType.EMOTION_STATE:
        # Display emotional state as gauge
        display_emotion_dashboard(message)
```

### 4.3 Real-Time Monitoring Display

```python
from rich.dashboard import Dashboard
from rich.progress import Progress, SpinnerColumn, BarColumn

async def display_streaming_with_progress(agent):
    """Show progress during agent work"""

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
    ) as progress:

        task = progress.add_task("[cyan]Processing...", total=100)

        async for event in agent.stream():
            if event.type == "thinking":
                progress.update(task, advance=10)
            elif event.type == "tool_call":
                progress.update(task, advance=20)
            elif event.type == "tool_result":
                progress.update(task, advance=30)
```

---

## 5. Special Commands and Debugging Interface

### 5.1 Meta-Commands Pattern

Commands prefixed with special character (e.g., `/`, `.`, `:`) for control operations.

```python
class SpecialCommands:
    """Built-in meta-commands for REPL"""

    COMMANDS = {
        "/help": "Show available commands",
        "/history": "Display command history",
        "/clear": "Clear screen",
        "/exit": "Exit REPL",
        "/memory": "Inspect memory systems",
        "/emotions": "Show emotional state",
        "/debug": "Enable debug mode",
        "/trace": "Show execution trace",
        "/inspect": "Inspect variable/object",
        "/config": "Show/modify configuration",
        "/session": "Manage current session",
    }

    async def handle_special_command(self, command: str):
        """Route to appropriate handler"""
        cmd_name = command.split()[0]

        if cmd_name == "/help":
            return await self.cmd_help()
        elif cmd_name == "/emotions":
            return await self.cmd_show_emotions()
        elif cmd_name == "/memory":
            return await self.cmd_inspect_memory()
        # ... etc
```

### 5.2 Debugging Commands Integration

**Using pdb for interactive debugging:**

```python
import pdb
import sys

async def debug_mode(agent, breakpoint_on_error=True):
    """Enable debugging features"""

    original_trace = sys.gettrace()

    def trace_calls(frame, event, arg):
        if event == 'exception' and breakpoint_on_error:
            # Launch pdb debugger
            pdb.post_mortem()
        return trace_calls

    if breakpoint_on_error:
        sys.settrace(trace_calls)

    try:
        await agent.run()
    finally:
        sys.settrace(original_trace)
```

### 5.3 Inspection Commands

```python
async def cmd_inspect(self, target: str):
    """Inspect object or variable"""
    from rich.pretty import Pretty

    # Handle memory inspection
    if target.startswith("memory:"):
        key = target.split(":", 1)[1]
        result = await self.memory.retrieve(key)
        console.print(Pretty(result))

    # Handle emotion inspection
    elif target.startswith("emotion:"):
        emotion = target.split(":", 1)[1]
        state = self.emotional_system.get_state(emotion)
        display_emotion_gauge(emotion, state)

    # Handle variable inspection
    else:
        try:
            obj = eval(target)
            console.print(Pretty(obj))
        except Exception as e:
            console.print(f"[red]Error inspecting {target}: {e}[/red]")
```

### 5.4 Debugging Interface Architecture

```
┌─────────────────────────────────────┐
│ REPL Input Layer                    │
│ (prompt_toolkit + Rich)             │
└──────────────┬──────────────────────┘
               │
        ┌──────▼──────┐
        │ Parse Input │
        └──────┬──────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
Regular Command    Special Command (/)
    │                     │
    ├─> Execute          ├─> /debug → Enable tracing
    ├─> Return Result    ├─> /trace → Show execution path
    └─> Record           ├─> /inspect → Show state
                         ├─> /memory → Query memory
                         ├─> /emotions → Show emotions
                         └─> /breakpoint → Launch pdb
```

---

## 6. Implementation Patterns and Code Examples

### 6.1 Basic REPL Structure (prompt_toolkit + Rich)

```python
import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.panel import Panel

class ElpisREPL:
    def __init__(self, agent, history_dir="./history"):
        self.agent = agent
        self.console = Console()

        self.session = PromptSession(
            history=FileHistory(f"{history_dir}/commands.txt"),
            completer=WordCompleter(self.get_commands()),
            multiline=False,
            mouse_support=True,
        )

    def get_commands(self):
        """Get available commands for completion"""
        return [
            "help", "history", "clear", "exit",
            "memory", "emotions", "debug", "trace",
            "inspect", "config", "session"
        ]

    async def run(self):
        """Main REPL loop"""
        self.console.print(Panel(
            "[bold cyan]Elpis Interactive Coding Agent[/bold cyan]\n"
            "Type [yellow]/help[/yellow] for available commands"
        ))

        while True:
            try:
                # Get user input
                user_input = await self.session.prompt_async('elpis> ')

                if not user_input.strip():
                    continue

                # Handle special commands
                if user_input.startswith('/'):
                    await self.handle_special_command(user_input)
                else:
                    # Send to agent
                    await self.agent.process(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Continuing...[/yellow]")
            except EOFError:
                self.console.print("\n[cyan]Goodbye![/cyan]")
                break

    async def handle_special_command(self, command: str):
        """Route special commands"""
        cmd_name = command.split()[0].lstrip('/')

        if cmd_name == "help":
            self.console.print(Panel(self._help_text()))
        elif cmd_name == "emotions":
            await self.show_emotions()
        elif cmd_name == "clear":
            self.console.clear()
        # ... etc
```

### 6.2 Async Agent Loop with Streaming

```python
class AgentLoop:
    def __init__(self, llm, memory, emotions, tools):
        self.llm = llm
        self.memory = memory
        self.emotions = emotions
        self.tools = tools
        self.console = Console()

    async def process(self, user_input: str):
        """Process user input through agent loop"""

        # 1. Retrieve context
        context = await self.memory.retrieve_context(user_input)
        emotional_state = self.emotions.get_state()

        # 2. Build prompt with emotional modulation
        prompt = self.build_prompt(user_input, context, emotional_state)

        # 3. Stream agent response
        with Live(Panel("Thinking..."), refresh_per_second=4) as live:
            buffer = ""

            async for chunk in self.llm.stream(prompt):
                buffer += chunk

                # Update display
                live.update(Panel(
                    buffer,
                    title="[bold blue]Agent Output[/bold blue]"
                ))

        # 4. Parse and execute tool calls if needed
        tool_results = await self.execute_tools(buffer)

        # 5. Update emotional state based on results
        await self.emotions.update(
            success=tool_results["success"],
            error=tool_results.get("error")
        )

        # 6. Store in memory
        await self.memory.store_interaction(
            command=user_input,
            response=buffer,
            emotional_state=emotional_state,
            results=tool_results
        )

    async def execute_tools(self, response: str):
        """Extract and execute tool calls from response"""

        tool_calls = self.parse_tool_calls(response)
        results = []

        for tool_call in tool_calls:
            try:
                result = await self.tools.execute(
                    tool_call.name,
                    **tool_call.args
                )
                results.append({
                    "success": True,
                    "tool": tool_call.name,
                    "result": result
                })

                # Display tool execution
                self.console.print(Panel(
                    f"✓ {tool_call.name}: {result}",
                    style="green"
                ))

            except Exception as e:
                results.append({
                    "success": False,
                    "tool": tool_call.name,
                    "error": str(e)
                })

                self.console.print(Panel(
                    f"✗ {tool_call.name}: {e}",
                    style="red"
                ))

        return {
            "success": all(r["success"] for r in results),
            "results": results
        }
```

### 6.3 Emotional State Display

```python
def display_emotion_dashboard(emotions: dict):
    """Display emotional state as rich dashboard"""
    from rich.table import Table
    from rich.progress import BarColumn

    table = Table(title="Emotional State")
    table.add_column("Neuromodulator", style="cyan")
    table.add_column("Level", style="magenta")
    table.add_column("Status", justify="left")

    for neurotransmitter, value in emotions.items():
        # Create visual bar
        bar_length = int(value * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)

        # Determine status
        if value > 0.7:
            status = "🔥 High"
            style = "bold red"
        elif value > 0.4:
            status = "→ Normal"
            style = "yellow"
        else:
            status = "❄️ Low"
            style = "blue"

        table.add_row(
            neurotransmitter,
            f"{value:.2f}",
            f"[{style}]{status}[/{style}]"
        )

    console.print(table)
```

### 6.4 Session Management Example

```python
class SessionManager:
    def __init__(self, session_dir="./sessions"):
        self.session_dir = session_dir
        self.current_session = None

    async def create_session(self, name: str = None):
        """Create new session"""
        from datetime import datetime
        import uuid

        if name is None:
            name = f"session_{datetime.now().isoformat()}"

        session_id = str(uuid.uuid4())

        self.current_session = {
            "id": session_id,
            "name": name,
            "created_at": datetime.now(),
            "interactions": [],
            "emotional_state": {},
            "memory": {}
        }

        # Save to file
        await self.save_session()
        return session_id

    async def save_session(self):
        """Persist session to disk"""
        import json

        session_file = f"{self.session_dir}/{self.current_session['id']}.json"

        with open(session_file, 'w') as f:
            json.dump(self.current_session, f, indent=2, default=str)

    async def load_session(self, session_id: str):
        """Load previous session"""
        import json

        session_file = f"{self.session_dir}/{session_id}.json"

        with open(session_file, 'r') as f:
            self.current_session = json.load(f)

        return self.current_session
```

---

## 7. Architecture Recommendation for Elpis

### 7.1 Recommended Stack

```
┌────────────────────────────────────────────────────┐
│ Input Layer                                        │
│ (prompt_toolkit PromptSession)                     │
│ - Multiline editing, completion, history          │
└────────────────┬─────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────┐
│ Command Processing                               │
│ - Parse input (regular vs special commands)      │
│ - Route to handlers                              │
└────────────────┬─────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼────────┐  ┌──────▼──────────────┐
│ Regular Cmd    │  │ Special Command     │
│                │  │                     │
│ Send to agent  │  │ /emotions → Display │
│ Execute tools  │  │ /memory → Query     │
│ Return result  │  │ /debug → Enable     │
└────────┬───────┘  └──────┬──────────────┘
         │                 │
         └─────────────────┘
                 │
    ┌────────────▼──────────────┐
    │ Output Formatting         │
    │ (Rich Console)            │
    │ - Syntax highlighting     │
    │ - Panels, tables, bars    │
    │ - Streaming updates       │
    └────────────┬──────────────┘
                 │
┌────────────────▼─────────────────────────────────┐
│ History & Session Management                     │
│ - Store in FileHistory                           │
│ - Record emotional context                       │
│ - Persist session state                          │
└────────────────────────────────────────────────────┘

Async Foundation: asyncio event loop
- Non-blocking I/O for agent processing
- Concurrent task management
- Real-time streaming
```

### 7.2 Technology Stack Summary

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Input Handling** | prompt_toolkit | Multiline, completion, history, cross-platform |
| **Output Formatting** | Rich | Beautiful terminals, syntax highlighting, tables |
| **Event Loop** | asyncio | Non-blocking I/O, concurrent tasks |
| **Debugging** | pdb + custom commands | Standard Python debugging + agent-specific inspection |
| **Session Management** | FileHistory + custom JSON | Prompt_toolkit history + metadata storage |
| **Streaming** | asyncio + Rich Live | Real-time updates during agent thinking |

---

## 8. Implementation Roadmap for Elpis REPL

### Phase 1: Basic REPL (Week 1-2)
- [ ] Set up prompt_toolkit PromptSession
- [ ] Implement basic command routing
- [ ] Add Rich-formatted output
- [ ] Test with simple agent loop

**Deliverable:** Working REPL that accepts commands and displays agent responses.

### Phase 2: Advanced Features (Week 3-4)
- [ ] Add special commands (/help, /history, /clear)
- [ ] Implement emotion display dashboard
- [ ] Add session management
- [ ] Integrate FileHistory for persistence

**Deliverable:** Feature-complete REPL with emotional state monitoring.

### Phase 3: Debugging & Inspection (Week 5-6)
- [ ] Implement /debug command
- [ ] Add /inspect for variable inspection
- [ ] Integrate pdb for breakpoints
- [ ] Add execution tracing

**Deliverable:** Full debugging interface for monitoring agent behavior.

### Phase 4: Polish & Integration (Week 7-8)
- [ ] Streaming output optimization
- [ ] Error handling refinement
- [ ] Performance tuning
- [ ] Documentation and examples

**Deliverable:** Production-ready REPL interface.

---

## 9. Key Code Patterns and Best Practices

### 9.1 Error Recovery Pattern

```python
async def safe_execute(command: str, handler):
    """Execute with graceful error recovery"""
    try:
        return await handler(command)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted[/yellow]")
        return None
    except asyncio.TimeoutError:
        console.print("[red]Timeout[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        # Continue REPL loop
        return None
```

### 9.2 Streaming Pattern

```python
async def stream_with_display(generator, title: str):
    """Stream output with live display"""
    with Live(Panel(title), refresh_per_second=4) as live:
        content = ""
        async for chunk in generator:
            content += chunk
            live.update(Panel(content, title=title))
```

### 9.3 State Persistence Pattern

```python
async def auto_save_state(interval: float = 30):
    """Periodically save state"""
    while True:
        await asyncio.sleep(interval)
        await session_manager.save_session()
```

---

## 10. Comparison: REPL vs Chat Interface

### When to Use REPL:
- Agent operates as long-running process
- User provides commands incrementally
- History and context matter
- Debugging and inspection needed
- Emotion/state monitoring desired

### When to Use Chat Interface:
- Conversational, back-and-forth interaction
- Single request-response cycles
- Minimal state persistence needed
- Simpler user mental model

**For Elpis:** REPL is the correct choice due to multi-session memory, emotional continuity, and inspection requirements.

---

## 11. Potential Challenges and Solutions

### Challenge 1: Async/Await Complexity
**Issue:** Managing async code in REPL context
**Solution:**
- Use asyncio REPL (Python 3.8+)
- Hide async complexity in helper functions
- Provide clear async examples

### Challenge 2: Streaming Output Corruption
**Issue:** Background tasks interfering with terminal output
**Solution:**
- Use Rich's `patch_stdout()` context manager
- Implement output buffering
- Queue messages to display

### Challenge 3: History Size Growth
**Issue:** History file becomes unwieldy
**Solution:**
- Implement periodic history rotation (e.g., daily files)
- Archive old sessions
- Compress metadata

### Challenge 4: Performance with Large Responses
**Issue:** Streaming large outputs becomes slow
**Solution:**
- Implement pagination for long responses
- Use async generators with batching
- Add response size limits

---

## 12. Testing Strategy

```python
# Unit tests for command parsing
async def test_parse_command():
    assert parse_command("/emotions") == ("special", "emotions")
    assert parse_command("help me") == ("regular", "help me")

# Integration tests for REPL
async def test_repl_loop():
    repl = ElpisREPL(mock_agent)
    # Simulate user input
    # Verify output

# Performance tests
async def test_streaming_performance():
    # Measure latency for large responses
    # Ensure sub-second display updates
```

---

## 13. Additional Resources

### Official Documentation
- [prompt_toolkit Documentation](https://python-prompt-toolkit.readthedocs.io/)
- [Rich Documentation](https://rich.readthedocs.io/)
- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Python cmd Module](https://docs.python.org/3/library/cmd.html)

### Real-World Examples
- [ptpython - Full Python REPL](https://github.com/prompt-toolkit/ptpython)
- [IPython - Interactive Python Shell](https://ipython.org/)
- [litecli - SQLite CLI](https://github.com/dbcli/litecli)

### Articles & Tutorials
- Building a REPL in Python (bernsteinbear.com)
- PEP 762 - REPL-acing the default REPL
- Debugging in a REPL (Julia Evans blog)

---

## Conclusion

**Primary Recommendation:** Build Elpis REPL using **prompt_toolkit** as the foundation, with **Rich** for output formatting and **asyncio** for concurrent execution.

This combination provides:
1. Modern, user-friendly interface (completion, multiline, syntax highlighting)
2. Scalability for future enhancements (debugging, inspection, streaming)
3. Separation of concerns (input, processing, output, persistence)
4. Cross-platform compatibility
5. Active community and maintainability

The implementation should follow an async-first architecture with proper error handling, state persistence, and emotional state integration for the full Elpis system.

---

**Report Completed:** January 11, 2026
**Agent:** REPL-Agent
**Status:** Ready for implementation
