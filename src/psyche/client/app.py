"""Psyche TUI Application built with Textual."""

import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static

from psyche.client.widgets import (
    ChatView,
    Sidebar,
    UserInput,
    UserSubmitted,
    ThoughtPanel,
    ToolActivity,
    EmotionalStateDisplay,
)
from psyche.client.widgets.sidebar import StatusDisplay
from psyche.client.commands import get_command, format_help_text, format_shortcut_help, format_startup_hint
from psyche.memory.server import MemoryServer, ServerState, ThoughtEvent


class PsycheApp(App):
    """
    Psyche TUI Application.

    A Textual-based terminal UI for the Psyche continuous inference agent.
    Features streaming responses, tool activity display, and emotional state tracking.
    """

    TITLE = "Psyche"
    SUB_TITLE = "Continuous Inference Agent"

    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("ctrl+c", "interrupt_or_quit", "Stop/Quit"),
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+t", "toggle_thoughts", "Thoughts"),
        Binding("escape", "focus_input", "Focus Input", show=False),
    ]

    # Double-tap threshold for Ctrl+C quit when idle (seconds)
    DOUBLE_TAP_THRESHOLD = 1.5

    def __init__(self, memory_server: MemoryServer, *args, **kwargs):
        """
        Initialize the Psyche app.

        Args:
            memory_server: The memory server instance to use
        """
        super().__init__(*args, **kwargs)
        self.memory_server = memory_server
        self._server_task: Optional[asyncio.Task] = None

        # Tracking for double-tap Ctrl+C to quit
        self._last_ctrl_c: float = 0.0

        # Register callbacks
        self.memory_server.on_token = self._on_token
        self.memory_server.on_thought = self._on_thought
        self.memory_server.on_response = self._on_response
        self.memory_server.on_tool_call = self._on_tool_call
        self.memory_server.on_thinking = self._on_thinking

    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield Header()

        with Horizontal(id="main-container"):
            yield Sidebar(id="sidebar")

            with Vertical(id="content"):
                yield ChatView(id="chat")
                yield ThoughtPanel(id="thoughts")
                yield UserInput(id="input")

        yield Footer()

    async def on_mount(self) -> None:
        """Start memory server when app mounts."""
        # Focus the input
        self.query_one("#input", UserInput).focus()

        # Show startup hint
        chat = self.query_one("#chat", ChatView)
        chat.add_system_message(format_startup_hint())

        # Small delay to let Textual fully initialize before spawning subprocesses
        # This helps avoid race conditions with event loop setup
        await asyncio.sleep(0.1)

        # Start the memory server in the background
        self._server_task = asyncio.create_task(self._run_server())

        # Start periodic emotional state updates
        self.set_interval(1.0, self._update_emotional_display)

    async def _run_server(self) -> None:
        """Run the memory server as a background task with retry logic."""
        from loguru import logger
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                await self.memory_server.start()
                return  # Server ran and exited normally
            except asyncio.CancelledError:
                # Normal shutdown, don't retry
                return
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                if attempt < max_retries - 1:
                    logger.warning(f"Server connection failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    logger.error(f"Server failed after {max_retries} attempts: {error_msg}")
                    try:
                        chat = self.query_one("#chat", ChatView)
                        chat.add_system_message(f"[CRITICAL] Server error: {e}")
                    except Exception:
                        pass  # Widgets not ready
                    self._server_task = None

    def _on_token(self, token: str) -> None:
        """Handle streaming token callback."""
        # Use call_later to safely schedule widget update on Textual's event loop
        def update():
            try:
                chat = self.query_one("#chat", ChatView)
                if not chat.is_streaming:
                    chat.start_stream()
                chat.append_token(token)
            except Exception:
                pass  # Widget may not exist during shutdown
        self.call_later(update)

    def _on_thought(self, thought: ThoughtEvent) -> None:
        """Handle internal thought callback."""
        # Use call_later to safely schedule widget update on Textual's event loop
        def update():
            try:
                thoughts = self.query_one("#thoughts", ThoughtPanel)
                thoughts.add_thought(thought.content, thought.thought_type)
            except Exception:
                pass  # Widget may not exist during shutdown
        self.call_later(update)

    def _on_response(self, content: str) -> None:
        """Handle response completion callback."""
        # Use call_later to safely schedule widget update on Textual's event loop
        def update():
            try:
                chat = self.query_one("#chat", ChatView)
                if chat.is_streaming:
                    chat.end_stream()
            except Exception:
                pass  # Widget may not exist during shutdown
        self.call_later(update)

    def _on_tool_call(self, name: str, args: Dict[str, Any], result: Optional[Dict[str, Any]]) -> None:
        """Handle tool execution callback."""
        # Use call_later to safely schedule widget update on Textual's event loop
        def update():
            try:
                tool_widget = self.query_one("#tool-activity", ToolActivity)
                if result is None:
                    tool_widget.add_tool_start(name, args)
                else:
                    tool_widget.update_tool_complete(name, result)
            except Exception:
                pass  # Widget may not exist during shutdown
        self.call_later(update)

    def _on_thinking(self, token: str) -> None:
        """Handle streaming token during idle thought generation."""
        # Use call_later to safely schedule widget update on Textual's event loop
        def update():
            try:
                thoughts = self.query_one("#thoughts", ThoughtPanel)
                thoughts.on_thinking_token(token)
            except Exception:
                pass  # Widget may not exist during shutdown
        self.call_later(update)

    async def _update_emotional_display(self) -> None:
        """Periodically update emotional state display."""
        # Check if server task is still alive
        if self._server_task is None or self._server_task.done():
            # Server died - update display to show disconnected state
            status_display = self.query_one("#status-display", StatusDisplay)
            status_display.state = "disconnected"
            return

        # Check connection state before making calls
        if not self.memory_server.client.is_connected:
            return

        try:
            emotion = await self.memory_server.client.get_emotion()
            display = self.query_one("#emotion-display", EmotionalStateDisplay)
            display.valence = emotion.valence
            display.arousal = emotion.arousal
            display.quadrant = emotion.quadrant

            # Also update status
            status = await self.memory_server.get_context_summary()
            status_display = self.query_one("#status-display", StatusDisplay)
            status_display.state = status.get("state", "idle")
            status_display.message_count = status.get("message_count", 0)
            status_display.token_usage = (
                f"{status.get('total_tokens', 0)}/{status.get('available_tokens', 0)}"
            )
        except Exception:
            pass  # Ignore errors during status updates

    async def on_user_submitted(self, message: UserSubmitted) -> None:
        """Handle user input submission."""
        if message.is_command:
            await self._handle_command(message.value)
        else:
            # Add user message to chat
            chat = self.query_one("#chat", ChatView)
            chat.add_user_message(message.value)

            # Submit to memory server
            self.memory_server.submit_input(message.value)

    async def _handle_command(self, command: str) -> None:
        """Handle slash commands."""
        cmd_text = command.lower().strip()
        chat = self.query_one("#chat", ChatView)

        # Remove leading slash and extract command name + args
        if cmd_text.startswith("/"):
            cmd_text = cmd_text[1:]
        parts = cmd_text.split(maxsplit=1)
        cmd_name = parts[0] if parts else ""
        cmd_args = parts[1] if len(parts) > 1 else ""

        # Look up command by name or alias
        cmd = get_command(cmd_name)

        if cmd is None:
            chat.add_system_message(f"Unknown command: {command}")
            return

        # Execute the command based on its canonical name
        if cmd.name == "help":
            self._show_help()
        elif cmd.name == "status":
            status = await self.memory_server.get_context_summary()
            chat.add_system_message(
                f"State: {status.get('state', 'unknown')}, "
                f"Messages: {status.get('message_count', 0)}, "
                f"Tokens: {status.get('total_tokens', 0)}/{status.get('available_tokens', 0)}"
            )
        elif cmd.name == "clear":
            self.memory_server.clear_context()
            chat.clear()
            chat.add_system_message("Context cleared")
        elif cmd.name == "quit":
            await self.action_quit()
        elif cmd.name == "thoughts":
            # Handle subcommands
            if cmd_args == "on":
                self.query_one("#thoughts", ThoughtPanel).show()
            elif cmd_args == "off":
                self.query_one("#thoughts", ThoughtPanel).hide()
            else:
                self.action_toggle_thoughts()
        elif cmd.name == "emotion":
            emotion = await self.memory_server.client.get_emotion()
            chat.add_system_message(
                f"Emotional state: {emotion.quadrant} "
                f"(v={emotion.valence:.2f}, a={emotion.arousal:.2f})"
            )

    def _show_help(self) -> None:
        """Show help information in chat."""
        chat = self.query_one("#chat", ChatView)
        chat.add_system_message(format_help_text())
        chat.add_system_message(format_shortcut_help())

    def action_toggle_thoughts(self) -> None:
        """Toggle thought panel visibility."""
        thoughts = self.query_one("#thoughts", ThoughtPanel)
        thoughts.toggle()

    def action_clear(self) -> None:
        """Clear chat and context."""
        self.memory_server.clear_context()
        chat = self.query_one("#chat", ChatView)
        chat.clear()
        chat.add_system_message("Context cleared")

    def action_focus_input(self) -> None:
        """Focus the input widget."""
        self.query_one("#input", UserInput).focus()

    async def action_interrupt_or_quit(self) -> None:
        """
        Handle Ctrl+C: interrupt generation if running, or quit on double-tap when idle.

        Behavior:
        - If generating: interrupt and show "[Interrupted]"
        - If idle (first tap): show "Press again to quit"
        - If idle (second tap within threshold): quit
        """
        now = time.time()

        # If the server is generating, interrupt it
        if self.memory_server.state == ServerState.THINKING:
            interrupted = self.memory_server.interrupt()
            if interrupted:
                self.notify("Generation interrupted", severity="warning")
            return

        # Not generating - implement double-tap to quit
        time_since_last = now - self._last_ctrl_c
        if time_since_last < self.DOUBLE_TAP_THRESHOLD:
            # Double-tap detected - quit
            await self.action_quit()
        else:
            # First tap - show message and record time
            self._last_ctrl_c = now
            self.notify("Press Ctrl+C again to quit, or Ctrl+Q", severity="information")

    async def action_quit(self) -> None:
        """Quit the application with graceful memory consolidation."""
        # Show visual feedback
        self.notify("Shutting down...", severity="information", timeout=10)
        try:
            chat = self.query_one("#chat", ChatView)
            chat.add_system_message("Consolidating memories and shutting down...")
        except Exception:
            pass  # Widget may not exist

        # Store context and consolidate memories before shutdown
        await self.memory_server.shutdown_with_consolidation()

        self.notify("Memories saved. Goodbye!", severity="information", timeout=2)

        await self.memory_server.stop()
        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        self.exit()
