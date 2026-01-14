"""Psyche TUI Application built with Textual."""

import asyncio
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
from psyche.memory.server import MemoryServer, ThoughtEvent


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
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+t", "toggle_thoughts", "Thoughts"),
        Binding("escape", "focus_input", "Focus Input", show=False),
    ]

    def __init__(self, memory_server: MemoryServer, *args, **kwargs):
        """
        Initialize the Psyche app.

        Args:
            memory_server: The memory server instance to use
        """
        super().__init__(*args, **kwargs)
        self.memory_server = memory_server
        self._server_task: Optional[asyncio.Task] = None

        # Register callbacks
        self.memory_server.on_token = self._on_token
        self.memory_server.on_thought = self._on_thought
        self.memory_server.on_response = self._on_response
        self.memory_server.on_tool_call = self._on_tool_call

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

    def _on_tool_call(self, name: str, result: Optional[Dict[str, Any]]) -> None:
        """Handle tool execution callback."""
        # Use call_later to safely schedule widget update on Textual's event loop
        def update():
            try:
                tool_widget = self.query_one("#tool-activity", ToolActivity)
                if result is None:
                    tool_widget.add_tool_start(name)
                else:
                    tool_widget.update_tool_complete(name, result)
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
        cmd = command.lower().strip()
        chat = self.query_one("#chat", ChatView)

        if cmd == "/help":
            self._show_help()
        elif cmd == "/status":
            status = await self.memory_server.get_context_summary()
            chat.add_system_message(
                f"State: {status.get('state', 'unknown')}, "
                f"Messages: {status.get('message_count', 0)}, "
                f"Tokens: {status.get('total_tokens', 0)}/{status.get('available_tokens', 0)}"
            )
        elif cmd == "/clear":
            self.memory_server.clear_context()
            chat.clear()
            chat.add_system_message("Context cleared")
        elif cmd in ("/quit", "/exit", "/q"):
            self.exit()
        elif cmd == "/thoughts" or cmd == "/thoughts toggle":
            self.action_toggle_thoughts()
        elif cmd == "/thoughts on":
            self.query_one("#thoughts", ThoughtPanel).show()
        elif cmd == "/thoughts off":
            self.query_one("#thoughts", ThoughtPanel).hide()
        elif cmd == "/emotion":
            emotion = await self.memory_server.client.get_emotion()
            chat.add_system_message(
                f"Emotional state: {emotion.quadrant} "
                f"(v={emotion.valence:.2f}, a={emotion.arousal:.2f})"
            )
        else:
            chat.add_system_message(f"Unknown command: {command}")

    def _show_help(self) -> None:
        """Show help information in chat."""
        chat = self.query_one("#chat", ChatView)
        chat.add_system_message(
            "Commands: /help, /status, /clear, /emotion, /thoughts [on|off], /quit"
        )
        chat.add_system_message(
            "Shortcuts: Ctrl+C (quit), Ctrl+L (clear), Ctrl+T (toggle thoughts)"
        )

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

    async def action_quit(self) -> None:
        """Quit the application with graceful memory consolidation."""
        # Store context and consolidate memories before shutdown
        await self.memory_server.shutdown_with_consolidation()
        await self.memory_server.stop()
        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        self.exit()
