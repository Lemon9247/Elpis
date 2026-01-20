"""
Hermes TUI Application - Terminal interface for Psyche.

A Textual-based terminal UI for interacting with Psyche in remote mode.
Connects to a running Psyche server via HTTP and executes tools locally.

Features:
- Streaming responses with chat view
- Tool activity display with approval workflow
- Emotional state tracking (valence-arousal)
- Slash commands (/help, /status, /emotion, etc.)
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from enum import Enum

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer

from loguru import logger

from hermes.widgets import (
    ChatView,
    Sidebar,
    UserInput,
    UserSubmitted,
    ThoughtPanel,
    ToolActivity,
    EmotionalStateDisplay,
)
from hermes.widgets.sidebar import StatusDisplay
from hermes.commands import get_command, format_help_text, format_shortcut_help, format_startup_hint
from hermes.handlers import PsycheClient, RemotePsycheClient
from psyche.tools.tool_engine import ToolEngine


@dataclass
class ThoughtEvent:
    """Event representing an internal thought."""
    content: str
    thought_type: str  # "reflection", "planning", "memory", "idle"
    triggered_by: Optional[str] = None


class AppState(Enum):
    """Application states."""

    IDLE = "idle"
    PROCESSING = "processing"
    DISCONNECTED = "disconnected"


class Hermes(App):
    """
    Hermes TUI Application.

    A Textual-based terminal UI for the Psyche continuous inference agent.
    Features streaming responses, tool activity display, and emotional state tracking.

    Connects to a remote Psyche server and executes file/bash/search tools locally.
    """

    TITLE = "Hermes"
    SUB_TITLE = "Psyche TUI Client"

    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("ctrl+c", "interrupt_or_quit", "Stop/Quit"),
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+t", "toggle_thoughts", "Thoughts"),
        Binding("ctrl+r", "toggle_reasoning", "Reasoning", show=False),
        Binding("escape", "focus_input", "Focus Input", show=False),
    ]

    # Double-tap threshold for Ctrl+C quit when idle (seconds)
    DOUBLE_TAP_THRESHOLD = 1.5

    def __init__(
        self,
        client: Optional[PsycheClient] = None,
        tool_engine: Optional[ToolEngine] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the Hermes app.

        Args:
            client: PsycheClient for connecting to the Psyche server
            tool_engine: ToolEngine for local tool execution
        """
        super().__init__(*args, **kwargs)

        self._client = client
        self._tool_engine = tool_engine

        # Application state
        self._state = AppState.IDLE
        self._is_processing = False

        # Tracking for double-tap Ctrl+C to quit
        self._last_ctrl_c: float = 0.0

    @property
    def state(self) -> AppState:
        """Get current application state."""
        return self._state

    @property
    def is_processing(self) -> bool:
        """Check if currently processing user input."""
        return self._is_processing

    @property
    def reasoning_enabled(self) -> bool:
        """Check if reasoning mode is enabled."""
        return self._client.reasoning_enabled if self._client else False

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
        """Start services when app mounts."""
        # Focus the input
        self.query_one("#input", UserInput).focus()

        # Show startup hint
        chat = self.query_one("#chat", ChatView)
        chat.add_system_message(format_startup_hint())

        # Small delay to let Textual fully initialize
        await asyncio.sleep(0.1)

        # Connect to server
        if self._client and isinstance(self._client, RemotePsycheClient):
            try:
                await self._client.connect()
                logger.info(f"Connected to Psyche server at {self._client.base_url}")
            except Exception as e:
                logger.error(f"Failed to connect to server: {e}")
                chat.add_system_message(f"[Error] Failed to connect to server: {e}")
                self._state = AppState.DISCONNECTED

        # Start periodic emotional state updates
        self.set_interval(1.0, self._update_emotional_display)

    def _on_token(self, token: str) -> None:
        """Handle streaming token callback."""

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

        def update():
            try:
                thoughts = self.query_one("#thoughts", ThoughtPanel)
                thoughts.add_thought(thought.content, thought.thought_type)
            except Exception:
                pass  # Widget may not exist during shutdown

        self.call_later(update)

    def _on_response(self, content: str) -> None:
        """Handle response completion callback."""

        def update():
            try:
                chat = self.query_one("#chat", ChatView)
                if chat.is_streaming:
                    chat.end_stream()
            except Exception:
                pass  # Widget may not exist during shutdown

        self.call_later(update)

    def _on_tool_call(
        self, name: str, args: Dict[str, Any], result: Optional[Dict[str, Any]]
    ) -> None:
        """Handle tool execution callback."""

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

    async def _update_emotional_display(self) -> None:
        """Periodically update emotional state display."""
        if not self._client:
            return

        try:
            emotion = await self._client.get_emotion()
            display = self.query_one("#emotion-display", EmotionalStateDisplay)
            display.valence = emotion.get("valence", 0.0)
            display.arousal = emotion.get("arousal", 0.0)
            display.quadrant = emotion.get("quadrant", "neutral")

            # Also update status
            status = self._client.context_summary
            status_display = self.query_one("#status-display", StatusDisplay)
            status_display.state = self._state.value
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

            # Process through client
            await self._process_user_input(message.value)

    async def _process_user_input(self, text: str) -> None:
        """Process user input through the remote client."""
        if not self._client:
            logger.error("No client configured")
            chat = self.query_one("#chat", ChatView)
            chat.add_system_message("[Error: Not connected]")
            return

        self._state = AppState.PROCESSING
        self._is_processing = True

        try:
            await self._process_via_client(text)
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            chat = self.query_one("#chat", ChatView)
            chat.add_system_message(f"[Error: {e}]")
        finally:
            self._state = AppState.IDLE
            self._is_processing = False

    async def _process_via_client(self, text: str) -> None:
        """Process input via remote client with tool execution loop."""
        import json

        MAX_ITERATIONS = 10
        chat = self.query_one("#chat", ChatView)

        # Add user message to client history
        await self._client.add_user_message(text)

        for iteration in range(MAX_ITERATIONS):
            # Stream response and display
            chat.start_stream()
            full_response = ""

            try:
                async for token in self._client.generate_stream():
                    full_response += token
                    chat.append_token(token)
            except Exception as e:
                chat.end_stream()
                raise

            chat.end_stream()

            # Check for tool calls
            tool_calls = self._client.get_pending_tool_calls()

            if not tool_calls or not self._tool_engine:
                # No tools to execute - add response to history and we're done
                if full_response:
                    await self._client.add_assistant_message(full_response, user_message=text)
                break

            # Add assistant message with tool calls to history
            # (The server expects the full conversation history including the tool-calling message)
            if full_response:
                await self._client.add_assistant_message(full_response, user_message=text)

            # Execute each tool locally
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                args_str = func.get("arguments", "{}")

                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}

                # UI: show tool start
                self._on_tool_call(name, args, None)

                # Execute tool
                try:
                    result = await self._tool_engine.execute_tool_call(tc)
                except Exception as e:
                    result = {"success": False, "error": str(e)}

                # UI: show tool complete
                self._on_tool_call(name, args, result)

                # Send result back to server
                self._client.add_tool_result(name, json.dumps(result))

            # Loop continues - server will generate next response with tool results

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
            await self._show_status()
        elif cmd.name == "clear":
            self._clear_context()
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
            await self._show_emotion()
        elif cmd.name == "thinking":
            await self._toggle_reasoning(cmd_args)

    async def _show_status(self) -> None:
        """Show status information."""
        chat = self.query_one("#chat", ChatView)

        if self._client:
            status = self._client.context_summary
            chat.add_system_message(
                f"State: {self._state.value}, "
                f"Messages: {status.get('message_count', 0)}, "
                f"Server: {status.get('server_url', 'unknown')}"
            )
        else:
            chat.add_system_message("Status unavailable: no client configured")

    def _clear_context(self) -> None:
        """Clear conversation context."""
        chat = self.query_one("#chat", ChatView)

        if self._client:
            self._client.clear_context()

        chat.clear()
        chat.add_system_message("Context cleared")

    async def _show_emotion(self) -> None:
        """Show emotional state."""
        chat = self.query_one("#chat", ChatView)

        if self._client:
            emotion = await self._client.get_emotion()
            chat.add_system_message(
                f"Emotional state: {emotion.get('quadrant', 'neutral')} "
                f"(v={emotion.get('valence', 0.0):.2f}, a={emotion.get('arousal', 0.0):.2f})"
            )
        else:
            chat.add_system_message("Emotion unavailable: no client configured")

    async def _toggle_reasoning(self, cmd_args: str) -> None:
        """Toggle or set reasoning mode."""
        chat = self.query_one("#chat", ChatView)

        if cmd_args.lower() in ("on", "true", "1"):
            if self._client:
                self._client.set_reasoning_mode(True)
            chat.add_system_message("[dim]Reasoning display enabled[/]")
        elif cmd_args.lower() in ("off", "false", "0"):
            if self._client:
                self._client.set_reasoning_mode(False)
            chat.add_system_message("[dim]Reasoning display disabled[/]")
        else:
            # Toggle current state
            new_state = not self.reasoning_enabled
            if self._client:
                self._client.set_reasoning_mode(new_state)
            status = "enabled" if new_state else "disabled"
            chat.add_system_message(f"[dim]Reasoning display {status}[/]")

    def _show_help(self) -> None:
        """Show help information in chat."""
        chat = self.query_one("#chat", ChatView)
        chat.add_system_message(format_help_text())
        chat.add_system_message(format_shortcut_help())

    def action_toggle_thoughts(self) -> None:
        """Toggle thought panel visibility."""
        thoughts = self.query_one("#thoughts", ThoughtPanel)
        thoughts.toggle()

    def action_toggle_reasoning(self) -> None:
        """Toggle reasoning display mode."""
        # Use the command handler for consistency
        asyncio.create_task(self._handle_command("/thinking"))

    def action_clear(self) -> None:
        """Clear chat and context."""
        self._clear_context()

    def action_focus_input(self) -> None:
        """Focus the input widget."""
        self.query_one("#input", UserInput).focus()

    async def action_interrupt_or_quit(self) -> None:
        """
        Handle Ctrl+C: quit on double-tap when idle.

        Behavior:
        - If idle (first tap): show "Press again to quit"
        - If idle (second tap within threshold): quit
        """
        now = time.time()

        # If processing, just notify (can't interrupt remote generation easily)
        if self._is_processing:
            self.notify("Processing... please wait", severity="information")
            return

        # Not processing - implement double-tap to quit
        time_since_last = now - self._last_ctrl_c
        if time_since_last < self.DOUBLE_TAP_THRESHOLD:
            # Double-tap detected - quit
            await self.action_quit()
        else:
            # First tap - show message and record time
            self._last_ctrl_c = now
            self.notify("Press Ctrl+C again to quit, or Ctrl+Q", severity="information")

    async def action_quit(self) -> None:
        """Quit the application."""
        # Show visual feedback
        self.notify("Shutting down...", severity="information", timeout=10)
        try:
            chat = self.query_one("#chat", ChatView)
            chat.add_system_message("Disconnecting from server...")
        except Exception:
            pass  # Widget may not exist

        # Disconnect from server
        if self._client:
            await self._client.shutdown()

        self.notify("Goodbye!", severity="information", timeout=2)

        self.exit()
