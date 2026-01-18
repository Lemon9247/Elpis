"""Hermes TUI Application built with Textual."""

import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

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
from psyche.handlers import PsycheClient, ReactHandler
from psyche.handlers import IdleHandler, ThoughtEvent

if TYPE_CHECKING:
    from psyche.mcp.client import ElpisClient, MnemosyneClient


class AppState(Enum):
    """Application states."""

    IDLE = "idle"
    PROCESSING = "processing"
    THINKING = "thinking"
    DISCONNECTED = "disconnected"


class Hermes(App):
    """
    Hermes TUI Application.

    A Textual-based terminal UI for the Psyche continuous inference agent.
    Features streaming responses, tool activity display, and emotional state tracking.
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
        react_handler: Optional[ReactHandler] = None,
        idle_handler: Optional[IdleHandler] = None,
        elpis_client: Optional["ElpisClient"] = None,
        mnemosyne_client: Optional["MnemosyneClient"] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the Hermes app.

        Args:
            client: PsycheClient for memory/inference operations
            react_handler: ReactHandler for user input processing
            idle_handler: IdleHandler for idle thinking
            elpis_client: ElpisClient for inference
            mnemosyne_client: MnemosyneClient for memory (optional)
        """
        super().__init__(*args, **kwargs)

        self._client = client
        self._react_handler = react_handler
        self._idle_handler = idle_handler
        self._elpis_client = elpis_client
        self._mnemosyne_client = mnemosyne_client

        # Application state
        self._state = AppState.IDLE
        self._idle_task: Optional[asyncio.Task] = None
        self._connection_task: Optional[asyncio.Task] = None

        # Tracking for double-tap Ctrl+C to quit
        self._last_ctrl_c: float = 0.0

    @property
    def state(self) -> AppState:
        """Get current application state."""
        return self._state

    @property
    def is_processing(self) -> bool:
        """Check if currently processing user input."""
        return self._react_handler.is_processing if self._react_handler else False

    @property
    def is_thinking(self) -> bool:
        """Check if currently in idle thinking mode."""
        return self._idle_handler.is_thinking if self._idle_handler else False

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

        # Small delay to let Textual fully initialize before spawning subprocesses
        await asyncio.sleep(0.1)

        # Connect clients and start services
        self._connection_task = asyncio.create_task(self._run_connected())

        # Start periodic emotional state updates
        self.set_interval(1.0, self._update_emotional_display)

    async def _run_connected(self) -> None:
        """Run with client connections and idle loop."""
        if not self._elpis_client:
            logger.error("ElpisClient not configured")
            return

        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Connect to Elpis
                async with self._elpis_client.connect():
                    logger.info("Connected to Elpis inference server")

                    # Optionally connect to Mnemosyne
                    if self._mnemosyne_client:
                        async with self._mnemosyne_client.connect():
                            logger.info("Connected to Mnemosyne memory server")
                            await self._run_connected_loop()
                    else:
                        await self._run_connected_loop()

                return  # Normal exit

            except asyncio.CancelledError:
                # Normal shutdown
                return
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Connection failed (attempt {attempt + 1}/{max_retries}): {error_msg}"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Connection failed after {max_retries} attempts: {error_msg}")
                    try:
                        chat = self.query_one("#chat", ChatView)
                        chat.add_system_message(f"[CRITICAL] Connection error: {e}")
                    except Exception:
                        pass
                    self._state = AppState.DISCONNECTED

    async def _run_connected_loop(self) -> None:
        """Run the main loop while clients are connected."""
        # Start the idle thinking loop
        if self._idle_handler:
            self._idle_task = asyncio.create_task(self._run_idle_loop())

        # Keep running until cancelled
        try:
            while True:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            # Cancel idle task on shutdown
            if self._idle_task and not self._idle_task.done():
                self._idle_task.cancel()
                try:
                    await self._idle_task
                except asyncio.CancelledError:
                    pass

    async def _run_idle_loop(self) -> None:
        """Run the idle thinking loop."""
        if not self._idle_handler:
            return

        # Get idle think interval from handler config
        idle_interval = self._idle_handler.config.post_interaction_delay

        try:
            while True:
                # Wait for idle interval
                await asyncio.sleep(idle_interval / 2)

                # Skip if processing or not ready for idle thinking
                if self.is_processing or not self._idle_handler.can_start_thinking():
                    continue

                # Generate idle thought
                try:
                    self._state = AppState.THINKING
                    await self._idle_handler.generate_thought(
                        on_token=self._on_thinking,
                        on_tool_call=self._on_tool_call,
                        on_thought=self._on_thought,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Idle thinking error: {e}")
                finally:
                    self._state = AppState.IDLE

        except asyncio.CancelledError:
            pass

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

    def _on_thinking(self, token: str) -> None:
        """Handle streaming token during idle thought generation."""

        def update():
            try:
                thoughts = self.query_one("#thoughts", ThoughtPanel)
                thoughts.on_thinking_token(token)
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

            # Process through ReactHandler
            await self._process_user_input(message.value)

    async def _process_user_input(self, text: str) -> None:
        """Process user input through ReactHandler or client directly."""
        # Record user interaction for idle timing
        if self._idle_handler:
            self._idle_handler.record_user_interaction()

        # Interrupt idle thinking if active
        if self._idle_handler and self._idle_handler.is_thinking:
            self._idle_handler.interrupt()

        self._state = AppState.PROCESSING

        try:
            if self._react_handler:
                # Local mode: use ReactHandler
                await self._react_handler.process_input(
                    text,
                    on_token=self._on_token,
                    on_tool_call=self._on_tool_call,
                    on_response=self._on_response,
                    on_thought=self._on_thought,
                )
            elif self._client:
                # Remote mode: use client directly
                await self._process_via_client(text)
            else:
                logger.error("No handler or client configured")
                chat = self.query_one("#chat", ChatView)
                chat.add_system_message("[Error: Not connected]")
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            chat = self.query_one("#chat", ChatView)
            chat.add_system_message(f"[Error: {e}]")
        finally:
            self._state = AppState.IDLE

    async def _process_via_client(self, text: str) -> None:
        """Process input via remote client (remote mode)."""
        chat = self.query_one("#chat", ChatView)

        # Add user message to client
        await self._client.add_user_message(text)

        # Start streaming response
        chat.start_assistant_message()

        full_response = ""
        try:
            async for token in self._client.generate_stream():
                full_response += token
                chat.append_to_assistant(token)

            # Finalize the message
            chat.finalize_assistant_message()

            # Add response to client history
            await self._client.add_assistant_message(full_response, user_message=text)

        except Exception as e:
            chat.finalize_assistant_message()
            raise

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
                f"Tokens: {status.get('total_tokens', 0)}/{status.get('available_tokens', 0)}"
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
        Handle Ctrl+C: interrupt generation if running, or quit on double-tap when idle.

        Behavior:
        - If generating: interrupt and show "[Interrupted]"
        - If idle (first tap): show "Press again to quit"
        - If idle (second tap within threshold): quit
        """
        now = time.time()

        # Check if actively processing
        is_active = self.is_processing or self.is_thinking

        if is_active:
            # Interrupt the active process
            interrupted = False
            if self._react_handler and self._react_handler.is_processing:
                interrupted = self._react_handler.interrupt()
            elif self._idle_handler and self._idle_handler.is_thinking:
                interrupted = self._idle_handler.interrupt()

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

        # Graceful shutdown
        if self._client:
            await self._client.shutdown()

        self.notify("Memories saved. Goodbye!", severity="information", timeout=2)

        # Cancel background tasks
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass

        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass

        self.exit()
