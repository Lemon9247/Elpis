"""Interactive REPL for Psyche."""

import asyncio
from pathlib import Path
from typing import Optional

from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from psyche.client.display import DisplayManager, DisplayConfig
from psyche.memory.server import MemoryServer, ThoughtEvent


class PsycheREPL:
    """
    Interactive REPL (Read-Eval-Print Loop) for Psyche.

    Provides a command-line interface for interacting with the memory server,
    displaying responses, and managing the conversation.
    """

    def __init__(
        self,
        server: MemoryServer,
        history_file: Optional[str] = None,
        display_config: Optional[DisplayConfig] = None,
    ):
        """
        Initialize the REPL.

        Args:
            server: Memory server instance
            history_file: Path to command history file
            display_config: Display configuration
        """
        self.server = server
        self.display = DisplayManager(display_config)

        # Set up command history
        history_path = history_file or str(Path.home() / ".psyche_history")
        self.session = PromptSession(history=FileHistory(history_path))

        # Register callbacks
        self.server.on_thought = self._on_thought
        self.server.on_response = self._on_response

        self._waiting_for_response = False

    def _on_thought(self, thought: ThoughtEvent) -> None:
        """Handle internal thought events."""
        self.display.print_thought(thought)

    def _on_response(self, content: str) -> None:
        """Handle response events."""
        self.display.stop_thinking_indicator()
        self.display.print_response(content)
        self._waiting_for_response = False

    async def run(self) -> None:
        """Run the REPL main loop."""
        self.display.print_welcome()

        # Start the server in the background
        server_task = asyncio.create_task(self.server.start())

        try:
            while self.server.is_running:
                try:
                    # Get user input
                    user_input = await self._get_input()

                    if user_input is None:
                        continue

                    # Check for commands
                    if user_input.startswith("/"):
                        handled = await self._handle_command(user_input)
                        if handled:
                            continue

                    # Submit input to server
                    self.display.start_thinking_indicator()
                    self._waiting_for_response = True
                    self.server.submit_input(user_input)

                    # Wait for response
                    while self._waiting_for_response and self.server.is_running:
                        await asyncio.sleep(0.1)

                except EOFError:
                    logger.info("EOF received, shutting down")
                    break
                except KeyboardInterrupt:
                    logger.info("Interrupt received")
                    break

        finally:
            await self.server.stop()
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    async def _get_input(self) -> Optional[str]:
        """Get user input from the prompt."""
        try:
            # Get emotional state for prompt display
            emotional_quadrant = None
            if self.display.config.show_emotional_state:
                try:
                    emotion = await self.server.client.get_emotion()
                    emotional_quadrant = emotion.quadrant
                except Exception:
                    pass

            prompt = self.display.get_prompt(emotional_quadrant)
            user_input = await self.session.prompt_async(prompt)
            return user_input.strip() if user_input else None
        except EOFError:
            raise
        except Exception as e:
            logger.warning(f"Input error: {e}")
            return None

    async def _handle_command(self, command: str) -> bool:
        """
        Handle a REPL command.

        Args:
            command: Command string (starts with /)

        Returns:
            True if command was handled, False otherwise
        """
        cmd = command.lower().strip()

        if cmd == "/help":
            self.display.print_help()
            return True

        elif cmd == "/status":
            try:
                status = await self.server.get_context_summary()
                self.display.print_status(status)
            except Exception as e:
                self.display.print_error(f"Could not get status: {e}")
            return True

        elif cmd == "/clear":
            self.server.clear_context()
            self.display.print_info("Context cleared")
            return True

        elif cmd == "/emotion":
            try:
                emotion = await self.server.client.get_emotion()
                self.display.print_info(
                    f"Emotional state: {emotion.quadrant} "
                    f"(v={emotion.valence:.2f}, a={emotion.arousal:.2f})"
                )
            except Exception as e:
                self.display.print_error(f"Could not get emotion: {e}")
            return True

        elif cmd in ("/quit", "/exit", "/q"):
            await self.server.stop()
            return True

        elif cmd == "/thoughts on":
            self.display.config.show_thoughts = True
            self.display.print_info("Internal thoughts enabled")
            return True

        elif cmd == "/thoughts off":
            self.display.config.show_thoughts = False
            self.display.print_info("Internal thoughts disabled")
            return True

        else:
            self.display.print_error(f"Unknown command: {command}")
            return True
