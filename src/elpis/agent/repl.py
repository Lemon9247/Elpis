"""
Interactive async REPL interface for Elpis.

Provides a user-friendly command-line interface using prompt_toolkit
for input handling and Rich for beautiful output formatting.
"""

import asyncio
from pathlib import Path
from typing import Optional

from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from elpis.agent.orchestrator import AgentOrchestrator


class ElpisREPL:
    """
    Interactive async REPL (Read-Eval-Print Loop) interface.

    Features:
    - Async input handling with prompt_toolkit
    - Rich text formatting for beautiful output
    - Command history persistence
    - Special commands: /help, /clear, /exit
    - Markdown and code syntax highlighting
    """

    def __init__(
        self,
        agent: AgentOrchestrator,
        history_file: str = ".elpis_history",
    ):
        """
        Initialize the REPL interface.

        Args:
            agent: Agent orchestrator instance
            history_file: Path to command history file
        """
        self.agent = agent
        self.console = Console()

        # Create history file path
        history_path = Path.home() / history_file

        # Initialize prompt session with history
        self.session = PromptSession(history=FileHistory(str(history_path)))

        logger.info(f"REPL initialized with history at {history_path}")

    async def run(self) -> None:
        """
        Main REPL loop.

        Continuously prompts for user input, processes it through the agent,
        and displays formatted responses. Handles special commands and
        graceful shutdown.
        """
        self._display_welcome()

        while True:
            try:
                # Get user input asynchronously
                user_input = await self.session.prompt_async(
                    "elpis> ", multiline=False
                )

                # Skip empty input
                if not user_input.strip():
                    continue

                # Handle special commands
                if user_input.startswith("/"):
                    should_continue = await self._handle_special_command(user_input)
                    if not should_continue:
                        break
                    continue

                # Process with agent
                self.console.print("[dim]Thinking...[/dim]")
                response = await self.agent.process(user_input)

                # Display response with formatting
                self._display_response(response)

            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                self.console.print("\n[yellow]Interrupted (use /exit to quit)[/yellow]")
                continue

            except EOFError:
                # Handle Ctrl+D
                self.console.print("\n[cyan]Goodbye![/cyan]")
                break

            except Exception as e:
                # Handle unexpected errors
                logger.exception(f"Error in REPL loop: {e}")
                self.console.print(f"[red]Error: {e}[/red]")

    def _display_welcome(self) -> None:
        """Display welcome banner with usage information."""
        welcome_text = """# Elpis - Emotional Coding Agent

**Phase 1: Basic Agent Harness**

Welcome! I'm here to help you with coding tasks.

## Available Commands

- `/help` - Show this help message
- `/clear` - Clear conversation history
- `/exit` - Exit the REPL

## Usage

Just type naturally to interact with me. For example:
- "Read the README.md file"
- "Write a hello world function in Python"
- "Search for TODO comments in the codebase"

Let's get started!
"""
        self.console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="blue"))

    async def _handle_special_command(self, command: str) -> bool:
        """
        Handle special REPL commands starting with /.

        Args:
            command: Command string (e.g., "/help", "/clear")

        Returns:
            True to continue REPL loop, False to exit
        """
        command = command.strip().lower()

        if command == "/help":
            self._display_welcome()
            return True

        elif command == "/clear":
            self.agent.clear_history()
            self.console.print("[green]Conversation history cleared[/green]")
            return True

        elif command == "/exit" or command == "/quit":
            self.console.print("[cyan]Goodbye![/cyan]")
            return False

        elif command == "/status":
            # Hidden command for debugging
            history_len = self.agent.get_history_length()
            last_msg = self.agent.get_last_message()
            self.console.print(f"[dim]History length: {history_len}[/dim]")
            if last_msg:
                self.console.print(f"[dim]Last message role: {last_msg.get('role')}[/dim]")
            return True

        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("[dim]Available commands: /help, /clear, /exit[/dim]")
            return True

    def _display_response(self, response: str) -> None:
        """
        Display agent response with appropriate formatting.

        Automatically detects and renders:
        - Markdown (if contains markdown syntax)
        - Code blocks (with syntax highlighting)
        - Plain text (in a panel)

        Args:
            response: Agent's response text
        """
        try:
            # Check if response contains markdown formatting
            has_markdown = any(
                marker in response
                for marker in ["```", "**", "*", "#", "- ", "1.", "[", "]"]
            )

            if has_markdown:
                # Render as markdown
                self.console.print(Panel(Markdown(response), title="Elpis", border_style="green"))
            else:
                # Render as plain text in a panel
                self.console.print(Panel(response, title="Elpis", border_style="green"))

        except Exception as e:
            # Fallback to simple output if formatting fails
            logger.warning(f"Failed to format response: {e}")
            self.console.print(response)

    def display_error(self, error_message: str) -> None:
        """
        Display an error message with red formatting.

        Args:
            error_message: Error message to display
        """
        self.console.print(Panel(error_message, title="Error", border_style="red"))

    def display_info(self, info_message: str) -> None:
        """
        Display an informational message.

        Args:
            info_message: Info message to display
        """
        self.console.print(f"[cyan]{info_message}[/cyan]")

    def display_success(self, success_message: str) -> None:
        """
        Display a success message with green formatting.

        Args:
            success_message: Success message to display
        """
        self.console.print(f"[green]{success_message}[/green]")
