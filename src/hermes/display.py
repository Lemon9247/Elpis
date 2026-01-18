"""Display manager for rendering output to the terminal."""

from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from psyche.handlers import ThoughtEvent


@dataclass
class DisplayConfig:
    """Configuration for the display manager."""

    show_thoughts: bool = True  # Show internal thoughts
    show_emotional_state: bool = False  # Show emotional state in prompt
    markdown_responses: bool = True  # Render responses as markdown
    thought_style: str = "dim italic"  # Style for thought display


class DisplayManager:
    """
    Manages terminal output display for Hermes.

    Handles:
    - User input prompting
    - Response rendering (with optional markdown)
    - Internal thought display
    - Status indicators (thinking, processing, etc.)
    """

    def __init__(self, config: Optional[DisplayConfig] = None):
        """
        Initialize the display manager.

        Args:
            config: Display configuration
        """
        self.config = config or DisplayConfig()
        self.console = Console()
        self._live: Optional[Live] = None
        self._current_status: Optional[str] = None

    def print_welcome(self) -> None:
        """Print welcome message."""
        welcome = Panel(
            Text.from_markup(
                "[bold cyan]Hermes[/] - Psyche TUI Client\n"
                "[dim]Connected to Elpis inference server[/]\n\n"
                "Type your message and press Enter.\n"
                "Commands: /help, /status, /clear, /quit"
            ),
            title="Welcome",
            border_style="cyan",
        )
        self.console.print(welcome)
        self.console.print()

    def print_response(self, content: str) -> None:
        """
        Print an assistant response.

        Args:
            content: Response content
        """
        if self.config.markdown_responses:
            self.console.print(Markdown(content))
        else:
            self.console.print(content)
        self.console.print()

    def print_thought(self, thought: ThoughtEvent) -> None:
        """
        Print an internal thought.

        Args:
            thought: The thought event
        """
        if not self.config.show_thoughts:
            return

        thought_text = Text()
        thought_text.append(f"[{thought.thought_type}] ", style="dim cyan")
        thought_text.append(thought.content, style=self.config.thought_style)

        self.console.print(thought_text)
        self.console.print()

    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"[bold red]Error:[/] {message}")

    def print_info(self, message: str) -> None:
        """Print an info message."""
        self.console.print(f"[dim]{message}[/]")

    def print_status(self, status: dict) -> None:
        """
        Print server status information.

        Args:
            status: Status dictionary from server
        """
        status_panel = Panel(
            Text.from_markup(
                f"[bold]Server State:[/] {status.get('state', 'unknown')}\n"
                f"[bold]Messages:[/] {status.get('message_count', 0)}\n"
                f"[bold]Tokens:[/] {status.get('total_tokens', 0)}/{status.get('available_tokens', 0)}\n"
                f"[bold]Idle Thoughts:[/] {status.get('idle_thought_count', 0)}\n"
                f"[bold]Emotion:[/] {status.get('emotional_state', {}).get('quadrant', 'neutral')}"
            ),
            title="Status",
            border_style="blue",
        )
        self.console.print(status_panel)
        self.console.print()

    def start_thinking_indicator(self) -> None:
        """Show a thinking indicator."""
        self._current_status = "thinking"
        spinner = Spinner("dots", text="Thinking...", style="cyan")
        self._live = Live(spinner, console=self.console, refresh_per_second=10)
        self._live.start()

    def stop_thinking_indicator(self) -> None:
        """Stop the thinking indicator."""
        if self._live:
            self._live.stop()
            self._live = None
        self._current_status = None

    def get_prompt(self, emotional_quadrant: Optional[str] = None) -> str:
        """
        Get the input prompt string.

        Args:
            emotional_quadrant: Current emotional quadrant for display

        Returns:
            Formatted prompt string
        """
        if self.config.show_emotional_state and emotional_quadrant:
            emotion_indicator = {
                "excited": "[bright_green]^[/]",
                "frustrated": "[red]![/]",
                "calm": "[blue]~[/]",
                "depleted": "[dim]v[/]",
            }.get(emotional_quadrant, "")
            return f"{emotion_indicator} [bold cyan]>[/] "
        return "[bold cyan]>[/] "

    def print_help(self) -> None:
        """Print help information."""
        help_text = Panel(
            Text.from_markup(
                "[bold]Available Commands:[/]\n\n"
                "[cyan]/help[/]    - Show this help message\n"
                "[cyan]/status[/]  - Show server status\n"
                "[cyan]/clear[/]   - Clear conversation context\n"
                "[cyan]/emotion[/] - Show emotional state\n"
                "[cyan]/quit[/]    - Exit Hermes\n"
            ),
            title="Help",
            border_style="green",
        )
        self.console.print(help_text)
        self.console.print()
