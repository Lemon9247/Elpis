"""Thought panel widget for displaying internal reflections."""

from textual.widgets import Static, RichLog
from textual.reactive import reactive


class ThoughtPanel(RichLog):
    """
    Collapsible panel for displaying internal thoughts/reflections.

    Shows the AI's internal reasoning and idle thoughts when enabled.
    Supports streaming indicator during idle thought generation.
    """

    visible: reactive[bool] = reactive(True)
    is_thinking: reactive[bool] = reactive(False)

    def __init__(self, *args, **kwargs):
        """Initialize thought panel."""
        super().__init__(*args, markup=True, highlight=True, wrap=True, max_lines=50, **kwargs)
        self.border_title = "Internal Thoughts"
        self.display = True  # Start visible
        self._thinking_token_count = 0  # Track tokens received during thinking

    def add_thought(self, content: str, thought_type: str = "reflection") -> None:
        """
        Add a thought to the panel.

        Args:
            content: The thought content
            thought_type: Type of thought (reflection, planning, idle, etc.)
        """
        # If we were thinking, stop the indicator
        if self.is_thinking:
            self.stop_thinking()

        type_colors = {
            "reflection": "cyan",
            "planning": "yellow",
            "idle": "dim",
            "memory": "magenta",
        }
        color = type_colors.get(thought_type, "white")

        self.write(f"[{color}][{thought_type}][/] {content}")

    def start_thinking(self) -> None:
        """
        Show thinking indicator when idle thought generation starts.

        Due to RichLog limitations, we show a simple indicator that
        will be replaced when the final thought is displayed.
        """
        if not self.is_thinking:
            self.is_thinking = True
            self._thinking_token_count = 0
            self.write("[dim italic][thinking...][/]")

    def on_thinking_token(self, token: str) -> None:
        """
        Handle a streaming token during thinking.

        Updates the thinking indicator to show activity.

        Args:
            token: The token received during streaming
        """
        if not self.is_thinking:
            self.start_thinking()

        self._thinking_token_count += 1

        # Update border title to show activity (token count)
        # This provides visual feedback without modifying RichLog content
        self.border_title = f"Internal Thoughts [thinking... {self._thinking_token_count} tokens]"

    def stop_thinking(self) -> None:
        """Stop the thinking indicator."""
        if self.is_thinking:
            self.is_thinking = False
            self._thinking_token_count = 0
            self.border_title = "Internal Thoughts"

    def watch_is_thinking(self, is_thinking: bool) -> None:
        """React to thinking state changes."""
        if is_thinking:
            self.add_class("thinking")
        else:
            self.remove_class("thinking")

    def toggle(self) -> None:
        """Toggle visibility of the thought panel."""
        self.visible = not self.visible

    def watch_visible(self, visible: bool) -> None:
        """React to visibility changes."""
        self.display = visible
        if visible:
            self.remove_class("hidden")
        else:
            self.add_class("hidden")

    def show(self) -> None:
        """Show the thought panel."""
        self.visible = True

    def hide(self) -> None:
        """Hide the thought panel."""
        self.visible = False
