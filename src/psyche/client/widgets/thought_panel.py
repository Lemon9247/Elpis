"""Thought panel widget for displaying internal reflections."""

from textual.widgets import Static, RichLog
from textual.reactive import reactive


class ThoughtPanel(RichLog):
    """
    Collapsible panel for displaying internal thoughts/reflections.

    Shows the AI's internal reasoning and idle thoughts when enabled.
    """

    visible: reactive[bool] = reactive(False)

    def __init__(self, *args, **kwargs):
        """Initialize thought panel."""
        super().__init__(*args, markup=True, highlight=True, wrap=True, max_lines=50, **kwargs)
        self.border_title = "Internal Thoughts"
        self.display = False  # Start hidden

    def add_thought(self, content: str, thought_type: str = "reflection") -> None:
        """
        Add a thought to the panel.

        Args:
            content: The thought content
            thought_type: Type of thought (reflection, planning, idle, etc.)
        """
        type_colors = {
            "reflection": "cyan",
            "planning": "yellow",
            "idle": "dim",
            "memory": "magenta",
        }
        color = type_colors.get(thought_type, "white")

        self.write(f"[{color}][{thought_type}][/] {content}")

    def toggle(self) -> None:
        """Toggle visibility of the thought panel."""
        self.visible = not self.visible

    def watch_visible(self, visible: bool) -> None:
        """React to visibility changes."""
        self.display = visible
        if visible:
            self.add_class("visible")
        else:
            self.remove_class("visible")

    def show(self) -> None:
        """Show the thought panel."""
        self.visible = True

    def hide(self) -> None:
        """Hide the thought panel."""
        self.visible = False
