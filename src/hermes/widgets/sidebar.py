"""Sidebar widgets for status and emotional state display."""

from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Static


class EmotionalStateDisplay(Static):
    """Displays current emotional state with visual bars."""

    valence: reactive[float] = reactive(0.0)
    arousal: reactive[float] = reactive(0.0)
    quadrant: reactive[str] = reactive("neutral")

    def __init__(self, *args, **kwargs):
        """Initialize with border title."""
        super().__init__(*args, **kwargs)
        self.border_title = "Emotional State"

    def render(self) -> str:
        """Render emotional state with visual bars."""
        # Valence bar: [-1, 1] -> [0, 10]
        v_pos = int((self.valence + 1) * 5)
        v_bar = "=" * v_pos + "-" * (10 - v_pos)

        # Arousal bar: [0, 1] -> [0, 10]
        a_pos = int(max(0, min(1, self.arousal)) * 10)
        a_bar = "=" * a_pos + "-" * (10 - a_pos)

        # Color based on quadrant
        quadrant_colors = {
            "excited": "bright_green",
            "frustrated": "red",
            "calm": "blue",
            "depleted": "dim",
            "neutral": "white",
        }
        color = quadrant_colors.get(self.quadrant, "white")

        return (
            f"[{color}]{self.quadrant.capitalize()}[/]\n\n"
            f"Valence: [{v_bar}] {self.valence:+.2f}\n"
            f"Arousal: [{a_bar}] {self.arousal:.2f}"
        )


class StatusDisplay(Static):
    """Displays server status information."""

    state: reactive[str] = reactive("idle")
    message_count: reactive[int] = reactive(0)
    token_usage: reactive[str] = reactive("0/0")

    def __init__(self, *args, **kwargs):
        """Initialize with border title."""
        super().__init__(*args, **kwargs)
        self.border_title = "Status"

    def render(self) -> str:
        """Render status information."""
        state_colors = {
            "idle": "dim",
            "thinking": "yellow",
            "processing_tools": "cyan",
            "waiting_input": "green",
        }
        color = state_colors.get(self.state, "white")

        return (
            f"State: [{color}]{self.state}[/]\n"
            f"Messages: {self.message_count}\n"
            f"Tokens: {self.token_usage}"
        )


class Sidebar(Vertical):
    """Sidebar container with status widgets."""

    def compose(self):
        """Compose sidebar widgets."""
        yield EmotionalStateDisplay(id="emotion-display")
        yield StatusDisplay(id="status-display")
        # ToolActivity imported separately to avoid circular import
        from hermes.widgets.tool_activity import ToolActivity

        yield ToolActivity(id="tool-activity")
