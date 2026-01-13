"""User input widget for the dedicated input area."""

from textual.widgets import Input
from textual.message import Message


class UserInput(Input):
    """
    User input widget with command support.

    Handles user text input and distinguishes between regular messages
    and slash commands (e.g., /help, /status).
    """

    class Submitted(Message):
        """Message posted when user submits input."""

        def __init__(self, value: str, is_command: bool):
            """
            Initialize submitted message.

            Args:
                value: The submitted text
                is_command: Whether this is a slash command
            """
            self.value = value
            self.is_command = is_command
            super().__init__()

    def __init__(self, *args, **kwargs):
        """Initialize with placeholder text."""
        # Set default placeholder if not provided
        if "placeholder" not in kwargs:
            kwargs["placeholder"] = "Type your message... (/ for commands)"
        super().__init__(*args, **kwargs)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        value = event.value.strip()
        if value:
            is_command = value.startswith("/")
            self.post_message(self.Submitted(value, is_command))
            self.value = ""
