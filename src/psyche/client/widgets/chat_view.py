"""Chat view widget for displaying conversation history with streaming support."""

from textual.widgets import RichLog
from textual.reactive import reactive


class ChatView(RichLog):
    """
    Chat history widget with streaming message support.

    Displays conversation between user and assistant, with support for
    streaming tokens in real-time as they are generated.
    """

    is_streaming: reactive[bool] = reactive(False)

    def __init__(self, *args, **kwargs):
        """Initialize chat view with markdown support."""
        super().__init__(*args, markup=True, highlight=True, wrap=True, **kwargs)
        self._stream_buffer: list[str] = []

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the chat.

        Args:
            content: The user's message text
        """
        self.write(f"[bold cyan]You:[/] {content}")

    def start_stream(self) -> None:
        """Start a new streaming assistant response."""
        self.is_streaming = True
        self._stream_buffer = []
        self.write("[bold green]Assistant:[/] ", end="")

    def append_token(self, token: str) -> None:
        """
        Append a token to the current streaming response.

        Args:
            token: The token to append
        """
        self._stream_buffer.append(token)
        self.write(token, end="")

    def end_stream(self) -> None:
        """Complete the current streaming response."""
        self.is_streaming = False
        self.write("")  # New line after stream

    def add_assistant_message(self, content: str) -> None:
        """
        Add a complete assistant message (non-streaming).

        Args:
            content: The assistant's message text
        """
        self.write(f"[bold green]Assistant:[/] {content}")

    def add_system_message(self, content: str) -> None:
        """
        Add a system message (errors, status updates).

        Args:
            content: The system message text
        """
        self.write(f"[dim yellow]{content}[/]")

    def watch_is_streaming(self, streaming: bool) -> None:
        """React to streaming state changes for visual feedback."""
        if streaming:
            self.add_class("streaming")
        else:
            self.remove_class("streaming")
