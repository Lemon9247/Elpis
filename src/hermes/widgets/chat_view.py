"""Chat view widget for displaying conversation history with streaming support."""

from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import RichLog, Static


class StreamingMessage(Static):
    """Widget for displaying a message that's currently being streamed."""

    DEFAULT_CSS = """
    StreamingMessage {
        height: auto;
        padding: 0;
        margin: 0;
    }
    """

    def __init__(self, *args, **kwargs):
        """Initialize streaming message."""
        super().__init__(*args, **kwargs)
        self._buffer: list[str] = []

    def start(self) -> None:
        """Start a new streaming message."""
        self._buffer = []
        self.update("[bold green]Assistant:[/] ▌")

    def append_token(self, token: str) -> None:
        """Append a token to the streaming message."""
        self._buffer.append(token)
        content = "".join(self._buffer)
        self.update(f"[bold green]Assistant:[/] {content}▌")

    def get_content(self) -> str:
        """Get the accumulated content."""
        return "".join(self._buffer)

    def finish(self) -> None:
        """Finish streaming and remove cursor."""
        content = "".join(self._buffer)
        self.update(f"[bold green]Assistant:[/] {content}")

    def clear_stream(self) -> None:
        """Clear the streaming message."""
        self._buffer = []
        self.update("")


class ChatView(VerticalScroll):
    """
    Chat history widget with streaming message support.

    Displays conversation between user and assistant, with support for
    streaming tokens in real-time as they are generated.
    """

    is_streaming: reactive[bool] = reactive(False)

    DEFAULT_CSS = """
    ChatView {
        height: 1fr;
        scrollbar-color: $primary;
    }
    """

    def __init__(self, *args, **kwargs):
        """Initialize chat view."""
        super().__init__(*args, **kwargs)
        self._history: RichLog | None = None
        self._streaming: StreamingMessage | None = None

    def compose(self):
        """Compose the chat view with history and streaming widgets."""
        self._history = RichLog(markup=True, highlight=True, wrap=True, id="chat-history")
        self._streaming = StreamingMessage(id="streaming-message")
        yield self._history
        yield self._streaming

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the chat.

        Args:
            content: The user's message text
        """
        self._history.write(f"[bold cyan]You:[/] {content}")
        self.scroll_end(animate=False)

    def start_stream(self) -> None:
        """Start a new streaming assistant response."""
        self.is_streaming = True
        self._streaming.start()
        self.scroll_end(animate=False)

    def append_token(self, token: str) -> None:
        """
        Append a token to the current streaming response.

        Args:
            token: The token to append
        """
        self._streaming.append_token(token)
        self.scroll_end(animate=False)

    def end_stream(self) -> None:
        """Complete the current streaming response."""
        self.is_streaming = False
        # Move completed message to history
        content = self._streaming.get_content()
        if content:
            self._history.write(f"[bold green]Assistant:[/] {content}")
        self._streaming.clear_stream()
        self.scroll_end(animate=False)

    def add_assistant_message(self, content: str) -> None:
        """
        Add a complete assistant message (non-streaming).

        Args:
            content: The assistant's message text
        """
        self._history.write(f"[bold green]Assistant:[/] {content}")
        self.scroll_end(animate=False)

    def add_system_message(self, content: str) -> None:
        """
        Add a system message (errors, status updates).

        Args:
            content: The system message text
        """
        self._history.write(f"[dim yellow]{content}[/]")
        self.scroll_end(animate=False)

    def clear(self) -> None:
        """Clear all chat history."""
        self._history.clear()
        self._streaming.clear_stream()

    def watch_is_streaming(self, streaming: bool) -> None:
        """React to streaming state changes for visual feedback."""
        if streaming:
            self.add_class("streaming")
        else:
            self.remove_class("streaming")
