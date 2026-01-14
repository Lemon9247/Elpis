"""Context compaction for managing conversation history within token limits."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


@dataclass
class Message:
    """A message in the conversation history."""

    role: str
    content: str
    timestamp: float = 0.0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        """Convert to API format."""
        return {"role": self.role, "content": self.content}


@dataclass
class CompactionResult:
    """Result of a compaction operation."""

    messages: List[Message]
    summary: Optional[Message]
    messages_compacted: int
    tokens_saved: int
    dropped_messages: List[Message] = field(default_factory=list)


class ContextCompactor:
    """
    Manages conversation context by compacting old messages when approaching token limits.

    Strategies:
    1. Sliding window: Keep last N messages
    2. Summarization: Compress old messages into a summary
    3. Importance-based: Keep messages marked as important
    """

    def __init__(
        self,
        max_tokens: int = 6000,
        reserve_tokens: int = 2000,
        min_messages_to_keep: int = 4,
        summarize_fn: Optional[Callable[[List[Message]], str]] = None,
    ):
        """
        Initialize the compactor.

        Args:
            max_tokens: Maximum tokens allowed in context
            reserve_tokens: Tokens reserved for response generation
            min_messages_to_keep: Minimum recent messages to always keep
            summarize_fn: Async function to summarize messages (if None, uses sliding window)
        """
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.min_messages_to_keep = min_messages_to_keep
        self.summarize_fn = summarize_fn
        self.available_tokens = max_tokens - reserve_tokens

        self._messages: List[Message] = []
        self._summary: Optional[Message] = None
        self._total_tokens = 0

    @property
    def messages(self) -> List[Message]:
        """Get current messages including summary if present."""
        if self._summary:
            return [self._summary] + self._messages
        return self._messages.copy()

    @property
    def total_tokens(self) -> int:
        """Get total tokens in current context."""
        return self._total_tokens + (self._summary.token_count if self._summary else 0)

    def add_message(self, message: Message) -> Optional[CompactionResult]:
        """
        Add a message to the context.

        Args:
            message: Message to add

        Returns:
            CompactionResult if compaction was performed, None otherwise
        """
        self._messages.append(message)
        self._total_tokens += message.token_count

        # Check if compaction is needed
        if self.total_tokens > self.available_tokens:
            return self._compact()

        return None

    def add_messages(self, messages: List[Message]) -> Optional[CompactionResult]:
        """Add multiple messages at once."""
        for msg in messages:
            self._messages.append(msg)
            self._total_tokens += msg.token_count

        if self.total_tokens > self.available_tokens:
            return self._compact()

        return None

    def _compact(self) -> CompactionResult:
        """
        Perform context compaction.

        Uses sliding window strategy if no summarize_fn provided,
        otherwise uses summarization.
        """
        if self.summarize_fn:
            return self._compact_with_summary()
        return self._compact_sliding_window()

    def _compact_sliding_window(self) -> CompactionResult:
        """Compact using sliding window - drop oldest messages."""
        messages_removed = []
        tokens_removed = 0

        while (
            self.total_tokens > self.available_tokens
            and len(self._messages) > self.min_messages_to_keep
        ):
            removed = self._messages.pop(0)
            messages_removed.append(removed)
            tokens_removed += removed.token_count
            self._total_tokens -= removed.token_count

        logger.debug(
            f"Compacted {len(messages_removed)} messages, "
            f"freed {tokens_removed} tokens, "
            f"remaining: {self.total_tokens}/{self.available_tokens}"
        )

        return CompactionResult(
            messages=self._messages.copy(),
            summary=self._summary,
            messages_compacted=len(messages_removed),
            tokens_saved=tokens_removed,
            dropped_messages=messages_removed,
        )

    def _compact_with_summary(self) -> CompactionResult:
        """Compact by summarizing old messages."""
        # Find messages to summarize (all except min_messages_to_keep)
        to_summarize = self._messages[: -self.min_messages_to_keep]
        to_keep = self._messages[-self.min_messages_to_keep :]

        if not to_summarize:
            # Can't summarize - fall back to sliding window
            return self._compact_sliding_window()

        # Keep track of actual messages being dropped (not previous summary)
        dropped = to_summarize.copy()

        # Include existing summary in what gets re-summarized
        if self._summary:
            to_summarize.insert(0, self._summary)

        # Generate new summary
        summary_text = self.summarize_fn(to_summarize)
        tokens_removed = sum(m.token_count for m in to_summarize)

        # Estimate summary tokens (rough heuristic)
        summary_tokens = len(summary_text.split()) * 1.3

        self._summary = Message(
            role="system",
            content=f"[Previous conversation summary]\n{summary_text}",
            token_count=int(summary_tokens),
        )

        self._messages = to_keep
        self._total_tokens = sum(m.token_count for m in to_keep)

        logger.debug(
            f"Summarized {len(to_summarize)} messages, "
            f"saved {tokens_removed - summary_tokens:.0f} tokens"
        )

        return CompactionResult(
            messages=self.messages,
            summary=self._summary,
            messages_compacted=len(to_summarize),
            tokens_saved=int(tokens_removed - summary_tokens),
            dropped_messages=dropped,
        )

    def clear(self) -> None:
        """Clear all messages and summary."""
        self._messages.clear()
        self._summary = None
        self._total_tokens = 0

    def get_api_messages(self) -> List[Dict[str, str]]:
        """Get messages in API format."""
        return [m.to_dict() for m in self.messages]

    def mark_important(self, index: int) -> None:
        """Mark a message as important (won't be compacted in importance-based mode)."""
        if 0 <= index < len(self._messages):
            self._messages[index].metadata["important"] = True


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    This is a rough heuristic - for accurate counts, use a tokenizer.
    """
    # Rough estimate: ~1.3 tokens per word for English
    words = len(text.split())
    return int(words * 1.3)


def create_message(role: str, content: str) -> Message:
    """Create a message with estimated token count."""
    import time

    return Message(
        role=role,
        content=content,
        timestamp=time.time(),
        token_count=estimate_tokens(content),
    )
