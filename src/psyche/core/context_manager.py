"""Context management for working memory with compaction support.

Extracts context-related functionality from MemoryServer into a dedicated module
for better separation of concerns.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from psyche.memory.compaction import (
    CompactionResult,
    ContextCompactor,
    Message,
    create_message,
)
from psyche.mcp.client import MnemosyneClient


@dataclass
class ContextConfig:
    """Configuration for context management."""

    # Token limits (sized for typical 32k context window)
    max_context_tokens: int = 24000
    reserve_tokens: int = 4000

    # Checkpoint settings
    enable_checkpoints: bool = True
    checkpoint_interval: int = 20  # Save checkpoint every N messages


class ContextManager:
    """
    Manages working memory context with compaction support.

    Wraps ContextCompactor and provides:
    - Message addition with automatic compaction triggering
    - Periodic checkpoint saving
    - Context clearing
    - Summary/stats generation
    """

    def __init__(
        self,
        config: Optional[ContextConfig] = None,
        mnemosyne_client: Optional[MnemosyneClient] = None,
        summarize_fn: Optional[Callable[[List[Message]], str]] = None,
    ):
        """
        Initialize the context manager.

        Args:
            config: Context configuration settings
            mnemosyne_client: Optional Mnemosyne client for checkpoint storage
            summarize_fn: Optional function to summarize messages during compaction
        """
        self.config = config or ContextConfig()
        self.mnemosyne_client = mnemosyne_client

        # Initialize the compactor
        self._compactor = ContextCompactor(
            max_tokens=self.config.max_context_tokens,
            reserve_tokens=self.config.reserve_tokens,
            summarize_fn=summarize_fn,
        )

        # Message counter for periodic checkpoints
        self._message_count: int = 0

        # System prompt tracking
        self._system_prompt: Optional[str] = None

    @property
    def messages(self) -> List[Message]:
        """Get current messages in context."""
        return self._compactor.messages

    @property
    def total_tokens(self) -> int:
        """Get total token count in current context."""
        return self._compactor.total_tokens

    @property
    def available_tokens(self) -> int:
        """Get available tokens for new content."""
        return self._compactor.available_tokens

    @property
    def message_count(self) -> int:
        """Get count of messages added since last reset."""
        return self._message_count

    def set_system_prompt(self, prompt: str) -> Optional[CompactionResult]:
        """
        Set the system prompt for the context.

        Args:
            prompt: The system prompt content

        Returns:
            CompactionResult if compaction was triggered, None otherwise
        """
        self._system_prompt = prompt
        return self._compactor.add_message(create_message("system", prompt))

    def update_system_prompt(self, prompt: str) -> None:
        """
        Update an existing system prompt without adding a new message.

        Finds the first system message and updates it in place.

        Args:
            prompt: The new system prompt content
        """
        self._system_prompt = prompt

        # Find and update the system prompt in the compactor
        for i, msg in enumerate(self._compactor._messages):
            if msg.role == "system":
                new_msg = create_message("system", prompt)
                new_msg.timestamp = msg.timestamp

                # Update token count difference
                token_diff = new_msg.token_count - msg.token_count
                self._compactor._total_tokens += token_diff
                self._compactor._messages[i] = new_msg
                return

        # No existing system prompt found, add one
        self._compactor.add_message(create_message("system", prompt))

    def add_message(self, role: str, content: str) -> Optional[CompactionResult]:
        """
        Add a message to the context.

        Args:
            role: Message role (user, assistant, system)
            content: Message content

        Returns:
            CompactionResult if compaction was triggered, None otherwise
        """
        message = create_message(role, content)
        result = self._compactor.add_message(message)
        self._message_count += 1
        return result

    def add_raw_message(self, message: Message) -> Optional[CompactionResult]:
        """
        Add a pre-created Message object to the context.

        Args:
            message: Message object to add

        Returns:
            CompactionResult if compaction was triggered, None otherwise
        """
        result = self._compactor.add_message(message)
        self._message_count += 1
        return result

    def get_api_messages(self) -> List[Dict[str, str]]:
        """
        Get messages formatted for API calls.

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        return self._compactor.get_api_messages()

    def should_checkpoint(self) -> bool:
        """
        Check if a checkpoint should be saved based on message count.

        Returns:
            True if checkpoint interval reached, False otherwise
        """
        if not self.config.enable_checkpoints:
            return False

        return self._message_count % self.config.checkpoint_interval == 0

    def get_checkpoint_messages(self) -> List[Message]:
        """
        Get messages suitable for checkpoint storage.

        Excludes system prompts as they don't need to be checkpointed.

        Returns:
            List of non-system messages
        """
        return [m for m in self._compactor.messages if m.role != "system"]

    def clear(self) -> None:
        """
        Clear all context including messages and counters.

        If a system prompt was set, it is re-added after clearing.
        """
        self._compactor.clear()
        self._message_count = 0

        # Re-add system prompt if one was set
        if self._system_prompt:
            self._compactor.add_message(create_message("system", self._system_prompt))

        logger.info("Context cleared")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context state.

        Returns:
            Dictionary with context statistics
        """
        return {
            "message_count": len(self._compactor.messages),
            "total_tokens": self._compactor.total_tokens,
            "available_tokens": self._compactor.available_tokens,
            "messages_added": self._message_count,
            "has_system_prompt": self._system_prompt is not None,
        }

    def mark_important(self, index: int) -> None:
        """
        Mark a message as important (won't be compacted in importance-based mode).

        Args:
            index: Index of the message to mark
        """
        self._compactor.mark_important(index)
