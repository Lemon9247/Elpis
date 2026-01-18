"""Memory handler for long-term storage via Mnemosyne with fallback."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from psyche.mcp.client import ElpisClient, MnemosyneClient
from psyche.memory.compaction import CompactionResult, Message


# Default path for local fallback storage
DEFAULT_FALLBACK_DIR = Path.home() / ".psyche" / "fallback_memories"


@dataclass
class MemoryHandlerConfig:
    """Configuration for memory handler."""

    # Auto-retrieval settings
    enable_auto_retrieval: bool = True
    auto_retrieval_count: int = 3

    # Auto-storage settings
    auto_storage: bool = True
    auto_storage_threshold: float = 0.6


class MemoryHandler:
    """
    Handles long-term memory storage via Mnemosyne with fallback.

    Provides methods for:
    - Retrieving relevant memories for context injection
    - Storing messages to Mnemosyne short-term memory
    - Generating and storing conversation summaries
    - Local fallback storage when Mnemosyne is unavailable
    - Handling compaction results with staged message storage
    """

    def __init__(
        self,
        mnemosyne_client: Optional[MnemosyneClient],
        elpis_client: ElpisClient,
        fallback_dir: Optional[Path] = None,
        config: Optional[MemoryHandlerConfig] = None,
    ):
        """
        Initialize the memory handler.

        Args:
            mnemosyne_client: Client for Mnemosyne memory server (can be None)
            elpis_client: Client for Elpis inference server (for summarization)
            fallback_dir: Directory for fallback storage (defaults to ~/.psyche/fallback_memories)
            config: Configuration options
        """
        self.mnemosyne_client = mnemosyne_client
        self.elpis_client = elpis_client
        self.fallback_dir = fallback_dir or DEFAULT_FALLBACK_DIR
        self.config = config or MemoryHandlerConfig()

        # Staged messages buffer for delayed Mnemosyne storage
        # Messages are staged for one compaction cycle before being stored
        self._staged_messages: List[Message] = []

    @property
    def is_mnemosyne_available(self) -> bool:
        """Check if Mnemosyne client is connected and available."""
        return (
            self.mnemosyne_client is not None
            and self.mnemosyne_client.is_connected
        )

    async def retrieve_relevant(
        self,
        query: str,
        n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to query from Mnemosyne.

        Args:
            query: The query text (typically user input)
            n: Number of memories to retrieve (uses config default if None)

        Returns:
            List of memory dictionaries, empty if no memories found or unavailable
        """
        if not self.config.enable_auto_retrieval:
            return []

        if n is None:
            n = self.config.auto_retrieval_count

        if not self.is_mnemosyne_available:
            return []

        try:
            memories = await self.mnemosyne_client.search_memories(query, n_results=n)

            if not memories:
                logger.debug("No relevant memories found")
                return []

            logger.debug(f"Retrieved {len(memories)} relevant memories")
            return memories

        except Exception as e:
            logger.warning(f"Failed to retrieve memories: {e}")
            return []

    def format_memories_for_context(self, memories: List[Dict[str, Any]]) -> Optional[str]:
        """
        Format retrieved memories for injection into conversation context.

        Args:
            memories: List of memory dictionaries from retrieve_relevant

        Returns:
            Formatted memory context string, or None if no memories
        """
        if not memories:
            return None

        formatted = []
        for i, memory in enumerate(memories, 1):
            # Extract content and summary
            content = memory.get("content", "")
            summary = memory.get("summary", "")
            memory_type = memory.get("memory_type", "unknown")

            # Use summary if available and content is long
            display_content = summary if summary and len(content) > 200 else content

            # Truncate very long content
            if len(display_content) > 300:
                display_content = display_content[:300] + "..."

            formatted.append(f"{i}. [{memory_type}] {display_content}")

        return "\n".join(formatted)

    async def store_messages(
        self,
        messages: List[Message],
        emotional_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store messages to Mnemosyne short-term memory.

        Args:
            messages: List of messages to store as episodic memories
            emotional_context: Optional emotional state {valence, arousal, quadrant}

        Returns:
            True if all messages stored successfully, False otherwise
        """
        if not self.mnemosyne_client:
            logger.warning("No Mnemosyne client, cannot store messages")
            return False

        if not self.mnemosyne_client.is_connected:
            logger.warning("Mnemosyne not connected, cannot store messages")
            return False

        success_count = 0
        total_count = 0

        for msg in messages:
            if msg.role == "system":
                continue  # Skip system prompts

            total_count += 1
            try:
                await self.mnemosyne_client.store_memory(
                    content=msg.content,
                    summary=msg.content[:500],
                    memory_type="episodic",
                    tags=["compacted", msg.role],
                    emotional_context=emotional_context,
                )
                success_count += 1
                logger.debug(f"Stored {msg.role} message to Mnemosyne")
            except Exception as e:
                logger.error(f"Failed to store message to Mnemosyne: {type(e).__name__}: {e}")

        if total_count == 0:
            return True  # No messages to store is a success

        return success_count == total_count

    async def store_summary(
        self,
        summary: str,
        emotional_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store a conversation summary as semantic memory.

        Args:
            summary: The summary text to store
            emotional_context: Optional emotional state {valence, arousal, quadrant}

        Returns:
            True if stored successfully, False otherwise
        """
        if not self.is_mnemosyne_available:
            return False

        if not summary:
            logger.debug("No summary to store")
            return False

        try:
            await self.mnemosyne_client.store_memory(
                content=summary,
                summary=summary[:500] + "..." if len(summary) > 500 else summary,
                memory_type="semantic",  # Semantic memory for distilled knowledge
                tags=["conversation_summary", "shutdown"],
                emotional_context=emotional_context,
            )
            logger.info("Stored conversation summary as semantic memory")
            return True

        except Exception as e:
            logger.error(f"Failed to store conversation summary: {e}")
            return False

    async def summarize_conversation(self, messages: List[Message]) -> str:
        """
        Use Elpis to generate a conversation summary.

        Args:
            messages: List of messages to summarize

        Returns:
            Summary string, or empty string on failure
        """
        if not messages:
            return ""

        try:
            # Build conversation text for summarization
            conversation_parts = []
            for msg in messages:
                if msg.role == "system":
                    continue
                # Truncate very long messages for summarization
                content = msg.content[:500] if len(msg.content) > 500 else msg.content
                conversation_parts.append(f"{msg.role}: {content}")

            if not conversation_parts:
                return ""

            conversation_text = "\n".join(conversation_parts)

            # Use Elpis to generate summary
            summary_prompt = [
                {
                    "role": "system",
                    "content": (
                        "Summarize this conversation concisely. Extract key facts, "
                        "decisions, topics discussed, and important details. "
                        "Focus on information worth remembering long-term."
                    ),
                },
                {"role": "user", "content": conversation_text},
            ]

            result = await self.elpis_client.generate(
                messages=summary_prompt,
                max_tokens=500,
                temperature=0.3,
            )

            summary = result.content.strip()
            logger.debug(f"Generated conversation summary: {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            return ""

    async def store_conversation_summary(
        self,
        messages: List[Message],
        emotional_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Generate and store a conversation summary as semantic memory.

        Args:
            messages: List of messages to summarize
            emotional_context: Optional emotional state

        Returns:
            True if summary stored successfully, False otherwise
        """
        summary = await self.summarize_conversation(messages)
        if not summary:
            logger.debug("No summary generated, skipping storage")
            return False

        return await self.store_summary(summary, emotional_context)

    async def handle_compaction(
        self,
        result: CompactionResult,
        emotional_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Handle compaction result by storing staged messages and staging new ones.

        Implements delayed storage: messages are staged for one compaction cycle
        before being stored to Mnemosyne short-term memory. Falls back to local
        storage if Mnemosyne is unavailable.

        Args:
            result: CompactionResult from context compaction
            emotional_context: Optional emotional state for stored memories
        """
        # Store previously staged messages to Mnemosyne (or local fallback)
        if self._staged_messages:
            staged_count = len(self._staged_messages)
            stored = False

            if self.is_mnemosyne_available:
                logger.debug(f"Storing {staged_count} staged messages to Mnemosyne")
                stored = await self.store_messages(self._staged_messages, emotional_context)

            if not stored:
                # Mnemosyne unavailable or failed - use local fallback
                logger.warning(
                    f"Mnemosyne unavailable, saving {staged_count} messages to local fallback"
                )
                if self.save_to_fallback(self._staged_messages, reason="compaction"):
                    stored = True

            if stored:
                self._staged_messages = []
            else:
                logger.error(
                    f"Failed to store {staged_count} staged messages to both "
                    "Mnemosyne and local fallback"
                )

        # Stage newly dropped messages (append to any failed messages)
        if result.dropped_messages:
            self._staged_messages.extend(result.dropped_messages)
            logger.debug(
                f"Staged {len(result.dropped_messages)} new messages, "
                f"total staged: {len(self._staged_messages)}"
            )

    def save_to_fallback(
        self,
        messages: List[Message],
        reason: str = "fallback",
    ) -> Optional[Path]:
        """
        Save messages to local JSON file as fallback when Mnemosyne is unavailable.

        Creates timestamped JSON files in the fallback directory.

        Args:
            messages: List of messages to save
            reason: Reason for fallback (for logging and filename)

        Returns:
            Path to saved file if successful, None otherwise
        """
        if not messages:
            return None

        try:
            # Ensure fallback directory exists
            self.fallback_dir.mkdir(parents=True, exist_ok=True)

            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fallback_{timestamp}_{reason}.json"
            filepath = self.fallback_dir / filename

            # Convert messages to serializable format
            data = {
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "message_count": len(messages),
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "token_count": msg.token_count,
                        "metadata": msg.metadata,
                    }
                    for msg in messages
                    if msg.role != "system"  # Skip system prompts
                ],
            }

            # Write to file
            filepath.write_text(json.dumps(data, indent=2))
            logger.info(f"Saved {len(messages)} messages to local fallback: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save to local fallback: {e}")
            return None

    def get_pending_fallbacks(self) -> List[Path]:
        """
        Get list of pending fallback files that haven't been restored.

        Returns:
            List of Path objects for pending fallback files
        """
        if not self.fallback_dir.exists():
            return []

        return sorted(self.fallback_dir.glob("fallback_*.json"))

    @property
    def staged_message_count(self) -> int:
        """Get the number of currently staged messages."""
        return len(self._staged_messages)

    def clear_staged_messages(self) -> None:
        """Clear all staged messages without storing them."""
        self._staged_messages = []

    async def flush_staged_messages(
        self,
        emotional_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Immediately flush all staged messages to storage.

        Useful for graceful shutdown scenarios.

        Args:
            emotional_context: Optional emotional state for stored memories

        Returns:
            True if all staged messages were stored successfully
        """
        if not self._staged_messages:
            return True

        staged_count = len(self._staged_messages)
        stored = False

        if self.is_mnemosyne_available:
            logger.debug(f"Flushing {staged_count} staged messages to Mnemosyne")
            stored = await self.store_messages(self._staged_messages, emotional_context)

        if not stored:
            logger.warning(
                f"Mnemosyne unavailable, saving {staged_count} messages to local fallback"
            )
            if self.save_to_fallback(self._staged_messages, reason="flush"):
                stored = True

        if stored:
            self._staged_messages = []
            return True

        return False
