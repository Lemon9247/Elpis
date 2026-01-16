"""Unit tests for MemoryHandler."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from psyche.core.memory_handler import (
    MemoryHandler,
    MemoryHandlerConfig,
    DEFAULT_FALLBACK_DIR,
)
from psyche.memory.compaction import CompactionResult, Message


class TestMemoryHandlerInit:
    """Tests for MemoryHandler initialization."""

    def test_init_with_defaults(self):
        """MemoryHandler should initialize with default config."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        assert handler.mnemosyne_client is None
        assert handler.elpis_client == mock_elpis
        assert handler.fallback_dir == DEFAULT_FALLBACK_DIR
        assert handler.config.enable_auto_retrieval is True
        assert handler.config.auto_retrieval_count == 3
        assert handler._staged_messages == []

    def test_init_with_custom_config(self):
        """MemoryHandler should accept custom configuration."""
        mock_elpis = MagicMock()
        custom_config = MemoryHandlerConfig(
            enable_auto_retrieval=False,
            auto_retrieval_count=5,
            auto_storage=False,
            auto_storage_threshold=0.8,
        )

        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
            config=custom_config,
        )

        assert handler.config.enable_auto_retrieval is False
        assert handler.config.auto_retrieval_count == 5
        assert handler.config.auto_storage is False
        assert handler.config.auto_storage_threshold == 0.8

    def test_init_with_custom_fallback_dir(self, tmp_path):
        """MemoryHandler should accept custom fallback directory."""
        mock_elpis = MagicMock()
        custom_dir = tmp_path / "custom_fallback"

        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
            fallback_dir=custom_dir,
        )

        assert handler.fallback_dir == custom_dir

    def test_init_with_mnemosyne_client(self):
        """MemoryHandler should accept Mnemosyne client."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        assert handler.mnemosyne_client == mock_mnemosyne
        assert handler.is_mnemosyne_available is True


class TestMnemosyneAvailability:
    """Tests for Mnemosyne availability checking."""

    def test_is_mnemosyne_available_when_none(self):
        """Should return False when no Mnemosyne client."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        assert handler.is_mnemosyne_available is False

    def test_is_mnemosyne_available_when_disconnected(self):
        """Should return False when Mnemosyne is disconnected."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = False

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        assert handler.is_mnemosyne_available is False

    def test_is_mnemosyne_available_when_connected(self):
        """Should return True when Mnemosyne is connected."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        assert handler.is_mnemosyne_available is True


class TestRetrieveRelevant:
    """Tests for retrieve_relevant method."""

    @pytest.mark.asyncio
    async def test_retrieve_relevant_disabled(self):
        """Should return empty list when auto-retrieval is disabled."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True

        config = MemoryHandlerConfig(enable_auto_retrieval=False)
        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
            config=config,
        )

        result = await handler.retrieve_relevant("test query")

        assert result == []
        mock_mnemosyne.search_memories.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_relevant_no_mnemosyne(self):
        """Should return empty list when Mnemosyne unavailable."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        result = await handler.retrieve_relevant("test query")

        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_relevant_success(self):
        """Should return memories from Mnemosyne."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True

        expected_memories = [
            {"content": "Memory 1", "memory_type": "episodic"},
            {"content": "Memory 2", "memory_type": "semantic"},
        ]
        mock_mnemosyne.search_memories = AsyncMock(return_value=expected_memories)

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        result = await handler.retrieve_relevant("test query")

        assert result == expected_memories
        mock_mnemosyne.search_memories.assert_called_once_with("test query", n_results=3)

    @pytest.mark.asyncio
    async def test_retrieve_relevant_custom_count(self):
        """Should use custom n parameter."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True
        mock_mnemosyne.search_memories = AsyncMock(return_value=[])

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        await handler.retrieve_relevant("test query", n=10)

        mock_mnemosyne.search_memories.assert_called_once_with("test query", n_results=10)

    @pytest.mark.asyncio
    async def test_retrieve_relevant_handles_exception(self):
        """Should return empty list on exception."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True
        mock_mnemosyne.search_memories = AsyncMock(side_effect=Exception("Connection error"))

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        result = await handler.retrieve_relevant("test query")

        assert result == []


class TestFormatMemoriesForContext:
    """Tests for format_memories_for_context method."""

    def test_format_empty_list(self):
        """Should return None for empty list."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        result = handler.format_memories_for_context([])

        assert result is None

    def test_format_single_memory(self):
        """Should format single memory correctly."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        memories = [{"content": "Test content", "memory_type": "episodic"}]
        result = handler.format_memories_for_context(memories)

        assert result == "1. [episodic] Test content"

    def test_format_uses_summary_for_long_content(self):
        """Should use summary when content is long."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        long_content = "x" * 300
        memories = [
            {
                "content": long_content,
                "summary": "Short summary",
                "memory_type": "semantic",
            }
        ]
        result = handler.format_memories_for_context(memories)

        assert "Short summary" in result
        assert long_content not in result

    def test_format_truncates_very_long_content(self):
        """Should truncate content over 300 chars."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        long_content = "x" * 500
        memories = [{"content": long_content, "memory_type": "episodic"}]
        result = handler.format_memories_for_context(memories)

        assert "..." in result
        assert len(result) < 500


class TestStoreMessages:
    """Tests for store_messages method."""

    @pytest.mark.asyncio
    async def test_store_messages_no_mnemosyne(self):
        """Should return False when no Mnemosyne client."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        messages = [Message(role="user", content="Hello")]
        result = await handler.store_messages(messages)

        assert result is False

    @pytest.mark.asyncio
    async def test_store_messages_disconnected(self):
        """Should return False when Mnemosyne disconnected."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = False

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        messages = [Message(role="user", content="Hello")]
        result = await handler.store_messages(messages)

        assert result is False

    @pytest.mark.asyncio
    async def test_store_messages_skips_system(self):
        """Should skip system messages."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True
        mock_mnemosyne.store_memory = AsyncMock()

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="Hello"),
        ]
        result = await handler.store_messages(messages)

        assert result is True
        assert mock_mnemosyne.store_memory.call_count == 1

    @pytest.mark.asyncio
    async def test_store_messages_success(self):
        """Should store all non-system messages."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True
        mock_mnemosyne.store_memory = AsyncMock()

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        result = await handler.store_messages(messages)

        assert result is True
        assert mock_mnemosyne.store_memory.call_count == 2

    @pytest.mark.asyncio
    async def test_store_messages_with_emotional_context(self):
        """Should pass emotional context to store_memory."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True
        mock_mnemosyne.store_memory = AsyncMock()

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        messages = [Message(role="user", content="Hello")]
        emotional_context = {"valence": 0.5, "arousal": 0.3}
        await handler.store_messages(messages, emotional_context)

        mock_mnemosyne.store_memory.assert_called_once()
        call_kwargs = mock_mnemosyne.store_memory.call_args.kwargs
        assert call_kwargs["emotional_context"] == emotional_context

    @pytest.mark.asyncio
    async def test_store_messages_empty_list(self):
        """Should return True for empty message list."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        result = await handler.store_messages([])

        assert result is True

    @pytest.mark.asyncio
    async def test_store_messages_partial_failure(self):
        """Should return False if any message fails to store."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True
        mock_mnemosyne.store_memory = AsyncMock(side_effect=[None, Exception("Error")])

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        result = await handler.store_messages(messages)

        assert result is False


class TestStoreSummary:
    """Tests for store_summary method."""

    @pytest.mark.asyncio
    async def test_store_summary_no_mnemosyne(self):
        """Should return False when Mnemosyne unavailable."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        result = await handler.store_summary("Test summary")

        assert result is False

    @pytest.mark.asyncio
    async def test_store_summary_empty(self):
        """Should return False for empty summary."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        result = await handler.store_summary("")

        assert result is False

    @pytest.mark.asyncio
    async def test_store_summary_success(self):
        """Should store summary as semantic memory."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True
        mock_mnemosyne.store_memory = AsyncMock()

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        result = await handler.store_summary("Test summary")

        assert result is True
        mock_mnemosyne.store_memory.assert_called_once()
        call_kwargs = mock_mnemosyne.store_memory.call_args.kwargs
        assert call_kwargs["memory_type"] == "semantic"
        assert "conversation_summary" in call_kwargs["tags"]


class TestSummarizeConversation:
    """Tests for summarize_conversation method."""

    @pytest.mark.asyncio
    async def test_summarize_empty_messages(self):
        """Should return empty string for empty messages."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        result = await handler.summarize_conversation([])

        assert result == ""

    @pytest.mark.asyncio
    async def test_summarize_only_system_messages(self):
        """Should return empty string if only system messages."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        messages = [Message(role="system", content="System prompt")]
        result = await handler.summarize_conversation(messages)

        assert result == ""

    @pytest.mark.asyncio
    async def test_summarize_success(self):
        """Should generate summary via Elpis."""
        mock_elpis = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "This is a summary of the conversation."
        mock_elpis.generate = AsyncMock(return_value=mock_result)

        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        result = await handler.summarize_conversation(messages)

        assert result == "This is a summary of the conversation."
        mock_elpis.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_handles_exception(self):
        """Should return empty string on exception."""
        mock_elpis = MagicMock()
        mock_elpis.generate = AsyncMock(side_effect=Exception("Generation failed"))

        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        messages = [Message(role="user", content="Hello")]
        result = await handler.summarize_conversation(messages)

        assert result == ""


class TestHandleCompaction:
    """Tests for handle_compaction method."""

    @pytest.mark.asyncio
    async def test_handle_compaction_no_staged_messages(self):
        """Should stage dropped messages when no prior staged messages."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        dropped = [Message(role="user", content="Old message")]
        result = CompactionResult(
            messages=[],
            summary=None,
            messages_compacted=1,
            tokens_saved=10,
            dropped_messages=dropped,
        )

        await handler.handle_compaction(result)

        assert handler.staged_message_count == 1
        assert handler._staged_messages[0].content == "Old message"

    @pytest.mark.asyncio
    async def test_handle_compaction_stores_staged_to_mnemosyne(self):
        """Should store previously staged messages to Mnemosyne."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True
        mock_mnemosyne.store_memory = AsyncMock()

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        # Pre-stage some messages
        handler._staged_messages = [Message(role="user", content="Staged message")]

        result = CompactionResult(
            messages=[],
            summary=None,
            messages_compacted=0,
            tokens_saved=0,
            dropped_messages=[],
        )

        await handler.handle_compaction(result)

        # Staged messages should be stored and cleared
        assert handler.staged_message_count == 0
        mock_mnemosyne.store_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_compaction_fallback_when_mnemosyne_fails(self, tmp_path):
        """Should use fallback when Mnemosyne unavailable."""
        mock_elpis = MagicMock()

        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
            fallback_dir=tmp_path,
        )

        # Pre-stage some messages
        handler._staged_messages = [Message(role="user", content="Staged message")]

        result = CompactionResult(
            messages=[],
            summary=None,
            messages_compacted=0,
            tokens_saved=0,
            dropped_messages=[],
        )

        await handler.handle_compaction(result)

        # Should have created a fallback file
        fallback_files = list(tmp_path.glob("fallback_*.json"))
        assert len(fallback_files) == 1
        assert handler.staged_message_count == 0


class TestSaveToFallback:
    """Tests for save_to_fallback method."""

    def test_save_to_fallback_empty_messages(self, tmp_path):
        """Should return None for empty messages."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
            fallback_dir=tmp_path,
        )

        result = handler.save_to_fallback([])

        assert result is None

    def test_save_to_fallback_creates_file(self, tmp_path):
        """Should create JSON file with messages."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
            fallback_dir=tmp_path,
        )

        messages = [
            Message(role="user", content="Hello", timestamp=12345.0, token_count=5),
            Message(role="assistant", content="Hi", timestamp=12346.0, token_count=3),
        ]
        result = handler.save_to_fallback(messages, reason="test")

        assert result is not None
        assert result.exists()
        assert "test" in result.name

        # Verify content
        data = json.loads(result.read_text())
        assert data["reason"] == "test"
        assert data["message_count"] == 2
        assert len(data["messages"]) == 2

    def test_save_to_fallback_skips_system_messages(self, tmp_path):
        """Should skip system messages in saved data."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
            fallback_dir=tmp_path,
        )

        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="Hello"),
        ]
        result = handler.save_to_fallback(messages)

        data = json.loads(result.read_text())
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"

    def test_save_to_fallback_creates_directory(self, tmp_path):
        """Should create fallback directory if it doesn't exist."""
        mock_elpis = MagicMock()
        fallback_dir = tmp_path / "nested" / "fallback"

        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
            fallback_dir=fallback_dir,
        )

        messages = [Message(role="user", content="Hello")]
        result = handler.save_to_fallback(messages)

        assert result is not None
        assert fallback_dir.exists()


class TestGetPendingFallbacks:
    """Tests for get_pending_fallbacks method."""

    def test_get_pending_fallbacks_empty_dir(self, tmp_path):
        """Should return empty list for empty directory."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
            fallback_dir=tmp_path,
        )

        result = handler.get_pending_fallbacks()

        assert result == []

    def test_get_pending_fallbacks_no_dir(self, tmp_path):
        """Should return empty list if directory doesn't exist."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
            fallback_dir=tmp_path / "nonexistent",
        )

        result = handler.get_pending_fallbacks()

        assert result == []

    def test_get_pending_fallbacks_returns_sorted(self, tmp_path):
        """Should return sorted list of fallback files."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
            fallback_dir=tmp_path,
        )

        # Create some fallback files
        (tmp_path / "fallback_20240101_120000_test.json").write_text("{}")
        (tmp_path / "fallback_20240101_110000_test.json").write_text("{}")
        (tmp_path / "other_file.json").write_text("{}")  # Should be ignored

        result = handler.get_pending_fallbacks()

        assert len(result) == 2
        assert result[0].name == "fallback_20240101_110000_test.json"
        assert result[1].name == "fallback_20240101_120000_test.json"


class TestStagedMessages:
    """Tests for staged message management."""

    def test_staged_message_count(self):
        """Should track staged message count."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        assert handler.staged_message_count == 0

        handler._staged_messages = [
            Message(role="user", content="Message 1"),
            Message(role="assistant", content="Message 2"),
        ]

        assert handler.staged_message_count == 2

    def test_clear_staged_messages(self):
        """Should clear all staged messages."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        handler._staged_messages = [Message(role="user", content="Test")]
        handler.clear_staged_messages()

        assert handler.staged_message_count == 0

    @pytest.mark.asyncio
    async def test_flush_staged_messages_empty(self):
        """Should return True when no staged messages."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        result = await handler.flush_staged_messages()

        assert result is True

    @pytest.mark.asyncio
    async def test_flush_staged_messages_to_mnemosyne(self):
        """Should flush staged messages to Mnemosyne."""
        mock_elpis = MagicMock()
        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True
        mock_mnemosyne.store_memory = AsyncMock()

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        handler._staged_messages = [Message(role="user", content="Test")]
        result = await handler.flush_staged_messages()

        assert result is True
        assert handler.staged_message_count == 0

    @pytest.mark.asyncio
    async def test_flush_staged_messages_to_fallback(self, tmp_path):
        """Should flush staged messages to fallback when Mnemosyne unavailable."""
        mock_elpis = MagicMock()
        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
            fallback_dir=tmp_path,
        )

        handler._staged_messages = [Message(role="user", content="Test")]
        result = await handler.flush_staged_messages()

        assert result is True
        assert handler.staged_message_count == 0
        assert len(list(tmp_path.glob("fallback_*.json"))) == 1


class TestStoreConversationSummary:
    """Tests for store_conversation_summary method."""

    @pytest.mark.asyncio
    async def test_store_conversation_summary_success(self):
        """Should generate and store summary."""
        mock_elpis = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "Summary of conversation"
        mock_elpis.generate = AsyncMock(return_value=mock_result)

        mock_mnemosyne = MagicMock()
        mock_mnemosyne.is_connected = True
        mock_mnemosyne.store_memory = AsyncMock()

        handler = MemoryHandler(
            mnemosyne_client=mock_mnemosyne,
            elpis_client=mock_elpis,
        )

        messages = [Message(role="user", content="Hello")]
        result = await handler.store_conversation_summary(messages)

        assert result is True

    @pytest.mark.asyncio
    async def test_store_conversation_summary_no_summary_generated(self):
        """Should return False when no summary generated."""
        mock_elpis = MagicMock()
        mock_result = MagicMock()
        mock_result.content = ""
        mock_elpis.generate = AsyncMock(return_value=mock_result)

        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=mock_elpis,
        )

        messages = [Message(role="user", content="Hello")]
        result = await handler.store_conversation_summary(messages)

        assert result is False
