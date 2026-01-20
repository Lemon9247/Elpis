"""Tests for memory tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from psyche.shared.constants import MEMORY_SUMMARY_LENGTH
from psyche.tools.implementations.memory_tools import MemoryTools
from psyche.tools.tool_definitions import RecallMemoryInput, StoreMemoryInput


class TestRecallMemoryInput:
    """Test RecallMemoryInput validation."""

    def test_valid_input(self):
        """Test valid recall memory input."""
        input_data = RecallMemoryInput(query="test query")
        assert input_data.query == "test query"
        assert input_data.n_results == 5  # default

    def test_custom_n_results(self):
        """Test custom n_results."""
        input_data = RecallMemoryInput(query="test", n_results=10)
        assert input_data.n_results == 10

    def test_empty_query_rejected(self):
        """Test empty query is rejected."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            RecallMemoryInput(query="")

    def test_whitespace_query_rejected(self):
        """Test whitespace-only query is rejected."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            RecallMemoryInput(query="   ")

    def test_n_results_bounds(self):
        """Test n_results validation."""
        with pytest.raises(ValueError):
            RecallMemoryInput(query="test", n_results=0)
        with pytest.raises(ValueError):
            RecallMemoryInput(query="test", n_results=25)


class TestStoreMemoryInput:
    """Test StoreMemoryInput validation."""

    def test_valid_input(self):
        """Test valid store memory input."""
        input_data = StoreMemoryInput(content="test content")
        assert input_data.content == "test content"
        assert input_data.memory_type == "episodic"  # default
        assert input_data.summary is None

    def test_all_fields(self):
        """Test all fields populated."""
        input_data = StoreMemoryInput(
            content="test content",
            summary="test summary",
            memory_type="semantic",
            tags=["tag1", "tag2"],
        )
        assert input_data.content == "test content"
        assert input_data.summary == "test summary"
        assert input_data.memory_type == "semantic"
        assert input_data.tags == ["tag1", "tag2"]

    def test_empty_content_rejected(self):
        """Test empty content is rejected."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            StoreMemoryInput(content="")

    def test_invalid_memory_type(self):
        """Test invalid memory type is rejected."""
        with pytest.raises(ValueError, match="memory_type must be one of"):
            StoreMemoryInput(content="test", memory_type="invalid")

    def test_valid_memory_types(self):
        """Test all valid memory types."""
        for mem_type in ["episodic", "semantic", "procedural", "emotional"]:
            input_data = StoreMemoryInput(content="test", memory_type=mem_type)
            assert input_data.memory_type == mem_type


class TestMemoryToolsRecall:
    """Test MemoryTools recall functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Mnemosyne client."""
        client = AsyncMock()
        client.search_memories = AsyncMock(return_value=[
            {
                "content": "Test memory content",
                "summary": "Test summary",
                "memory_type": "episodic",
                "tags": ["test"],
                "emotional_context": {"valence": 0.5, "arousal": 0.3},
                "created_at": "2026-01-14T12:00:00",
                "distance": 0.1,
            }
        ])
        return client

    @pytest.fixture
    def memory_tools(self, mock_client):
        """Create MemoryTools instance with mock client."""
        return MemoryTools(mnemosyne_client=mock_client)

    async def test_recall_memory_success(self, memory_tools, mock_client):
        """Test successful memory recall."""
        result = await memory_tools.recall_memory(query="test query")

        assert result["success"] is True
        assert result["query"] == "test query"
        assert result["count"] == 1
        assert len(result["memories"]) == 1

        memory = result["memories"][0]
        assert memory["content"] == "Test memory content"
        assert memory["summary"] == "Test summary"

        mock_client.search_memories.assert_called_once_with(
            query="test query",
            n_results=5,
        )

    async def test_recall_memory_custom_n_results(self, memory_tools, mock_client):
        """Test recall with custom n_results."""
        await memory_tools.recall_memory(query="test", n_results=10)

        mock_client.search_memories.assert_called_once_with(
            query="test",
            n_results=10,
        )

    async def test_recall_memory_error_handling(self, mock_client):
        """Test error handling in recall."""
        mock_client.search_memories = AsyncMock(side_effect=Exception("Connection failed"))
        memory_tools = MemoryTools(mnemosyne_client=mock_client)

        result = await memory_tools.recall_memory(query="test")

        assert result["success"] is False
        assert "Connection failed" in result["error"]


class TestMemoryToolsStore:
    """Test MemoryTools store functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Mnemosyne client."""
        client = AsyncMock()
        client.store_memory = AsyncMock(return_value={
            "memory_id": "test-id-123",
            "status": "stored",
        })
        return client

    @pytest.fixture
    def mock_emotion_fn(self):
        """Create a mock emotion getter."""
        emotion = MagicMock()
        emotion.valence = 0.5
        emotion.arousal = 0.3
        emotion.quadrant = "content"

        async def get_emotion():
            return emotion

        return get_emotion

    @pytest.fixture
    def memory_tools(self, mock_client, mock_emotion_fn):
        """Create MemoryTools instance with mock client."""
        return MemoryTools(
            mnemosyne_client=mock_client,
            get_emotion_fn=mock_emotion_fn,
        )

    async def test_store_memory_success(self, memory_tools, mock_client):
        """Test successful memory storage."""
        result = await memory_tools.store_memory(content="Test content to store")

        assert result["success"] is True
        assert result["memory_id"] == "test-id-123"
        assert "Test content to store" in result["summary"]

        mock_client.store_memory.assert_called_once()
        call_args = mock_client.store_memory.call_args
        assert call_args.kwargs["content"] == "Test content to store"
        assert call_args.kwargs["memory_type"] == "episodic"

    async def test_store_memory_with_custom_fields(self, memory_tools, mock_client):
        """Test storage with custom fields."""
        result = await memory_tools.store_memory(
            content="Test content",
            summary="Custom summary",
            memory_type="semantic",
            tags=["important", "fact"],
        )

        assert result["success"] is True

        call_args = mock_client.store_memory.call_args
        assert call_args.kwargs["summary"] == "Custom summary"
        assert call_args.kwargs["memory_type"] == "semantic"
        assert call_args.kwargs["tags"] == ["important", "fact"]

    async def test_store_memory_includes_emotional_context(self, memory_tools, mock_client):
        """Test that emotional context is included."""
        await memory_tools.store_memory(content="Test content")

        call_args = mock_client.store_memory.call_args
        emotional_context = call_args.kwargs["emotional_context"]
        assert emotional_context["valence"] == 0.5
        assert emotional_context["arousal"] == 0.3
        assert emotional_context["quadrant"] == "content"

    async def test_store_memory_without_emotion_fn(self, mock_client):
        """Test storage without emotion function."""
        memory_tools = MemoryTools(mnemosyne_client=mock_client)

        await memory_tools.store_memory(content="Test content")

        call_args = mock_client.store_memory.call_args
        assert call_args.kwargs["emotional_context"] is None

    async def test_store_memory_error_handling(self, mock_client, mock_emotion_fn):
        """Test error handling in store."""
        mock_client.store_memory = AsyncMock(side_effect=Exception("Storage failed"))
        memory_tools = MemoryTools(
            mnemosyne_client=mock_client,
            get_emotion_fn=mock_emotion_fn,
        )

        result = await memory_tools.store_memory(content="Test content")

        assert result["success"] is False
        assert "Storage failed" in result["error"]

    async def test_auto_generated_summary(self, memory_tools, mock_client):
        """Test auto-generated summary for long content."""
        # Create content longer than MEMORY_SUMMARY_LENGTH to trigger truncation
        long_content = "A" * (MEMORY_SUMMARY_LENGTH + 100)

        await memory_tools.store_memory(content=long_content)

        call_args = mock_client.store_memory.call_args
        summary = call_args.kwargs["summary"]
        assert len(summary) == MEMORY_SUMMARY_LENGTH + 3  # truncated + "..."
        assert summary.endswith("...")
