"""Unit tests for context manager."""

import pytest

from psyche.core.context_manager import (
    ContextConfig,
    ContextManager,
)
from psyche.memory.compaction import Message, create_message


class TestContextConfig:
    """Tests for ContextConfig dataclass."""

    def test_default_values(self):
        """ContextConfig should have sensible defaults."""
        config = ContextConfig()
        assert config.max_context_tokens == 24000
        assert config.reserve_tokens == 4000
        assert config.enable_checkpoints is True
        assert config.checkpoint_interval == 20

    def test_custom_values(self):
        """ContextConfig should accept custom values."""
        config = ContextConfig(
            max_context_tokens=8000,
            reserve_tokens=1000,
            enable_checkpoints=False,
            checkpoint_interval=10,
        )
        assert config.max_context_tokens == 8000
        assert config.reserve_tokens == 1000
        assert config.enable_checkpoints is False
        assert config.checkpoint_interval == 10


class TestContextManagerInit:
    """Tests for ContextManager initialization."""

    def test_default_initialization(self):
        """ContextManager should initialize with defaults."""
        manager = ContextManager()
        assert manager.config is not None
        assert manager.total_tokens == 0
        assert manager.message_count == 0
        assert len(manager.messages) == 0

    def test_custom_config(self):
        """ContextManager should accept custom config."""
        config = ContextConfig(max_context_tokens=5000, reserve_tokens=500)
        manager = ContextManager(config=config)
        assert manager.config.max_context_tokens == 5000
        assert manager.config.reserve_tokens == 500

    def test_available_tokens(self):
        """available_tokens should reflect max minus reserve."""
        config = ContextConfig(max_context_tokens=10000, reserve_tokens=2000)
        manager = ContextManager(config=config)
        assert manager.available_tokens == 8000


class TestAddMessage:
    """Tests for adding messages."""

    def test_add_single_message(self):
        """add_message should add message to context."""
        manager = ContextManager()
        result = manager.add_message("user", "Hello world")

        assert len(manager.messages) == 1
        assert manager.messages[0].role == "user"
        assert manager.messages[0].content == "Hello world"
        assert result is None  # No compaction needed

    def test_add_multiple_messages(self):
        """Multiple messages should be added in order."""
        manager = ContextManager()
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there")
        manager.add_message("user", "How are you?")

        assert len(manager.messages) == 3
        assert manager.messages[0].role == "user"
        assert manager.messages[1].role == "assistant"
        assert manager.messages[2].role == "user"

    def test_message_count_increments(self):
        """message_count should increment with each add."""
        manager = ContextManager()
        assert manager.message_count == 0

        manager.add_message("user", "Hello")
        assert manager.message_count == 1

        manager.add_message("assistant", "Hi")
        assert manager.message_count == 2

    def test_add_raw_message(self):
        """add_raw_message should accept Message objects."""
        manager = ContextManager()
        msg = create_message("user", "Test message")
        result = manager.add_raw_message(msg)

        assert len(manager.messages) == 1
        assert manager.messages[0] is msg
        assert manager.message_count == 1
        assert result is None

    def test_total_tokens_increases(self):
        """total_tokens should increase when messages are added."""
        manager = ContextManager()
        assert manager.total_tokens == 0

        manager.add_message("user", "Hello world this is a test message")
        assert manager.total_tokens > 0


class TestCompactionTriggering:
    """Tests for compaction behavior."""

    def test_compaction_triggered_when_limit_exceeded(self):
        """Compaction should trigger when token limit is exceeded."""
        config = ContextConfig(max_context_tokens=100, reserve_tokens=20)
        manager = ContextManager(config=config)

        # Add messages to exceed the limit
        results = []
        for i in range(10):
            result = manager.add_message("user", f"This is message number {i} with some content")
            results.append(result)

        # At least one compaction should have occurred
        compaction_results = [r for r in results if r is not None]
        assert len(compaction_results) > 0
        assert manager.total_tokens <= manager.available_tokens

    def test_compaction_result_contains_dropped_messages(self):
        """CompactionResult should contain dropped messages."""
        config = ContextConfig(max_context_tokens=50, reserve_tokens=10)
        manager = ContextManager(config=config)

        # Add messages to trigger compaction
        for i in range(5):
            result = manager.add_message("user", f"Message {i} content here")
            if result is not None:
                assert result.messages_compacted > 0
                assert result.dropped_messages is not None


class TestSystemPrompt:
    """Tests for system prompt handling."""

    def test_set_system_prompt(self):
        """set_system_prompt should add system message."""
        manager = ContextManager()
        manager.set_system_prompt("You are a helpful assistant.")

        assert len(manager.messages) == 1
        assert manager.messages[0].role == "system"
        assert manager.messages[0].content == "You are a helpful assistant."

    def test_update_system_prompt(self):
        """update_system_prompt should modify existing system message."""
        manager = ContextManager()
        manager.set_system_prompt("Original prompt")
        manager.add_message("user", "Hello")

        manager.update_system_prompt("Updated prompt")

        # Should still have 2 messages (not 3)
        assert len(manager.messages) == 2
        # First message should be updated
        system_msgs = [m for m in manager.messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "Updated prompt"

    def test_update_system_prompt_when_none_exists(self):
        """update_system_prompt should add if no system message exists."""
        manager = ContextManager()
        manager.add_message("user", "Hello")

        manager.update_system_prompt("New system prompt")

        # Should have added a system message
        system_msgs = [m for m in manager.messages if m.role == "system"]
        assert len(system_msgs) == 1


class TestCheckpoint:
    """Tests for checkpoint functionality."""

    def test_should_checkpoint_respects_interval(self):
        """should_checkpoint should return True at interval."""
        config = ContextConfig(checkpoint_interval=5)
        manager = ContextManager(config=config)

        for i in range(1, 6):
            manager.add_message("user", f"Message {i}")
            if i < 5:
                assert manager.should_checkpoint() is False
            else:
                assert manager.should_checkpoint() is True

    def test_should_checkpoint_disabled(self):
        """should_checkpoint should return False when disabled."""
        config = ContextConfig(enable_checkpoints=False, checkpoint_interval=5)
        manager = ContextManager(config=config)

        for i in range(10):
            manager.add_message("user", f"Message {i}")
            assert manager.should_checkpoint() is False

    def test_get_checkpoint_messages_excludes_system(self):
        """get_checkpoint_messages should exclude system messages."""
        manager = ContextManager()
        manager.set_system_prompt("System prompt")
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")

        checkpoint_msgs = manager.get_checkpoint_messages()

        assert len(checkpoint_msgs) == 2
        assert all(m.role != "system" for m in checkpoint_msgs)


class TestClear:
    """Tests for context clearing."""

    def test_clear_removes_messages(self):
        """clear should remove all messages."""
        manager = ContextManager()
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")

        manager.clear()

        assert len(manager.messages) == 0
        assert manager.total_tokens == 0
        assert manager.message_count == 0

    def test_clear_preserves_system_prompt(self):
        """clear should re-add system prompt if one was set."""
        manager = ContextManager()
        manager.set_system_prompt("You are helpful.")
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")

        manager.clear()

        # Should have just the system prompt
        assert len(manager.messages) == 1
        assert manager.messages[0].role == "system"
        assert manager.messages[0].content == "You are helpful."

    def test_clear_resets_message_count(self):
        """clear should reset message count."""
        manager = ContextManager()
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")
        assert manager.message_count == 2

        manager.clear()
        assert manager.message_count == 0


class TestGetSummary:
    """Tests for context summary."""

    def test_get_summary_empty_context(self):
        """get_summary should work on empty context."""
        manager = ContextManager()
        summary = manager.get_summary()

        assert summary["message_count"] == 0
        assert summary["total_tokens"] == 0
        assert summary["messages_added"] == 0
        assert summary["has_system_prompt"] is False

    def test_get_summary_with_messages(self):
        """get_summary should reflect context state."""
        manager = ContextManager()
        manager.set_system_prompt("System prompt")
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there")

        summary = manager.get_summary()

        assert summary["message_count"] == 3  # system + 2 messages
        assert summary["total_tokens"] > 0
        assert summary["messages_added"] == 2  # add_message called twice
        assert summary["has_system_prompt"] is True
        assert "available_tokens" in summary


class TestGetApiMessages:
    """Tests for API message formatting."""

    def test_get_api_messages_format(self):
        """get_api_messages should return API-compatible format."""
        manager = ContextManager()
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there")

        api_messages = manager.get_api_messages()

        assert len(api_messages) == 2
        assert api_messages[0] == {"role": "user", "content": "Hello"}
        assert api_messages[1] == {"role": "assistant", "content": "Hi there"}

    def test_get_api_messages_with_system(self):
        """get_api_messages should include system messages."""
        manager = ContextManager()
        manager.set_system_prompt("You are helpful.")
        manager.add_message("user", "Hello")

        api_messages = manager.get_api_messages()

        assert len(api_messages) == 2
        assert api_messages[0]["role"] == "system"
        assert api_messages[1]["role"] == "user"


class TestMarkImportant:
    """Tests for message importance marking."""

    def test_mark_important(self):
        """mark_important should mark message metadata."""
        manager = ContextManager()
        manager.add_message("user", "Important info")
        manager.add_message("assistant", "Response")

        manager.mark_important(0)

        # Verify via compactor (internal access for testing)
        assert manager._compactor._messages[0].metadata.get("important") is True

    def test_mark_important_invalid_index(self):
        """mark_important should handle invalid index gracefully."""
        manager = ContextManager()
        manager.add_message("user", "Hello")

        # Should not raise
        manager.mark_important(99)
        manager.mark_important(-1)


class TestWithSummarization:
    """Tests for context manager with summarization function."""

    def test_summarize_fn_passed_to_compactor(self):
        """summarize_fn should be used during compaction."""
        summaries_called = []

        def mock_summarize(messages):
            summaries_called.append(len(messages))
            return "Summary of conversation"

        config = ContextConfig(max_context_tokens=50, reserve_tokens=10)
        manager = ContextManager(config=config, summarize_fn=mock_summarize)

        # Add enough messages to trigger compaction with summarization
        for i in range(10):
            manager.add_message("user", f"Message {i} with content")

        # Summarization should have been called
        assert len(summaries_called) > 0
