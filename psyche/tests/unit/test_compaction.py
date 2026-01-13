"""Unit tests for context compaction."""

import pytest

from psyche.memory.compaction import (
    ContextCompactor,
    Message,
    create_message,
    estimate_tokens,
)


class TestMessage:
    """Tests for Message class."""

    def test_create_message(self):
        """create_message should create Message with estimated tokens."""
        msg = create_message("user", "Hello world!")
        assert msg.role == "user"
        assert msg.content == "Hello world!"
        assert msg.token_count > 0
        assert msg.timestamp > 0

    def test_message_to_dict(self):
        """to_dict should return API-compatible format."""
        msg = Message(role="user", content="Test message")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Test message"}


class TestEstimateTokens:
    """Tests for token estimation."""

    def test_estimate_tokens_short_text(self):
        """Short text should have reasonable token estimate."""
        tokens = estimate_tokens("Hello world")
        assert 2 <= tokens <= 5

    def test_estimate_tokens_long_text(self):
        """Long text should have proportional token estimate."""
        short = estimate_tokens("Hello")
        long = estimate_tokens("Hello world this is a longer test string")
        assert long > short

    def test_estimate_tokens_empty(self):
        """Empty string should estimate to 0 tokens."""
        tokens = estimate_tokens("")
        assert tokens == 0


class TestContextCompactor:
    """Tests for ContextCompactor class."""

    def test_init_default_values(self):
        """Compactor should initialize with sensible defaults."""
        compactor = ContextCompactor()
        assert compactor.max_tokens == 6000
        assert compactor.reserve_tokens == 2000
        assert compactor.available_tokens == 4000
        assert compactor.min_messages_to_keep == 4

    def test_add_message(self):
        """add_message should add to internal list."""
        compactor = ContextCompactor()
        msg = create_message("user", "Hello")
        result = compactor.add_message(msg)

        assert len(compactor.messages) == 1
        assert compactor.messages[0] == msg
        assert result is None  # No compaction needed

    def test_messages_returns_copy(self):
        """messages property should return a copy."""
        compactor = ContextCompactor()
        msg = create_message("user", "Hello")
        compactor.add_message(msg)

        messages = compactor.messages
        messages.clear()

        assert len(compactor.messages) == 1  # Original unchanged

    def test_total_tokens_tracks_additions(self):
        """total_tokens should track message tokens."""
        compactor = ContextCompactor()
        assert compactor.total_tokens == 0

        msg = create_message("user", "Hello world")
        compactor.add_message(msg)
        assert compactor.total_tokens == msg.token_count

    def test_clear_resets_state(self):
        """clear should reset all state."""
        compactor = ContextCompactor()
        compactor.add_message(create_message("user", "Hello"))
        compactor.add_message(create_message("assistant", "Hi there"))

        compactor.clear()

        assert len(compactor.messages) == 0
        assert compactor.total_tokens == 0

    def test_sliding_window_compaction(self):
        """Compactor should drop old messages when over limit."""
        compactor = ContextCompactor(
            max_tokens=100,
            reserve_tokens=20,
            min_messages_to_keep=2,
        )

        # Add messages that exceed the limit
        for i in range(10):
            compactor.add_message(
                Message(role="user", content=f"Message {i}", token_count=20)
            )

        # Should have compacted to stay under limit
        assert compactor.total_tokens <= compactor.available_tokens
        assert len(compactor.messages) >= compactor.min_messages_to_keep

    def test_compaction_returns_result(self):
        """Compaction should return CompactionResult when triggered."""
        compactor = ContextCompactor(
            max_tokens=50,
            reserve_tokens=10,
            min_messages_to_keep=2,
        )

        # Fill up
        compactor.add_message(Message(role="user", content="A", token_count=15))
        compactor.add_message(Message(role="assistant", content="B", token_count=15))

        # This should trigger compaction
        result = compactor.add_message(
            Message(role="user", content="C", token_count=20)
        )

        assert result is not None
        assert result.messages_compacted > 0

    def test_get_api_messages(self):
        """get_api_messages should return API format."""
        compactor = ContextCompactor()
        compactor.add_message(create_message("user", "Hello"))
        compactor.add_message(create_message("assistant", "Hi"))

        api_messages = compactor.get_api_messages()

        assert len(api_messages) == 2
        assert api_messages[0] == {"role": "user", "content": "Hello"}
        assert api_messages[1] == {"role": "assistant", "content": "Hi"}

    def test_add_messages_batch(self):
        """add_messages should add multiple messages."""
        compactor = ContextCompactor()
        messages = [
            create_message("user", "Hello"),
            create_message("assistant", "Hi there"),
            create_message("user", "How are you?"),
        ]

        compactor.add_messages(messages)

        assert len(compactor.messages) == 3


class TestCompactionWithSummarization:
    """Tests for compaction with summarization function."""

    def test_summarization_called_when_provided(self):
        """Summarization function should be called during compaction."""
        summaries_created = []

        def mock_summarize(messages):
            summaries_created.append(len(messages))
            return "Summary of conversation"

        compactor = ContextCompactor(
            max_tokens=50,
            reserve_tokens=10,
            min_messages_to_keep=2,
            summarize_fn=mock_summarize,
        )

        # Add enough messages to trigger compaction
        for i in range(5):
            compactor.add_message(
                Message(role="user", content=f"Message {i}", token_count=15)
            )

        assert len(summaries_created) > 0

    def test_summary_included_in_messages(self):
        """Summary should be included at start of messages."""

        def mock_summarize(messages):
            return "Summary"

        compactor = ContextCompactor(
            max_tokens=60,
            reserve_tokens=10,
            min_messages_to_keep=2,
            summarize_fn=mock_summarize,
        )

        # Add enough to trigger compaction
        for i in range(5):
            compactor.add_message(
                Message(role="user", content=f"Message {i}", token_count=15)
            )

        messages = compactor.messages
        # First message should be summary
        if compactor._summary:
            assert "[Previous conversation summary]" in messages[0].content
