"""
Integration tests for Phase 3 reasoning workflow.

Tests verify:
- Reasoning mode enabled by default
- REASONING_PROMPT added to system prompt when enabled
- <reasoning> tags extracted and sent to thought callback
- Response cleaned of reasoning tags
- Toggle updates system prompt
- Backwards compatibility with <thinking> tags
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from psyche.mcp.client import ElpisClient, EmotionalState, GenerationResult
from psyche.memory.server import (
    MemoryServer,
    ServerConfig,
    ServerState,
    ThoughtEvent,
    REASONING_PROMPT,
)
from psyche.memory.reasoning import parse_reasoning


class MockElpisClient:
    """Mock ElpisClient for testing reasoning workflow."""

    def __init__(self):
        self.generate_responses = []
        self.response_index = 0
        self.emotion_state = EmotionalState()
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def generate_stream(
        self,
        messages,
        max_tokens=2048,
        temperature=None,
        emotional_modulation=True,
        poll_interval=0.05,
    ):
        """Yield response content as tokens."""
        if self.response_index < len(self.generate_responses):
            content = self.generate_responses[self.response_index]
            self.response_index += 1
        else:
            content = "Default response"

        for char in content:
            yield char

    async def get_emotion(self):
        return self.emotion_state

    async def update_emotion(self, event_type: str, intensity: float = 1.0):
        return self.emotion_state

    def connect(self):
        return MockConnectionContext(self)


class MockConnectionContext:
    """Mock async context manager for client connection."""

    def __init__(self, client):
        self.client = client

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_client():
    """Create a mock Elpis client."""
    return MockElpisClient()


@pytest.fixture
def server_config():
    """Create server config for testing."""
    return ServerConfig(
        idle_think_interval=60.0,  # Long interval to avoid idle thinking
        max_context_tokens=4000,
        reserve_tokens=500,
        allow_idle_tools=False,
        auto_storage=False,  # Disable auto-storage for reasoning tests
    )


class TestReasoningModeDefault:
    """Tests for reasoning mode default state."""

    def test_reasoning_enabled_by_default(self, mock_client, server_config):
        """Reasoning mode should be enabled by default."""
        server = MemoryServer(mock_client, server_config)

        assert server.reasoning_enabled is True
        assert server._reasoning_enabled is True

    def test_reasoning_prompt_in_system_message_when_enabled(self, mock_client, server_config):
        """REASONING_PROMPT should be in system prompt when enabled."""
        server = MemoryServer(mock_client, server_config)

        # Check that the system prompt contains reasoning instructions
        assert "<reasoning>" in server._system_prompt
        assert "Reasoning Mode" in server._system_prompt


class TestReasoningTagParsing:
    """Tests for parsing reasoning tags from responses."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_reasoning_tags_parsed_from_response(self, mock_client, server_config):
        """<reasoning> tags should be extracted and sent to thought callback."""
        mock_client.generate_responses = [
            "<reasoning>Let me think about this problem.\n1. Analyze\n2. Solve</reasoning>\n\nHere's my answer."
        ]

        thought_events = []
        def on_thought(event: ThoughtEvent):
            thought_events.append(event)

        server = MemoryServer(mock_client, server_config, on_thought=on_thought)
        server.client = mock_client

        await server._process_user_input("Help me solve this")

        # Check that a reasoning thought was emitted
        assert len(thought_events) == 1
        assert thought_events[0].thought_type == "reasoning"
        assert "Let me think about this problem" in thought_events[0].content
        assert "1. Analyze" in thought_events[0].content

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_response_cleaned_of_reasoning_tags(self, mock_client, server_config):
        """Final response should have reasoning tags removed."""
        mock_client.generate_responses = [
            "<reasoning>Internal thought</reasoning>\n\nClean response to user."
        ]

        responses = []
        def on_response(content: str):
            responses.append(content)

        server = MemoryServer(mock_client, server_config, on_response=on_response)
        server.client = mock_client

        await server._process_user_input("Test input")

        # Check response callback received cleaned content
        assert len(responses) == 1
        assert "<reasoning>" not in responses[0]
        assert "Clean response to user" in responses[0]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_legacy_thinking_tags_still_work(self, mock_client, server_config):
        """<thinking> tags should still be parsed for backwards compatibility."""
        mock_client.generate_responses = [
            "<thinking>Legacy thought process</thinking>\n\nResponse here."
        ]

        thought_events = []
        def on_thought(event: ThoughtEvent):
            thought_events.append(event)

        server = MemoryServer(mock_client, server_config, on_thought=on_thought)
        server.client = mock_client

        await server._process_user_input("Test")

        assert len(thought_events) == 1
        assert "Legacy thought process" in thought_events[0].content

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_empty_reasoning_tags_handled(self, mock_client, server_config):
        """Empty reasoning tags should not crash."""
        mock_client.generate_responses = [
            "<reasoning></reasoning>Response after empty tags."
        ]

        thought_events = []
        def on_thought(event: ThoughtEvent):
            thought_events.append(event)

        server = MemoryServer(mock_client, server_config, on_thought=on_thought)
        server.client = mock_client

        # Should not raise
        await server._process_user_input("Test")

        # Empty reasoning still triggers thought event (with empty content)
        assert len(thought_events) == 1
        assert thought_events[0].content == ""


class TestReasoningModeToggle:
    """Tests for toggling reasoning mode."""

    def test_reasoning_toggle_updates_flag(self, mock_client, server_config):
        """set_reasoning_mode should update the flag."""
        server = MemoryServer(mock_client, server_config)

        assert server.reasoning_enabled is True

        server.set_reasoning_mode(False)
        assert server.reasoning_enabled is False

        server.set_reasoning_mode(True)
        assert server.reasoning_enabled is True

    def test_reasoning_toggle_updates_system_prompt(self, mock_client, server_config):
        """Toggling reasoning mode should update the system prompt."""
        server = MemoryServer(mock_client, server_config)

        # Initially enabled
        assert "<reasoning>" in server._system_prompt

        # Disable
        server.set_reasoning_mode(False)
        assert "<reasoning>" not in server._system_prompt
        assert "Reasoning Mode" not in server._system_prompt

        # Re-enable
        server.set_reasoning_mode(True)
        assert "<reasoning>" in server._system_prompt

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_no_thought_callback_when_disabled(self, mock_client, server_config):
        """When reasoning disabled, no reasoning thoughts should be extracted."""
        mock_client.generate_responses = [
            "<reasoning>This should be ignored</reasoning>\n\nResponse."
        ]

        thought_events = []
        def on_thought(event: ThoughtEvent):
            thought_events.append(event)

        server = MemoryServer(mock_client, server_config, on_thought=on_thought)
        server.client = mock_client
        server.set_reasoning_mode(False)

        await server._process_user_input("Test")

        # No reasoning thoughts when disabled
        reasoning_thoughts = [e for e in thought_events if e.thought_type == "reasoning"]
        assert len(reasoning_thoughts) == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_response_includes_tags_when_disabled(self, mock_client, server_config):
        """When reasoning disabled, tags should remain in response."""
        mock_client.generate_responses = [
            "<reasoning>Not extracted</reasoning>Response."
        ]

        responses = []
        def on_response(content: str):
            responses.append(content)

        server = MemoryServer(mock_client, server_config, on_response=on_response)
        server.client = mock_client
        server.set_reasoning_mode(False)

        await server._process_user_input("Test")

        # Tags should still be in response when reasoning mode is off
        assert len(responses) == 1
        assert "<reasoning>" in responses[0]
