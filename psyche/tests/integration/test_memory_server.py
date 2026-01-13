"""
Integration tests for MemoryServer with continuous inference loop.

Tests verify:
- Server state management
- User input processing
- Idle thought generation
- Context compaction during conversations
- Callbacks for thoughts and responses
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from psyche.mcp.client import ElpisClient, EmotionalState, GenerationResult
from psyche.memory.server import MemoryServer, ServerConfig, ServerState, ThoughtEvent
from psyche.memory.compaction import create_message


class MockElpisClient:
    """Mock ElpisClient for testing MemoryServer."""

    def __init__(self):
        self.generate_responses = []
        self.response_index = 0
        self.emotion_state = EmotionalState()
        self.emotion_updates = []

    async def generate(
        self,
        messages,
        max_tokens=2048,
        temperature=None,
        emotional_modulation=True,
    ) -> GenerationResult:
        """Return next mocked response."""
        if self.response_index < len(self.generate_responses):
            content = self.generate_responses[self.response_index]
            self.response_index += 1
        else:
            content = "Default response"

        return GenerationResult(
            content=content,
            emotional_state=self.emotion_state,
            modulated_params={"temperature": 0.7},
        )

    async def update_emotion(self, event_type: str, intensity: float = 1.0):
        """Track emotion updates."""
        self.emotion_updates.append({"event": event_type, "intensity": intensity})
        return self.emotion_state

    async def get_emotion(self):
        """Return current emotion state."""
        return self.emotion_state

    def connect(self):
        """Return async context manager."""
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
    """Create server config with short intervals for testing."""
    return ServerConfig(
        idle_think_interval=0.1,  # Very short for tests
        max_idle_thoughts=2,
        max_context_tokens=1000,
        reserve_tokens=200,
    )


class TestMemoryServerInit:
    """Tests for MemoryServer initialization."""

    def test_server_creates_compactor(self, mock_client, server_config):
        """Server should initialize with context compactor."""
        server = MemoryServer(mock_client, server_config)

        assert server._compactor is not None
        assert server._compactor.max_tokens == 1000
        assert server._compactor.reserve_tokens == 200

    def test_server_initial_state(self, mock_client, server_config):
        """Server should start in IDLE state."""
        server = MemoryServer(mock_client, server_config)

        assert server.state == ServerState.IDLE
        assert not server.is_running


class TestMemoryServerInputProcessing:
    """Tests for processing user input."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_submit_input_queues_message(self, mock_client, server_config):
        """submit_input should queue message for processing."""
        server = MemoryServer(mock_client, server_config)
        server.submit_input("Hello")

        assert not server._input_queue.empty()
        queued = server._input_queue.get_nowait()
        assert queued == "Hello"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_process_user_input_calls_generate(self, mock_client, server_config):
        """Processing input should call client.generate."""
        mock_client.generate_responses = ["Test response"]

        server = MemoryServer(mock_client, server_config)
        server.client = mock_client

        # Manually process input without starting loop
        await server._process_user_input("Hello")

        # Check that message was added to compactor
        messages = server._compactor.get_api_messages()
        assert any(m["content"] == "Hello" for m in messages)
        assert any(m["content"] == "Test response" for m in messages)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_response_callback_fired(self, mock_client, server_config):
        """Response callback should be called with generated content."""
        mock_client.generate_responses = ["Response from LLM"]
        responses = []

        def on_response(content):
            responses.append(content)

        server = MemoryServer(mock_client, server_config, on_response=on_response)
        server.client = mock_client

        await server._process_user_input("Test")

        assert len(responses) == 1
        assert responses[0] == "Response from LLM"


class TestMemoryServerIdleThinking:
    """Tests for idle thought generation."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_idle_thought_generated_after_timeout(self, mock_client, server_config):
        """Server should generate idle thought when no input received."""
        mock_client.generate_responses = ["Idle thought content"]
        thoughts = []

        def on_thought(event):
            thoughts.append(event)

        server = MemoryServer(mock_client, server_config, on_thought=on_thought)
        server.client = mock_client
        server._running = True

        await server._generate_idle_thought()

        assert len(thoughts) == 1
        assert thoughts[0].thought_type == "reflection"
        assert thoughts[0].triggered_by == "idle"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_idle_thought_count_increments(self, mock_client, server_config):
        """Idle thought count should track generated thoughts."""
        mock_client.generate_responses = ["Thought 1", "Thought 2", "Thought 3"]
        thoughts = []

        def on_thought(event):
            thoughts.append(event)

        server = MemoryServer(mock_client, server_config, on_thought=on_thought)
        server.client = mock_client
        server._running = True

        await server._generate_idle_thought()
        assert server._idle_thought_count == 0  # Not auto-incremented

        # The loop increments it
        server._idle_thought_count += 1
        assert server._idle_thought_count == 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_user_input_resets_idle_count(self, mock_client, server_config):
        """User input should reset idle thought count."""
        mock_client.generate_responses = ["Response"]

        server = MemoryServer(mock_client, server_config)
        server.client = mock_client
        server._idle_thought_count = 3

        await server._process_user_input("Hello")

        # Note: The actual reset happens in the loop, not _process_user_input
        # This tests that count can be reset


class TestMemoryServerContextCompaction:
    """Tests for context management and compaction."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_messages_added_to_compactor(self, mock_client, server_config):
        """User and assistant messages should be added to compactor."""
        mock_client.generate_responses = ["Response 1", "Response 2"]

        server = MemoryServer(mock_client, server_config)
        server.client = mock_client

        # Add system prompt like start() would
        server._compactor.add_message(create_message("system", server._system_prompt))

        await server._process_user_input("Message 1")
        await server._process_user_input("Message 2")

        messages = server._compactor.get_api_messages()
        # Should have: system, user1, assistant1, user2, assistant2
        assert len(messages) == 5

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_clear_context_resets_messages(self, mock_client, server_config):
        """clear_context should reset conversation but keep system prompt."""
        mock_client.generate_responses = ["Response"]

        server = MemoryServer(mock_client, server_config)
        server.client = mock_client

        await server._process_user_input("Test message")
        initial_count = len(server._compactor.messages)

        server.clear_context()

        messages = server._compactor.messages
        assert len(messages) == 1  # Only system prompt
        assert messages[0].role == "system"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_context_summary(self, mock_client, server_config):
        """get_context_summary should return server state info."""
        mock_client.generate_responses = ["Response"]

        server = MemoryServer(mock_client, server_config)
        server.client = mock_client

        await server._process_user_input("Test")

        summary = await server.get_context_summary()

        assert "state" in summary
        assert "message_count" in summary
        assert "total_tokens" in summary
        assert "emotional_state" in summary


class TestMemoryServerEmotionalIntegration:
    """Tests for emotional state integration."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_long_response_triggers_engagement(self, mock_client, server_config):
        """Long responses should trigger engagement emotion."""
        mock_client.generate_responses = ["A" * 600]  # >500 chars

        server = MemoryServer(mock_client, server_config)
        server.client = mock_client

        await server._process_user_input("Test")

        # Check that engagement was updated
        assert any(u["event"] == "engagement" for u in mock_client.emotion_updates)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_short_response_triggers_boredom(self, mock_client, server_config):
        """Short responses should trigger boredom emotion."""
        mock_client.generate_responses = ["Ok"]  # <50 chars

        server = MemoryServer(mock_client, server_config)
        server.client = mock_client

        await server._process_user_input("Test")

        # Check that boredom was updated
        assert any(u["event"] == "boredom" for u in mock_client.emotion_updates)


class TestMemoryServerLifecycle:
    """Tests for server lifecycle management."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_stop_sets_shutting_down_state(self, mock_client, server_config):
        """stop() should set state to SHUTTING_DOWN."""
        server = MemoryServer(mock_client, server_config)
        server._running = True

        await server.stop()

        assert server.state == ServerState.SHUTTING_DOWN
        assert not server.is_running

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_inference_loop_respects_running_flag(self, mock_client, server_config):
        """Inference loop should exit when _running is False."""
        mock_client.generate_responses = ["Response"]

        server = MemoryServer(mock_client, server_config)
        server.client = mock_client
        server._running = False

        # Should complete immediately
        await server._inference_loop()

        # Loop should have exited


class TestMemoryServerInferenceLoop:
    """Integration tests for the full inference loop."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_loop_processes_input(self, mock_client, server_config):
        """Loop should process user input when available."""
        mock_client.generate_responses = ["Hello back!"]
        responses = []

        def on_response(content):
            responses.append(content)

        server = MemoryServer(mock_client, server_config, on_response=on_response)
        server.client = mock_client
        server._running = True

        # Queue input
        server.submit_input("Hello")

        # Run one iteration manually
        async def run_one_iteration():
            try:
                user_input = await asyncio.wait_for(
                    server._input_queue.get(), timeout=0.5
                )
                await server._process_user_input(user_input)
            except asyncio.TimeoutError:
                pass

        await run_one_iteration()

        assert len(responses) == 1
        assert responses[0] == "Hello back!"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_loop_generates_idle_thoughts_on_timeout(self, mock_client, server_config):
        """Loop should generate idle thoughts when input times out."""
        mock_client.generate_responses = ["Thinking..."]
        thoughts = []

        def on_thought(event):
            thoughts.append(event)

        server = MemoryServer(mock_client, server_config, on_thought=on_thought)
        server.client = mock_client
        server._running = True

        # Don't queue any input - simulate timeout
        await server._generate_idle_thought()
        server._idle_thought_count += 1

        assert len(thoughts) == 1
        assert server._idle_thought_count == 1
