"""
Integration tests for the MCP server.

Tests cover:
- Server initialization
- MCP tool listing and execution
- Emotional state management
- Resource reading
- Integration with LLM inference
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from elpis.emotion.state import EmotionalState
from elpis.emotion.regulation import HomeostasisRegulator
from elpis.server import ServerContext
import elpis.server as server_module


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock = MagicMock()
    mock.chat_completion = AsyncMock(return_value="Test response from LLM")
    mock.function_call = AsyncMock(return_value=[{
        "id": "call_test",
        "function": {"name": "test_tool", "arguments": "{}"}
    }])
    return mock


@pytest.fixture
def initialized_server(mock_llm):
    """Initialize server with mock components using ServerContext."""
    # Save original context
    original_context = server_module._context

    # Create test context
    emotion_state = EmotionalState()
    settings = MagicMock()
    settings.model = MagicMock()
    settings.model.temperature = 0.7
    settings.model.top_p = 0.9

    ctx = ServerContext(
        llm=mock_llm,
        emotion_state=emotion_state,
        regulator=HomeostasisRegulator(emotion_state),
        settings=settings,
    )
    server_module._context = ctx

    yield ctx

    # Restore original context
    server_module._context = original_context


class TestMCPServerTools:
    """Tests for MCP server tool functionality."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_tools_returns_expected_tools(self):
        """list_tools should return all expected tools."""
        tools = await server_module.list_tools()

        tool_names = [t.name for t in tools]
        assert "generate" in tool_names
        assert "function_call" in tool_names
        assert "update_emotion" in tool_names
        assert "reset_emotion" in tool_names
        assert "get_emotion" in tool_names
        assert "get_capabilities" in tool_names
        # Streaming tools
        assert "generate_stream_start" in tool_names
        assert "generate_stream_read" in tool_names
        assert "generate_stream_cancel" in tool_names
        assert len(tools) == 9

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_emotion_returns_state(self, initialized_server):
        """get_emotion tool should return current emotional state."""
        result = await server_module.call_tool("get_emotion", {})

        assert len(result) == 1
        assert result[0].type == "text"
        assert "valence" in result[0].text
        assert "arousal" in result[0].text
        assert "quadrant" in result[0].text

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_update_emotion_changes_state(self, initialized_server):
        """update_emotion tool should modify emotional state."""
        # Initial state
        initial_valence = initialized_server.emotion_state.valence

        # Trigger success event
        await server_module.call_tool("update_emotion", {
            "event_type": "success",
            "intensity": 1.0
        })

        # State should have changed
        assert initialized_server.emotion_state.valence > initial_valence

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_reset_emotion_returns_to_baseline(self, initialized_server):
        """reset_emotion tool should return state to baseline."""
        # Modify state
        initialized_server.emotion_state.shift(0.5, 0.5)
        assert initialized_server.emotion_state.valence == 0.5

        # Reset
        await server_module.call_tool("reset_emotion", {})

        # Should be back to baseline (0.0 default)
        assert initialized_server.emotion_state.valence == 0.0
        assert initialized_server.emotion_state.arousal == 0.0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_uses_emotional_modulation(self, initialized_server, mock_llm):
        """generate tool should apply emotional parameter modulation."""
        # Set high arousal state (should lower temperature)
        initialized_server.emotion_state.shift(0.0, 0.8)

        await server_module.call_tool("generate", {
            "messages": [{"role": "user", "content": "Test message"}],
            "emotional_modulation": True
        })

        # Check that chat_completion was called
        mock_llm.chat_completion.assert_called_once()

        # Check that temperature was modulated (lower due to high arousal)
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs["temperature"] < 0.7  # Base is 0.7

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_without_modulation(self, initialized_server, mock_llm):
        """generate tool should respect emotional_modulation=False."""
        initialized_server.emotion_state.shift(0.0, 0.8)

        await server_module.call_tool("generate", {
            "messages": [{"role": "user", "content": "Test"}],
            "temperature": 0.9,  # Override
            "emotional_modulation": False
        })

        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.9  # Should use override

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_function_call_tool(self, initialized_server, mock_llm):
        """function_call tool should generate tool calls."""
        result = await server_module.call_tool("function_call", {
            "messages": [{"role": "user", "content": "Use a tool"}],
            "tools": [{"type": "function", "function": {"name": "test"}}]
        })

        assert len(result) == 1
        assert "tool_calls" in result[0].text

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_unknown_tool_returns_error(self, initialized_server):
        """Unknown tool should return error response."""
        result = await server_module.call_tool("nonexistent_tool", {})

        assert len(result) == 1
        assert "error" in result[0].text
        assert "Unknown tool" in result[0].text


class TestMCPServerResources:
    """Tests for MCP server resource functionality."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_resources(self):
        """list_resources should return expected resources."""
        resources = await server_module.list_resources()

        uris = [str(r.uri) for r in resources]
        assert "emotion://state" in uris
        assert "emotion://events" in uris
        assert len(resources) == 2

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_read_emotion_state_resource(self, initialized_server):
        """Reading emotion://state should return current state."""
        result = await server_module.read_resource("emotion://state")

        assert "valence" in result
        assert "arousal" in result
        assert "quadrant" in result

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_read_events_resource(self, initialized_server):
        """Reading emotion://events should return available events."""
        result = await server_module.read_resource("emotion://events")

        assert "success" in result
        assert "failure" in result
        assert "frustration" in result

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_read_unknown_resource_raises(self, initialized_server):
        """Reading unknown resource should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown resource"):
            await server_module.read_resource("emotion://unknown")


class TestServerInitialization:
    """Tests for server initialization."""

    def test_get_context_raises_when_not_initialized(self):
        """get_context should raise if server not initialized."""
        # Save and clear context
        original_context = server_module._context
        server_module._context = None

        try:
            with pytest.raises(RuntimeError, match="Server not initialized"):
                server_module.get_context()
        finally:
            server_module._context = original_context

    @pytest.mark.integration
    def test_initialize_creates_server_context(self):
        """initialize() should create ServerContext with all components."""
        # Save original context
        original_context = server_module._context

        try:
            # Mock the backend creation
            mock_llm = MagicMock()
            with patch("elpis.server.create_backend", return_value=mock_llm):
                ctx = server_module.initialize()

                assert ctx is not None
                assert isinstance(ctx, ServerContext)
                assert ctx.llm == mock_llm
                assert isinstance(ctx.emotion_state, EmotionalState)
                assert isinstance(ctx.regulator, HomeostasisRegulator)
                assert ctx.settings is not None
        finally:
            # Restore
            server_module._context = original_context


class TestEmotionalModulation:
    """Integration tests for emotional modulation of inference."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_high_arousal_lowers_temperature(self, initialized_server, mock_llm):
        """High arousal should result in lower temperature (focused)."""
        initialized_server.emotion_state.shift(0.0, 1.0)  # Max arousal

        await server_module.call_tool("generate", {
            "messages": [{"role": "user", "content": "Test"}]
        })

        call_kwargs = mock_llm.chat_completion.call_args[1]
        # Base temp is 0.7, high arousal should lower it
        assert call_kwargs["temperature"] == 0.5  # 0.7 - 0.2 * 1.0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_low_arousal_raises_temperature(self, initialized_server, mock_llm):
        """Low arousal should result in higher temperature (exploratory)."""
        initialized_server.emotion_state.shift(0.0, -1.0)  # Min arousal

        await server_module.call_tool("generate", {
            "messages": [{"role": "user", "content": "Test"}]
        })

        call_kwargs = mock_llm.chat_completion.call_args[1]
        # Base temp is 0.7, low arousal should raise it
        assert call_kwargs["temperature"] == 0.9  # 0.7 + 0.2 * 1.0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_high_valence_raises_top_p(self, initialized_server, mock_llm):
        """High valence should result in higher top_p (broader sampling)."""
        initialized_server.emotion_state.shift(1.0, 0.0)  # Max valence

        await server_module.call_tool("generate", {
            "messages": [{"role": "user", "content": "Test"}]
        })

        call_kwargs = mock_llm.chat_completion.call_args[1]
        # Base top_p is 0.9, high valence should raise it
        assert call_kwargs["top_p"] == 1.0  # 0.9 + 0.1 * 1.0 = 1.0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_response_updates_emotional_state(self, initialized_server, mock_llm):
        """LLM response should update emotional state via regulator."""
        mock_llm.chat_completion = AsyncMock(return_value="Task completed successfully!")

        initial_valence = initialized_server.emotion_state.valence

        await server_module.call_tool("generate", {
            "messages": [{"role": "user", "content": "Test"}]
        })

        # "successfully" should trigger positive emotion
        assert initialized_server.emotion_state.valence > initial_valence


class TestErrorHandling:
    """Tests for error handling in the server."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tool_exception_returns_error_response(self, initialized_server, mock_llm):
        """Tool exceptions should be caught and returned as error."""
        mock_llm.chat_completion = AsyncMock(side_effect=Exception("Test error"))

        result = await server_module.call_tool("generate", {
            "messages": [{"role": "user", "content": "Test"}]
        })

        assert len(result) == 1
        assert "error" in result[0].text
        assert "Test error" in result[0].text
