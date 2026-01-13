"""
Integration tests for ElpisClient with Elpis MCP server.

Tests verify:
- Connection management via async context manager
- Tool calls through MCP protocol
- Resource reading
- Emotional state management
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from psyche.mcp.client import ElpisClient, EmotionalState, GenerationResult


class MockSession:
    """Mock MCP ClientSession for testing."""

    def __init__(self):
        self.call_tool = AsyncMock()
        self.read_resource = AsyncMock()
        self.initialize = AsyncMock()

    def set_tool_response(self, text: str):
        """Set up a tool response."""
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text=text)]
        self.call_tool.return_value = mock_result

    def set_resource_response(self, text: str):
        """Set up a resource response."""
        mock_result = MagicMock()
        mock_result.contents = [MagicMock(text=text)]
        self.read_resource.return_value = mock_result


@pytest.fixture
def mock_session():
    """Create a mock MCP session."""
    return MockSession()


@pytest.fixture
def connected_client(mock_session):
    """Create a client with mock session attached."""
    client = ElpisClient()
    client._session = mock_session
    client._connected = True
    return client


class TestElpisClientConnection:
    """Tests for client connection management."""

    def test_client_not_connected_initially(self):
        """Client should not be connected on creation."""
        client = ElpisClient()
        assert not client.is_connected

    def test_ensure_connected_raises_when_disconnected(self):
        """_ensure_connected should raise RuntimeError when not connected."""
        client = ElpisClient()
        with pytest.raises(RuntimeError, match="Not connected"):
            client._ensure_connected()

    def test_connected_client_passes_check(self, connected_client):
        """Connected client should pass connection check."""
        # Should not raise
        connected_client._ensure_connected()
        assert connected_client.is_connected


class TestElpisClientGenerate:
    """Tests for generate() method."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_calls_server_tool(self, connected_client, mock_session):
        """generate() should call the generate tool on server."""
        mock_session.set_tool_response(
            '{"content": "Hello!", "emotional_state": {"valence": 0.1}, "modulated_params": {}}'
        )

        result = await connected_client.generate(
            messages=[{"role": "user", "content": "Hi"}]
        )

        mock_session.call_tool.assert_called_once()
        call_args = mock_session.call_tool.call_args
        assert call_args[0][0] == "generate"
        assert "messages" in call_args[0][1]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_returns_result(self, connected_client, mock_session):
        """generate() should return GenerationResult with content."""
        mock_session.set_tool_response(
            '{"content": "Test response", "emotional_state": {"valence": 0.2, "arousal": 0.1, "quadrant": "calm"}, "modulated_params": {"temperature": 0.7}}'
        )

        result = await connected_client.generate(
            messages=[{"role": "user", "content": "Test"}]
        )

        assert isinstance(result, GenerationResult)
        assert result.content == "Test response"
        assert result.emotional_state.valence == 0.2
        assert result.modulated_params["temperature"] == 0.7

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_passes_parameters(self, connected_client, mock_session):
        """generate() should pass all parameters to server."""
        mock_session.set_tool_response(
            '{"content": "", "emotional_state": {}, "modulated_params": {}}'
        )

        await connected_client.generate(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1024,
            temperature=0.5,
            emotional_modulation=False,
        )

        call_args = mock_session.call_tool.call_args[0][1]
        assert call_args["max_tokens"] == 1024
        assert call_args["temperature"] == 0.5
        assert call_args["emotional_modulation"] is False


class TestElpisClientFunctionCall:
    """Tests for function_call() method."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_function_call_returns_tool_calls(self, connected_client, mock_session):
        """function_call() should return tool calls from server."""
        mock_session.set_tool_response(
            '{"tool_calls": [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}], "emotional_state": {}}'
        )

        result = await connected_client.function_call(
            messages=[{"role": "user", "content": "Use a tool"}],
            tools=[{"type": "function", "function": {"name": "test"}}]
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "test"


class TestElpisClientEmotion:
    """Tests for emotional state methods."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_update_emotion_sends_event(self, connected_client, mock_session):
        """update_emotion() should send event to server."""
        mock_session.set_tool_response(
            '{"valence": 0.3, "arousal": 0.2, "quadrant": "excited", "update_count": 1}'
        )

        state = await connected_client.update_emotion("success", intensity=1.5)

        call_args = mock_session.call_tool.call_args
        assert call_args[0][0] == "update_emotion"
        assert call_args[0][1]["event_type"] == "success"
        assert call_args[0][1]["intensity"] == 1.5
        assert state.valence == 0.3
        assert state.quadrant == "excited"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_reset_emotion_calls_server(self, connected_client, mock_session):
        """reset_emotion() should call server reset tool."""
        mock_session.set_tool_response(
            '{"valence": 0.0, "arousal": 0.0, "quadrant": "neutral", "update_count": 0}'
        )

        state = await connected_client.reset_emotion()

        mock_session.call_tool.assert_called_once_with("reset_emotion", {})
        assert state.valence == 0.0
        assert state.arousal == 0.0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_emotion_returns_state(self, connected_client, mock_session):
        """get_emotion() should return current emotional state."""
        mock_session.set_tool_response(
            '{"valence": -0.1, "arousal": 0.5, "quadrant": "frustrated", "update_count": 3}'
        )

        state = await connected_client.get_emotion()

        mock_session.call_tool.assert_called_once_with("get_emotion", {})
        assert state.valence == -0.1
        assert state.arousal == 0.5
        assert state.quadrant == "frustrated"


class TestElpisClientResources:
    """Tests for resource reading."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_read_resource_returns_content(self, connected_client, mock_session):
        """read_resource() should return resource content."""
        mock_session.set_resource_response('{"test": "data"}')

        content = await connected_client.read_resource("emotion://state")

        mock_session.read_resource.assert_called_once_with("emotion://state")
        assert content == '{"test": "data"}'

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_available_events(self, connected_client, mock_session):
        """list_available_events() should parse and return events."""
        mock_session.set_resource_response(
            '{"success": {"valence": 0.2}, "failure": {"valence": -0.2}}'
        )

        events = await connected_client.list_available_events()

        assert "success" in events
        assert "failure" in events
        assert events["success"]["valence"] == 0.2


class TestElpisClientIntegrationWithMocks:
    """Integration tests simulating full client-server interaction."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_conversation_flow(self, connected_client, mock_session):
        """Test a typical conversation flow."""
        # First message
        mock_session.set_tool_response(
            '{"content": "Hello! How can I help?", "emotional_state": {"valence": 0.1, "arousal": 0.0, "quadrant": "calm"}, "modulated_params": {"temperature": 0.7}}'
        )

        result1 = await connected_client.generate(
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert "Hello" in result1.content

        # Positive feedback updates emotion
        mock_session.set_tool_response(
            '{"valence": 0.3, "arousal": 0.1, "quadrant": "excited", "update_count": 1}'
        )

        state = await connected_client.update_emotion("success")
        assert state.valence > 0

        # Follow-up message with modified emotion
        mock_session.set_tool_response(
            '{"content": "Great to hear!", "emotional_state": {"valence": 0.3, "arousal": 0.1, "quadrant": "excited"}, "modulated_params": {"temperature": 0.68, "top_p": 0.93}}'
        )

        result2 = await connected_client.generate(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello! How can I help?"},
                {"role": "user", "content": "That was helpful, thanks!"}
            ]
        )

        # Emotional modulation should have affected parameters
        assert result2.emotional_state.quadrant == "excited"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tool_call_and_response(self, connected_client, mock_session):
        """Test tool call generation and handling."""
        # Request tool call
        mock_session.set_tool_response(
            '{"tool_calls": [{"id": "call_123", "function": {"name": "read_file", "arguments": "{\\"path\\": \\"/test.txt\\"}"}}], "emotional_state": {"valence": 0.0}}'
        )

        result = await connected_client.function_call(
            messages=[{"role": "user", "content": "Read the file test.txt"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}}
                    }
                }
            }]
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "read_file"
