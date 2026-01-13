"""Unit tests for MCP client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from psyche.mcp.client import ElpisClient, EmotionalState, GenerationResult


class TestEmotionalState:
    """Tests for EmotionalState class."""

    def test_default_values(self):
        """EmotionalState should have sensible defaults."""
        state = EmotionalState()
        assert state.valence == 0.0
        assert state.arousal == 0.0
        assert state.quadrant == "neutral"
        assert state.update_count == 0

    def test_from_dict(self):
        """from_dict should create state from server response."""
        data = {
            "valence": 0.5,
            "arousal": -0.3,
            "quadrant": "calm",
            "update_count": 5,
        }
        state = EmotionalState.from_dict(data)

        assert state.valence == 0.5
        assert state.arousal == -0.3
        assert state.quadrant == "calm"
        assert state.update_count == 5

    def test_from_dict_missing_fields(self):
        """from_dict should handle missing fields with defaults."""
        state = EmotionalState.from_dict({})

        assert state.valence == 0.0
        assert state.arousal == 0.0
        assert state.quadrant == "neutral"
        assert state.update_count == 0


class TestGenerationResult:
    """Tests for GenerationResult class."""

    def test_creation(self):
        """GenerationResult should store all fields."""
        state = EmotionalState(valence=0.5, arousal=0.2, quadrant="excited")
        result = GenerationResult(
            content="Test response",
            emotional_state=state,
            modulated_params={"temperature": 0.6, "top_p": 0.95},
        )

        assert result.content == "Test response"
        assert result.emotional_state.quadrant == "excited"
        assert result.modulated_params["temperature"] == 0.6


class TestElpisClient:
    """Tests for ElpisClient class."""

    def test_init_defaults(self):
        """Client should initialize with defaults."""
        client = ElpisClient()
        assert client.server_command == "elpis-server"
        assert client.server_args == []
        assert not client.is_connected

    def test_init_custom(self):
        """Client should accept custom server settings."""
        client = ElpisClient(
            server_command="/custom/elpis",
            server_args=["--debug"],
        )
        assert client.server_command == "/custom/elpis"
        assert client.server_args == ["--debug"]

    def test_ensure_connected_raises(self):
        """_ensure_connected should raise when not connected."""
        client = ElpisClient()
        with pytest.raises(RuntimeError, match="Not connected"):
            client._ensure_connected()

    @pytest.mark.asyncio
    async def test_generate_basic(self):
        """generate should call server and return result."""
        client = ElpisClient()
        client._connected = True
        client._session = MagicMock()

        mock_result = MagicMock()
        mock_result.content = [
            MagicMock(
                text='{"content": "Hello", "emotional_state": {"valence": 0.1}, "modulated_params": {}}'
            )
        ]
        client._session.call_tool = AsyncMock(return_value=mock_result)

        result = await client.generate(
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert result.content == "Hello"
        assert result.emotional_state.valence == 0.1
        client._session.call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_emotion(self):
        """update_emotion should call server with event."""
        client = ElpisClient()
        client._connected = True
        client._session = MagicMock()

        mock_result = MagicMock()
        mock_result.content = [
            MagicMock(text='{"valence": 0.2, "arousal": 0.1, "quadrant": "excited"}')
        ]
        client._session.call_tool = AsyncMock(return_value=mock_result)

        state = await client.update_emotion("success", intensity=1.5)

        assert state.valence == 0.2
        assert state.quadrant == "excited"

        # Verify call arguments
        call_args = client._session.call_tool.call_args
        assert call_args[0][0] == "update_emotion"
        assert call_args[0][1]["event_type"] == "success"
        assert call_args[0][1]["intensity"] == 1.5

    @pytest.mark.asyncio
    async def test_reset_emotion(self):
        """reset_emotion should call server reset."""
        client = ElpisClient()
        client._connected = True
        client._session = MagicMock()

        mock_result = MagicMock()
        mock_result.content = [
            MagicMock(text='{"valence": 0.0, "arousal": 0.0, "quadrant": "excited"}')
        ]
        client._session.call_tool = AsyncMock(return_value=mock_result)

        state = await client.reset_emotion()

        assert state.valence == 0.0
        assert state.arousal == 0.0
        client._session.call_tool.assert_called_once_with("reset_emotion", {})

    @pytest.mark.asyncio
    async def test_get_emotion(self):
        """get_emotion should return current state."""
        client = ElpisClient()
        client._connected = True
        client._session = MagicMock()

        mock_result = MagicMock()
        mock_result.content = [
            MagicMock(
                text='{"valence": -0.2, "arousal": 0.5, "quadrant": "frustrated"}'
            )
        ]
        client._session.call_tool = AsyncMock(return_value=mock_result)

        state = await client.get_emotion()

        assert state.valence == -0.2
        assert state.arousal == 0.5
        assert state.quadrant == "frustrated"

    @pytest.mark.asyncio
    async def test_read_resource(self):
        """read_resource should return resource content."""
        client = ElpisClient()
        client._connected = True
        client._session = MagicMock()

        mock_result = MagicMock()
        mock_result.contents = [MagicMock(text='{"test": "data"}')]
        client._session.read_resource = AsyncMock(return_value=mock_result)

        content = await client.read_resource("emotion://state")

        assert content == '{"test": "data"}'
        client._session.read_resource.assert_called_once_with("emotion://state")
