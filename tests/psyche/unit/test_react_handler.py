"""Unit tests for ReactHandler."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from psyche.client.react_handler import ReactHandler, ReactConfig, ToolCallResult
from psyche.memory.compaction import ContextCompactor, create_message


@pytest.fixture
def mock_client():
    """Create a mock ElpisClient."""
    client = MagicMock()
    client.is_connected = True
    client.generate_stream = AsyncMock()
    client.update_emotion = AsyncMock()
    return client


@pytest.fixture
def mock_tool_engine():
    """Create a mock ToolEngine."""
    engine = MagicMock()
    engine.execute_tool_call = AsyncMock(return_value={"success": True, "result": "test result"})
    return engine


@pytest.fixture
def compactor():
    """Create a real ContextCompactor for testing."""
    return ContextCompactor(max_tokens=10000, reserve_tokens=2000)


@pytest.fixture
def handler(mock_client, mock_tool_engine, compactor):
    """Create a ReactHandler instance for testing."""
    return ReactHandler(
        elpis_client=mock_client,
        tool_engine=mock_tool_engine,
        compactor=compactor,
        config=ReactConfig(max_tool_iterations=5),
    )


class TestParseToolCall:
    """Tests for parse_tool_call method."""

    def test_parse_tool_call_with_code_block(self, handler):
        """Parse tool call from code block format."""
        text = '''I'll list the files.
```tool_call
{"name": "list_directory", "arguments": {"path": ".", "recursive": false}}
```'''
        result = handler.parse_tool_call(text)
        assert result is not None
        assert result["name"] == "list_directory"
        assert result["arguments"]["path"] == "."
        assert result["arguments"]["recursive"] is False

    def test_parse_tool_call_minimal(self, handler):
        """Parse tool call with no arguments."""
        text = '''```tool_call
{"name": "list_directory"}
```'''
        result = handler.parse_tool_call(text)
        assert result is not None
        assert result["name"] == "list_directory"
        assert result["arguments"] == {}

    def test_parse_tool_call_json_only(self, handler):
        """Parse tool call from raw JSON at start."""
        text = '{"name": "execute_bash", "arguments": {"command": "ls"}}'
        result = handler.parse_tool_call(text)
        assert result is not None
        assert result["name"] == "execute_bash"
        assert result["arguments"]["command"] == "ls"

    def test_parse_tool_call_no_match(self, handler):
        """Return None when no tool call found."""
        text = "Just a regular response with no tool calls."
        result = handler.parse_tool_call(text)
        assert result is None

    def test_parse_tool_call_invalid_json(self, handler):
        """Handle invalid JSON gracefully."""
        text = '''```tool_call
{"name": "test", "arguments": {invalid json}
```'''
        result = handler.parse_tool_call(text)
        assert result is None

    def test_parse_tool_call_case_insensitive(self, handler):
        """Parse tool_call with different case."""
        text = '''```TOOL_CALL
{"name": "read_file", "arguments": {"file_path": "test.txt"}}
```'''
        result = handler.parse_tool_call(text)
        assert result is not None
        assert result["name"] == "read_file"

    def test_parse_tool_call_with_newlines(self, handler):
        """Parse tool call with extra newlines."""
        text = '''```tool_call

{"name": "read_file", "arguments": {"file_path": "out.txt"}}

```'''
        result = handler.parse_tool_call(text)
        assert result is not None
        assert result["name"] == "read_file"

    def test_parse_tool_call_nested_json(self, handler):
        """Parse tool call with nested braces in arguments."""
        text = '{"name": "execute_bash", "arguments": {"command": "echo \'{\\"key\\": \\"value\\"}\'"}}'
        result = handler.parse_tool_call(text)
        assert result is not None
        assert result["name"] == "execute_bash"


class TestExecuteTool:
    """Tests for execute_tool method."""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, handler, mock_tool_engine):
        """Execute tool successfully."""
        tool_call = {"name": "read_file", "arguments": {"file_path": "test.txt"}}

        result = await handler.execute_tool(tool_call)

        assert result.tool_name == "read_file"
        assert result.success is True
        assert result.error is None
        mock_tool_engine.execute_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_with_callback(self, handler, mock_tool_engine):
        """Execute tool with callback notification."""
        tool_call = {"name": "read_file", "arguments": {"file_path": "test.txt"}}
        callback_calls = []

        def callback(name, args, result):
            callback_calls.append((name, args, result))

        await handler.execute_tool(tool_call, on_tool_call=callback)

        # Should be called twice: once at start (None result), once at end
        assert len(callback_calls) == 2
        assert callback_calls[0][2] is None  # Start notification
        assert callback_calls[1][2] is not None  # End with result

    @pytest.mark.asyncio
    async def test_execute_tool_failure(self, handler, mock_tool_engine):
        """Handle tool execution failure."""
        mock_tool_engine.execute_tool_call.side_effect = Exception("Tool error")
        tool_call = {"name": "bad_tool", "arguments": {}}

        result = await handler.execute_tool(tool_call)

        assert result.success is False
        assert result.error == "Tool error"

    @pytest.mark.asyncio
    async def test_execute_tool_updates_emotion_on_success(self, handler, mock_client, mock_tool_engine):
        """Verify emotion is updated on successful tool execution."""
        mock_tool_engine.execute_tool_call.return_value = {"success": True}
        tool_call = {"name": "test_tool", "arguments": {}}

        await handler.execute_tool(tool_call)

        mock_client.update_emotion.assert_called()
        # Should be called with "success"
        call_args = mock_client.update_emotion.call_args
        assert call_args[0][0] == "success"


class TestProcessInput:
    """Tests for process_input method."""

    @pytest.mark.asyncio
    async def test_process_input_simple_response(self, handler, mock_client):
        """Process input with simple response (no tools)."""
        # Simulate streaming tokens
        async def mock_stream(*args, **kwargs):
            for token in ["Hello", ", ", "world", "!"]:
                yield token

        mock_client.generate_stream = mock_stream
        tokens_received = []

        result = await handler.process_input(
            "Hi",
            on_token=lambda t: tokens_received.append(t),
        )

        assert result == "Hello, world!"
        assert tokens_received == ["Hello", ", ", "world", "!"]

    @pytest.mark.asyncio
    async def test_process_input_disconnected(self, handler, mock_client):
        """Handle disconnected client."""
        mock_client.is_connected = False
        response_received = []

        result = await handler.process_input(
            "Hi",
            on_response=lambda r: response_received.append(r),
        )

        assert "disconnected" in result.lower()
        assert any("disconnected" in r.lower() for r in response_received)

    @pytest.mark.asyncio
    async def test_process_input_with_tool_call(self, handler, mock_client, mock_tool_engine):
        """Process input that triggers a tool call."""
        call_count = [0]

        async def mock_stream(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: return tool call
                for token in ['```tool_call\n{"name": "read_file", "arguments": {"file_path": "test.txt"}}\n```']:
                    yield token
            else:
                # Second call: return final response
                for token in ["File contents here"]:
                    yield token

        mock_client.generate_stream = mock_stream
        mock_tool_engine.execute_tool_call.return_value = {"success": True, "content": "test"}

        result = await handler.process_input("Read the file")

        assert result == "File contents here"
        mock_tool_engine.execute_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_input_max_iterations(self, handler, mock_client, mock_tool_engine):
        """Respect max tool iterations limit."""
        # Always return tool call
        async def mock_stream(*args, **kwargs):
            for token in ['```tool_call\n{"name": "read_file", "arguments": {}}\n```']:
                yield token

        mock_client.generate_stream = mock_stream
        handler.config.max_tool_iterations = 3

        result = await handler.process_input("Keep calling tools")

        assert "Max tool iterations reached" in result
        # Should have called tool engine 3 times (max_tool_iterations)
        assert mock_tool_engine.execute_tool_call.call_count == 3


class TestInterrupt:
    """Tests for interrupt handling."""

    def test_interrupt_when_processing(self, handler):
        """Interrupt returns True when processing."""
        handler._is_processing = True
        assert handler.interrupt() is True
        assert handler._interrupt_event.is_set()

    def test_interrupt_when_not_processing(self, handler):
        """Interrupt returns False when not processing."""
        handler._is_processing = False
        assert handler.interrupt() is False
        assert not handler._interrupt_event.is_set()

    def test_clear_interrupt(self, handler):
        """Clear interrupt flag."""
        handler._interrupt_event.set()
        handler.clear_interrupt()
        assert not handler._interrupt_event.is_set()

    @pytest.mark.asyncio
    async def test_interrupt_during_generation(self, handler, mock_client):
        """Interrupt during token generation."""
        tokens_yielded = []

        async def mock_stream(*args, **kwargs):
            for i, token in enumerate(["Hello", " ", "world"]):
                tokens_yielded.append(token)
                if i == 1:  # Interrupt after second token
                    handler._interrupt_event.set()
                yield token

        mock_client.generate_stream = mock_stream

        result = await handler.process_input("Hi")

        assert "[Interrupted]" in result


class TestIsProcessing:
    """Tests for is_processing property."""

    def test_is_processing_false_initially(self, handler):
        """is_processing is False initially."""
        assert handler.is_processing is False

    @pytest.mark.asyncio
    async def test_is_processing_during_operation(self, handler, mock_client):
        """is_processing is True during operation."""
        processing_states = []

        async def mock_stream(*args, **kwargs):
            processing_states.append(handler.is_processing)
            for token in ["test"]:
                yield token

        mock_client.generate_stream = mock_stream

        await handler.process_input("test")

        assert True in processing_states  # Was True during generation
        assert handler.is_processing is False  # False after completion


class TestToolCallResult:
    """Tests for ToolCallResult dataclass."""

    def test_tool_call_result_success(self):
        """Create successful tool call result."""
        result = ToolCallResult(
            tool_name="test_tool",
            arguments={"arg1": "value1"},
            result={"success": True},
            success=True,
        )

        assert result.tool_name == "test_tool"
        assert result.success is True
        assert result.error is None

    def test_tool_call_result_failure(self):
        """Create failed tool call result."""
        result = ToolCallResult(
            tool_name="test_tool",
            arguments={},
            result={"success": False},
            success=False,
            error="Some error",
        )

        assert result.success is False
        assert result.error == "Some error"


class TestReactConfig:
    """Tests for ReactConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReactConfig()

        assert config.max_tool_iterations == 10
        assert config.max_tool_result_chars == 16000
        assert config.generation_timeout == 120.0
        assert config.emotional_modulation is True
        assert config.reasoning_enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReactConfig(
            max_tool_iterations=5,
            max_tool_result_chars=8000,
            generation_timeout=60.0,
            emotional_modulation=False,
        )

        assert config.max_tool_iterations == 5
        assert config.max_tool_result_chars == 8000
        assert config.generation_timeout == 60.0
        assert config.emotional_modulation is False
