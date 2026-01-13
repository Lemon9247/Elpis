"""Unit tests for MemoryServer tool call parsing."""

import pytest
from unittest.mock import MagicMock

from psyche.mcp.client import ElpisClient
from psyche.memory.server import MemoryServer, ServerConfig


@pytest.fixture
def mock_client():
    """Create a mock ElpisClient."""
    return MagicMock(spec=ElpisClient)


@pytest.fixture
def server(mock_client, tmp_path):
    """Create a MemoryServer instance for testing."""
    config = ServerConfig(workspace_dir=str(tmp_path))
    return MemoryServer(elpis_client=mock_client, config=config)


class TestParseToolCall:
    """Tests for _parse_tool_call method."""

    def test_parse_tool_call_with_code_block(self, server):
        """Parse tool call from code block format."""
        text = '''I'll list the files.
```tool_call
{"name": "list_directory", "arguments": {"path": ".", "recursive": false}}
```'''
        result = server._parse_tool_call(text)
        assert result is not None
        assert result["name"] == "list_directory"
        assert result["arguments"]["path"] == "."
        assert result["arguments"]["recursive"] is False

    def test_parse_tool_call_minimal(self, server):
        """Parse tool call with no arguments."""
        text = '''```tool_call
{"name": "list_directory"}
```'''
        result = server._parse_tool_call(text)
        assert result is not None
        assert result["name"] == "list_directory"
        assert result["arguments"] == {}

    def test_parse_tool_call_json_only(self, server):
        """Parse tool call from raw JSON at start."""
        text = '{"name": "execute_bash", "arguments": {"command": "ls"}}'
        result = server._parse_tool_call(text)
        assert result is not None
        assert result["name"] == "execute_bash"
        assert result["arguments"]["command"] == "ls"

    def test_parse_tool_call_no_match(self, server):
        """Return None when no tool call found."""
        text = "Just a regular response with no tool calls."
        result = server._parse_tool_call(text)
        assert result is None

    def test_parse_tool_call_invalid_json(self, server):
        """Handle invalid JSON gracefully."""
        text = '''```tool_call
{"name": "test", "arguments": {invalid json}
```'''
        result = server._parse_tool_call(text)
        assert result is None

    def test_parse_tool_call_case_insensitive(self, server):
        """Parse tool_call with different case."""
        text = '''```TOOL_CALL
{"name": "read_file", "arguments": {"file_path": "test.txt"}}
```'''
        result = server._parse_tool_call(text)
        assert result is not None
        assert result["name"] == "read_file"

    def test_parse_tool_call_with_newlines(self, server):
        """Parse tool call with extra newlines."""
        text = '''```tool_call

{"name": "write_file", "arguments": {"file_path": "out.txt", "content": "hello"}}

```'''
        result = server._parse_tool_call(text)
        assert result is not None
        assert result["name"] == "write_file"

    def test_parse_tool_call_nested_json(self, server):
        """Parse tool call with nested braces in arguments."""
        text = '{"name": "execute_bash", "arguments": {"command": "echo \'{\\"key\\": \\"value\\"}\'"}}'
        result = server._parse_tool_call(text)
        assert result is not None
        assert result["name"] == "execute_bash"
