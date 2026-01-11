"""
Unit tests for the AgentOrchestrator class.

Tests cover:
- Basic message processing
- ReAct loop iterations
- Tool execution handling
- History management
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from elpis.agent.orchestrator import AgentOrchestrator


@pytest.fixture
def mock_llm():
    """Create a mock LLM instance."""
    llm = MagicMock()
    llm.chat_completion = AsyncMock()
    llm.function_call = AsyncMock()
    return llm


@pytest.fixture
def mock_tools():
    """Create a mock ToolEngine instance."""
    tools = MagicMock()
    tools.tools = {
        "read_file": MagicMock(name="read_file", description="Read a file"),
        "write_file": MagicMock(name="write_file", description="Write a file"),
    }
    tools.get_tool_schemas = MagicMock(
        return_value=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {},
                },
            }
        ]
    )
    tools.execute_tool_call = AsyncMock()
    return tools


@pytest.fixture
def orchestrator(mock_llm, mock_tools):
    """Create an AgentOrchestrator instance with mocks."""
    return AgentOrchestrator(llm=mock_llm, tools=mock_tools)


class TestAgentOrchestrator:
    """Test suite for AgentOrchestrator."""

    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator, mock_llm, mock_tools):
        """Test orchestrator initializes correctly."""
        assert orchestrator.llm == mock_llm
        assert orchestrator.tools == mock_tools
        assert orchestrator.message_history == []

    @pytest.mark.asyncio
    async def test_process_simple_response(self, orchestrator, mock_llm):
        """Test processing a simple user input that doesn't require tools."""
        # Setup: LLM returns no tool calls, just a text response
        mock_llm.function_call.return_value = None
        mock_llm.chat_completion.return_value = "This is my response"

        # Execute
        response = await orchestrator.process("Hello, how are you?")

        # Assert
        assert response == "This is my response"
        assert len(orchestrator.message_history) == 2  # user + assistant
        assert orchestrator.message_history[0]["role"] == "user"
        assert orchestrator.message_history[0]["content"] == "Hello, how are you?"
        assert orchestrator.message_history[1]["role"] == "assistant"
        assert orchestrator.message_history[1]["content"] == "This is my response"

    @pytest.mark.asyncio
    async def test_process_with_tool_calls(self, orchestrator, mock_llm, mock_tools):
        """Test processing input that requires tool execution."""
        # Setup: First call returns tool calls, second returns final response
        tool_call = {
            "id": "call_123",
            "function": {"name": "read_file", "arguments": '{"file_path": "test.txt"}'},
        }
        mock_llm.function_call.side_effect = [[tool_call], None]
        mock_llm.chat_completion.return_value = "I read the file successfully"

        mock_tools.execute_tool_call.return_value = {
            "tool_call_id": "call_123",
            "success": True,
            "result": {"success": True, "content": "file content"},
            "duration_ms": 10.5,
        }

        # Execute
        response = await orchestrator.process("Read test.txt")

        # Assert
        assert response == "I read the file successfully"
        assert len(orchestrator.message_history) == 4  # user, assistant+tools, tool, assistant
        assert mock_tools.execute_tool_call.called
        assert mock_llm.function_call.call_count == 2
        assert mock_llm.chat_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_react_loop_iterations(self, orchestrator, mock_llm, mock_tools):
        """Test that ReAct loop can iterate multiple times."""
        # Setup: Multiple tool calls before final response
        tool_call_1 = {
            "id": "call_1",
            "function": {"name": "read_file", "arguments": "{}"},
        }
        tool_call_2 = {
            "id": "call_2",
            "function": {"name": "write_file", "arguments": "{}"},
        }

        mock_llm.function_call.side_effect = [[tool_call_1], [tool_call_2], None]
        mock_llm.chat_completion.return_value = "Done!"

        mock_tools.execute_tool_call.return_value = {
            "tool_call_id": "call_x",
            "success": True,
            "result": {"success": True},
            "duration_ms": 5.0,
        }

        # Execute
        response = await orchestrator.process("Do multiple things")

        # Assert
        assert response == "Done!"
        assert mock_llm.function_call.call_count == 3
        assert mock_tools.execute_tool_call.call_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self, orchestrator, mock_llm, mock_tools):
        """Test that ReAct loop stops at max iterations."""
        # Setup: Always return tool calls (infinite loop scenario)
        mock_llm.function_call.return_value = [
            {"id": "call_x", "function": {"name": "test", "arguments": "{}"}}
        ]

        # Mock tool execution to return success
        mock_tools.execute_tool_call.return_value = {
            "tool_call_id": "call_x",
            "success": True,
            "result": {"success": True},
            "duration_ms": 5.0,
        }

        # Execute
        response = await orchestrator.process("Test infinite loop")

        # Assert: Should return fallback message
        assert "reasoning limit" in response.lower()
        assert mock_llm.function_call.call_count == 10  # max_iterations

    @pytest.mark.asyncio
    async def test_clear_history(self, orchestrator, mock_llm):
        """Test clearing conversation history."""
        # Setup: Add some history
        mock_llm.function_call.return_value = None
        mock_llm.chat_completion.return_value = "Response"
        await orchestrator.process("Test message")

        assert len(orchestrator.message_history) > 0

        # Execute
        orchestrator.clear_history()

        # Assert
        assert len(orchestrator.message_history) == 0

    @pytest.mark.asyncio
    async def test_get_history_length(self, orchestrator):
        """Test getting history length."""
        assert orchestrator.get_history_length() == 0

        orchestrator.message_history.append({"role": "user", "content": "test"})
        assert orchestrator.get_history_length() == 1

    @pytest.mark.asyncio
    async def test_get_last_message(self, orchestrator):
        """Test getting last message."""
        assert orchestrator.get_last_message() is None

        msg = {"role": "user", "content": "test"}
        orchestrator.message_history.append(msg)
        assert orchestrator.get_last_message() == msg

    @pytest.mark.asyncio
    async def test_tool_execution_error_handling(self, orchestrator, mock_llm, mock_tools):
        """Test handling of tool execution errors."""
        # Setup: Tool execution fails
        tool_call = {
            "id": "call_123",
            "function": {"name": "read_file", "arguments": "{}"},
        }
        mock_llm.function_call.side_effect = [[tool_call], None]
        mock_llm.chat_completion.return_value = "Handled the error"

        mock_tools.execute_tool_call.return_value = {
            "tool_call_id": "call_123",
            "success": False,
            "result": {"error": "File not found"},
            "duration_ms": 2.0,
        }

        # Execute
        response = await orchestrator.process("Read missing file")

        # Assert: Should continue and generate response
        assert response == "Handled the error"
        assert "error" in orchestrator.message_history[2]["content"].lower()

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, orchestrator, mock_llm, mock_tools):
        """Test that multiple tools are executed concurrently."""
        # Setup: Multiple tool calls at once
        tool_calls = [
            {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "write_file", "arguments": "{}"}},
            {"id": "call_3", "function": {"name": "read_file", "arguments": "{}"}},
        ]

        mock_llm.function_call.side_effect = [tool_calls, None]
        mock_llm.chat_completion.return_value = "All done"

        mock_tools.execute_tool_call.return_value = {
            "tool_call_id": "call_x",
            "success": True,
            "result": {"success": True},
            "duration_ms": 5.0,
        }

        # Execute
        response = await orchestrator.process("Do three things")

        # Assert: All tools should be called
        assert response == "All done"
        assert mock_tools.execute_tool_call.call_count == 3

    @pytest.mark.asyncio
    async def test_build_messages_includes_system_prompt(self, orchestrator):
        """Test that built messages include system prompt."""
        orchestrator.message_history.append({"role": "user", "content": "test"})

        messages = orchestrator._build_messages()

        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert "Elpis" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_format_tool_results_success(self, orchestrator):
        """Test formatting successful tool results."""
        results = [
            {
                "tool_call_id": "call_1",
                "success": True,
                "result": {"data": "test"},
                "duration_ms": 10.5,
            }
        ]

        formatted = orchestrator._format_tool_results(results)

        assert "call_1" in formatted
        assert "succeeded" in formatted
        assert "10.50ms" in formatted
        assert "test" in formatted

    @pytest.mark.asyncio
    async def test_format_tool_results_failure(self, orchestrator):
        """Test formatting failed tool results."""
        results = [
            {
                "tool_call_id": "call_1",
                "success": False,
                "result": {"error": "Something went wrong"},
                "duration_ms": 5.0,
            }
        ]

        formatted = orchestrator._format_tool_results(results)

        assert "call_1" in formatted
        assert "failed" in formatted
        assert "Something went wrong" in formatted

    @pytest.mark.asyncio
    async def test_exception_handling_in_process(self, orchestrator, mock_llm):
        """Test that exceptions in process are handled gracefully."""
        # Setup: LLM raises an exception
        mock_llm.function_call.side_effect = Exception("LLM error")

        # Execute
        response = await orchestrator.process("Test error")

        # Assert: Should return error message
        assert "error" in response.lower()
        assert "LLM error" in response

    @pytest.mark.asyncio
    async def test_tool_execution_with_exception(self, orchestrator, mock_llm, mock_tools):
        """Test handling when tool execution raises an exception."""
        # Setup: Tool raises exception
        tool_call = {
            "id": "call_123",
            "function": {"name": "read_file", "arguments": "{}"},
        }
        mock_llm.function_call.side_effect = [[tool_call], None]
        mock_llm.chat_completion.return_value = "Recovered from error"

        # Make execute_tool_call raise an exception
        mock_tools.execute_tool_call.side_effect = Exception("Tool crashed")

        # Execute
        response = await orchestrator.process("Test")

        # Assert: Should handle exception and convert to error result
        assert response == "Recovered from error"
        # The exception should be in the tool results (check messages with content)
        messages_with_content = [
            msg for msg in orchestrator.message_history if msg.get("content") is not None
        ]
        assert any("error" in msg["content"].lower() for msg in messages_with_content)
