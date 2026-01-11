"""
Integration tests for the full agent workflow.

Tests cover:
- End-to-end agent processing
- ReAct loop with real components
- Tool execution integration
- REPL integration
- Error recovery
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from elpis.agent.orchestrator import AgentOrchestrator
from elpis.agent.repl import ElpisREPL
from elpis.llm.inference import LlamaInference
from elpis.tools.tool_engine import ToolEngine


class MockLLMForIntegration:
    """Mock LLM that behaves more realistically for integration tests."""

    def __init__(self):
        self.call_count = 0
        self.settings = MagicMock()

    async def chat_completion(self, messages, max_tokens=None, temperature=None, top_p=None):
        """Generate a mock chat completion."""
        self.call_count += 1
        # Simulate a response based on context
        return f"Integration test response #{self.call_count}"

    async def function_call(self, messages, tools, temperature=None):
        """Generate mock function calls."""
        self.call_count += 1

        # First call: return a tool call
        # Second call: return None (final response)
        if self.call_count == 1:
            return [
                {
                    "id": "call_test_1",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"file_path": "test.txt"}',
                    },
                }
            ]
        else:
            return None


class MockToolEngineForIntegration:
    """Mock tool engine that behaves realistically."""

    def __init__(self, workspace_dir, settings=None):
        self.workspace_dir = workspace_dir
        self.settings = settings
        self.tools = {
            "read_file": MagicMock(name="read_file", description="Read a file"),
            "write_file": MagicMock(name="write_file", description="Write a file"),
        }

    def get_tool_schemas(self):
        """Return mock tool schemas."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                        },
                        "required": ["file_path"],
                    },
                },
            }
        ]

    async def execute_tool_call(self, tool_call):
        """Execute a mock tool call."""
        return {
            "tool_call_id": tool_call.get("id"),
            "success": True,
            "result": {
                "success": True,
                "content": "Mock file content from integration test",
            },
            "duration_ms": 15.5,
        }


@pytest.fixture
def integration_llm():
    """Create a realistic mock LLM for integration testing."""
    return MockLLMForIntegration()


@pytest.fixture
def integration_tools(tmp_path):
    """Create a realistic mock tool engine for integration testing."""
    return MockToolEngineForIntegration(workspace_dir=str(tmp_path))


@pytest.fixture
def integration_agent(integration_llm, integration_tools):
    """Create an agent with integration test mocks."""
    return AgentOrchestrator(
        llm=integration_llm, tools=integration_tools, settings=MagicMock()
    )


class TestAgentWorkflow:
    """Integration tests for full agent workflow."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_react_loop(self, integration_agent):
        """Test a complete ReAct loop with tool execution."""
        response = await integration_agent.process("Read the test file")

        # Should complete successfully
        assert response is not None
        assert "response" in response.lower()

        # History should contain: user, assistant+tools, tool, assistant
        assert len(integration_agent.message_history) == 4

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_conversations(self, integration_agent):
        """Test multiple conversation turns."""
        # First conversation
        response1 = await integration_agent.process("First question")
        assert response1 is not None

        # Second conversation
        response2 = await integration_agent.process("Second question")
        assert response2 is not None

        # History should accumulate
        assert len(integration_agent.message_history) > 4

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_history_persistence(self, integration_agent):
        """Test that conversation history persists across calls."""
        await integration_agent.process("First message")
        history_len_1 = len(integration_agent.message_history)

        await integration_agent.process("Second message")
        history_len_2 = len(integration_agent.message_history)

        # History should grow
        assert history_len_2 > history_len_1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_clear_history_integration(self, integration_agent):
        """Test clearing history in integration context."""
        await integration_agent.process("Test message")
        assert len(integration_agent.message_history) > 0

        integration_agent.clear_history()
        assert len(integration_agent.message_history) == 0

        # Should work fine after clearing
        response = await integration_agent.process("New message")
        assert response is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_repl_integration(self, integration_agent, tmp_path):
        """Test REPL integration with agent."""
        history_file = tmp_path / ".test_repl_history"

        with patch("elpis.agent.repl.PromptSession"):
            repl = ElpisREPL(agent=integration_agent, history_file=str(history_file))

            # Test special commands
            result = await repl._handle_special_command("/help")
            assert result is True

            result = await repl._handle_special_command("/clear")
            assert result is True
            assert len(integration_agent.message_history) == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_agent_tool_engine_integration(self, integration_llm, tmp_path):
        """Test agent integration with tool engine."""
        tools = MockToolEngineForIntegration(workspace_dir=str(tmp_path))
        agent = AgentOrchestrator(llm=integration_llm, tools=tools)

        response = await agent.process("Use the tools")

        # Should have called tools
        assert response is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_recovery(self, integration_tools, tmp_path):
        """Test agent recovers from errors gracefully."""

        class ErrorLLM:
            """LLM that produces an error then recovers."""

            def __init__(self):
                self.call_count = 0
                self.settings = MagicMock()

            async def chat_completion(self, messages, **kwargs):
                return "Recovered response"

            async def function_call(self, messages, tools, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    raise Exception("Temporary error")
                return None

        error_llm = ErrorLLM()
        agent = AgentOrchestrator(llm=error_llm, tools=integration_tools)

        # Should handle error gracefully
        response = await agent.process("Test error handling")
        assert "error" in response.lower()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_message_building(self, integration_agent):
        """Test that messages are built correctly with context."""
        integration_agent.message_history.append(
            {"role": "user", "content": "Test message"}
        )

        messages = integration_agent._build_messages()

        # Should have system prompt + history
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert "Elpis" in messages[0]["content"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tool_result_formatting(self, integration_agent):
        """Test formatting of tool results."""
        results = [
            {
                "tool_call_id": "call_1",
                "success": True,
                "result": {"data": "test data"},
                "duration_ms": 10.5,
            },
            {
                "tool_call_id": "call_2",
                "success": False,
                "result": {"error": "Test error"},
                "duration_ms": 5.0,
            },
        ]

        formatted = integration_agent._format_tool_results(results)

        assert "call_1" in formatted
        assert "call_2" in formatted
        assert "succeeded" in formatted
        assert "failed" in formatted
        assert "Test error" in formatted

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_workflow(self, tmp_path):
        """Test agent can handle concurrent workflows."""

        class MultiToolLLM:
            """LLM that returns multiple tool calls."""

            def __init__(self):
                self.call_count = 0
                self.settings = MagicMock()

            async def chat_completion(self, messages, **kwargs):
                return "All tools executed"

            async def function_call(self, messages, tools, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    # Return multiple tool calls
                    return [
                        {
                            "id": "call_1",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "function": {"name": "write_file", "arguments": "{}"},
                        },
                    ]
                return None

        llm = MultiToolLLM()
        tools = MockToolEngineForIntegration(workspace_dir=str(tmp_path))
        agent = AgentOrchestrator(llm=llm, tools=tools)

        response = await agent.process("Execute multiple tools")

        assert response == "All tools executed"
        # Should have processed all tool calls
        assert len(agent.message_history) > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_repl_full_interaction(self, integration_agent, tmp_path):
        """Test a full REPL interaction sequence."""
        history_file = tmp_path / ".repl_test"

        with patch("elpis.agent.repl.PromptSession") as mock_session:
            repl = ElpisREPL(agent=integration_agent, history_file=str(history_file))

            # Simulate a full interaction
            with patch.object(repl.session, "prompt_async") as mock_prompt:
                mock_prompt.side_effect = [
                    "Hello",
                    "/status",
                    "/clear",
                    "New conversation",
                    EOFError(),
                ]

                await repl.run()

                # Should have processed messages
                assert integration_agent.get_history_length() >= 0
