"""
Unit tests for the ElpisREPL class.

Tests cover:
- REPL initialization
- Special command handling
- Response formatting
- Error handling
- Display methods
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from pathlib import Path

from elpis.agent.repl import ElpisREPL


@pytest.fixture
def mock_agent():
    """Create a mock AgentOrchestrator instance."""
    agent = MagicMock()
    agent.process = AsyncMock()
    agent.clear_history = MagicMock()
    agent.get_history_length = MagicMock(return_value=0)
    agent.get_last_message = MagicMock(return_value=None)
    return agent


@pytest.fixture
def repl(mock_agent, tmp_path):
    """Create an ElpisREPL instance with mocks."""
    history_file = tmp_path / ".test_history"
    with patch("elpis.agent.repl.PromptSession"):
        repl = ElpisREPL(agent=mock_agent, history_file=str(history_file))
        return repl


class TestElpisREPL:
    """Test suite for ElpisREPL."""

    def test_initialization(self, mock_agent, tmp_path):
        """Test REPL initializes correctly."""
        history_file = tmp_path / ".test_history"

        with patch("elpis.agent.repl.PromptSession") as mock_session:
            repl = ElpisREPL(agent=mock_agent, history_file=str(history_file))

            assert repl.agent == mock_agent
            assert repl.console is not None
            assert mock_session.called

    @pytest.mark.asyncio
    async def test_handle_help_command(self, repl):
        """Test /help command."""
        result = await repl._handle_special_command("/help")

        assert result is True  # Should continue REPL

    @pytest.mark.asyncio
    async def test_handle_clear_command(self, repl, mock_agent):
        """Test /clear command."""
        result = await repl._handle_special_command("/clear")

        assert result is True  # Should continue REPL
        assert mock_agent.clear_history.called

    @pytest.mark.asyncio
    async def test_handle_exit_command(self, repl):
        """Test /exit command."""
        result = await repl._handle_special_command("/exit")

        assert result is False  # Should exit REPL

    @pytest.mark.asyncio
    async def test_handle_quit_command(self, repl):
        """Test /quit command (alias for exit)."""
        result = await repl._handle_special_command("/quit")

        assert result is False  # Should exit REPL

    @pytest.mark.asyncio
    async def test_handle_status_command(self, repl, mock_agent):
        """Test /status command (hidden debug command)."""
        mock_agent.get_history_length.return_value = 5
        mock_agent.get_last_message.return_value = {"role": "user", "content": "test"}

        result = await repl._handle_special_command("/status")

        assert result is True  # Should continue REPL
        assert mock_agent.get_history_length.called

    @pytest.mark.asyncio
    async def test_handle_unknown_command(self, repl):
        """Test handling of unknown special command."""
        result = await repl._handle_special_command("/unknown")

        assert result is True  # Should continue REPL

    @pytest.mark.asyncio
    async def test_handle_command_case_insensitive(self, repl):
        """Test that commands are case insensitive."""
        result_lower = await repl._handle_special_command("/exit")
        assert result_lower is False

        result_upper = await repl._handle_special_command("/EXIT")
        assert result_upper is False

        result_mixed = await repl._handle_special_command("/ExIt")
        assert result_mixed is False

    def test_display_response_with_markdown(self, repl):
        """Test displaying response with markdown content."""
        response = "# Heading\n\nThis is **bold** text with `code`"

        # Should not raise exception
        repl._display_response(response)

    def test_display_response_plain_text(self, repl):
        """Test displaying plain text response."""
        response = "This is a simple response without any markdown"

        # Should not raise exception
        repl._display_response(response)

    def test_display_response_with_code_blocks(self, repl):
        """Test displaying response with code blocks."""
        response = """
Here's some code:

```python
def hello():
    print("Hello, world!")
```
"""
        # Should not raise exception
        repl._display_response(response)

    def test_display_error(self, repl):
        """Test displaying error message."""
        # Should not raise exception
        repl.display_error("This is an error message")

    def test_display_info(self, repl):
        """Test displaying info message."""
        # Should not raise exception
        repl.display_info("This is an info message")

    def test_display_success(self, repl):
        """Test displaying success message."""
        # Should not raise exception
        repl.display_success("This is a success message")

    def test_display_welcome(self, repl):
        """Test displaying welcome banner."""
        # Should not raise exception
        repl._display_welcome()

    @pytest.mark.asyncio
    async def test_run_with_user_input(self, repl, mock_agent):
        """Test REPL run loop with user input."""
        mock_agent.process.return_value = "Response from agent"

        # Mock the prompt session to provide input and then EOF
        with patch.object(repl.session, "prompt_async", new_callable=AsyncMock) as mock_prompt:
            mock_prompt.side_effect = ["Hello agent", EOFError()]

            # Run the REPL
            await repl.run()

            # Verify agent was called with user input
            mock_agent.process.assert_called_once_with("Hello agent")

    @pytest.mark.asyncio
    async def test_run_with_empty_input(self, repl, mock_agent):
        """Test REPL skips empty input."""
        with patch.object(repl.session, "prompt_async", new_callable=AsyncMock) as mock_prompt:
            mock_prompt.side_effect = ["", "  ", EOFError()]

            await repl.run()

            # Agent should not be called for empty input
            mock_agent.process.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_with_special_command(self, repl, mock_agent):
        """Test REPL handles special commands."""
        with patch.object(repl.session, "prompt_async", new_callable=AsyncMock) as mock_prompt:
            mock_prompt.side_effect = ["/clear", EOFError()]

            await repl.run()

            # Should call clear_history, not process
            mock_agent.clear_history.assert_called_once()
            mock_agent.process.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_with_keyboard_interrupt(self, repl, mock_agent):
        """Test REPL handles Ctrl+C gracefully."""
        with patch.object(repl.session, "prompt_async", new_callable=AsyncMock) as mock_prompt:
            mock_prompt.side_effect = [KeyboardInterrupt(), EOFError()]

            # Should not raise exception
            await repl.run()

    @pytest.mark.asyncio
    async def test_run_with_exception(self, repl, mock_agent):
        """Test REPL handles unexpected exceptions."""
        mock_agent.process.side_effect = Exception("Unexpected error")

        with patch.object(repl.session, "prompt_async", new_callable=AsyncMock) as mock_prompt:
            mock_prompt.side_effect = ["Test input", EOFError()]

            # Should not raise exception, should handle it
            await repl.run()

    @pytest.mark.asyncio
    async def test_run_exit_command_breaks_loop(self, repl, mock_agent):
        """Test that /exit command breaks the REPL loop."""
        with patch.object(repl.session, "prompt_async", new_callable=AsyncMock) as mock_prompt:
            mock_prompt.side_effect = ["/exit"]

            # Run should complete without EOF
            await repl.run()

            # Agent should not be called
            mock_agent.process.assert_not_called()

    def test_display_response_handles_formatting_error(self, repl):
        """Test that response display handles formatting errors gracefully."""
        # Create a response that might cause formatting issues
        problematic_response = "```\nUnclosed code block"

        # Should not raise exception even if formatting fails
        repl._display_response(problematic_response)

    @pytest.mark.asyncio
    async def test_run_processes_multiple_inputs(self, repl, mock_agent):
        """Test REPL processes multiple user inputs in sequence."""
        mock_agent.process.return_value = "Response"

        with patch.object(repl.session, "prompt_async", new_callable=AsyncMock) as mock_prompt:
            mock_prompt.side_effect = [
                "First input",
                "Second input",
                "Third input",
                EOFError(),
            ]

            await repl.run()

            # Should process all three inputs
            assert mock_agent.process.call_count == 3
            mock_agent.process.assert_any_call("First input")
            mock_agent.process.assert_any_call("Second input")
            mock_agent.process.assert_any_call("Third input")

    @pytest.mark.asyncio
    async def test_handle_special_command_preserves_whitespace(self, repl):
        """Test that special commands handle whitespace correctly."""
        # Commands with extra whitespace should still work
        result = await repl._handle_special_command("  /exit  ")
        assert result is False

        result = await repl._handle_special_command("/help   ")
        assert result is True
