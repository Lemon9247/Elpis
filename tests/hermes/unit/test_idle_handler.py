"""Unit tests for IdleHandler (now in hermes.handlers)."""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hermes.handlers.idle_handler import (
    IdleHandler,
    IdleConfig,
    ThoughtEvent,
    SAFE_IDLE_TOOLS,
    SENSITIVE_PATH_PATTERNS,
)
from psyche.memory.compaction import ContextCompactor


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
def handler(mock_client, mock_tool_engine, compactor, tmp_path):
    """Create an IdleHandler instance for testing."""
    config = IdleConfig(
        post_interaction_delay=1.0,
        idle_tool_cooldown_seconds=1.0,
        startup_warmup_seconds=0.0,  # Disable warmup for testing
        workspace_dir=str(tmp_path),
    )
    return IdleHandler(
        elpis_client=mock_client,
        compactor=compactor,
        tool_engine=mock_tool_engine,
        config=config,
    )


class TestCanStartThinking:
    """Tests for can_start_thinking method."""

    def test_can_start_immediately_after_startup(self, handler):
        """Can start thinking if no recent interaction."""
        # Set last interaction far in the past
        handler._last_user_interaction = time.time() - 1000
        assert handler.can_start_thinking() is True

    def test_cannot_start_after_recent_interaction(self, handler):
        """Cannot start thinking right after user interaction."""
        handler.record_user_interaction()
        assert handler.can_start_thinking() is False

    def test_can_start_after_cooldown(self, handler):
        """Can start thinking after cooldown period."""
        handler._last_user_interaction = time.time() - 10  # 10 seconds ago
        handler.config.post_interaction_delay = 5.0  # 5 second delay
        assert handler.can_start_thinking() is True


class TestCanUseTools:
    """Tests for can_use_tools method."""

    def test_can_use_tools_after_warmup(self, handler):
        """Can use tools after startup warmup."""
        handler._startup_time = time.time() - 1000  # Started long ago
        handler._last_idle_tool_use = 0  # Never used
        assert handler.can_use_tools() is True

    def test_cannot_use_tools_during_warmup(self, handler):
        """Cannot use tools during startup warmup."""
        handler.config.startup_warmup_seconds = 120.0
        handler._startup_time = time.time()  # Just started
        assert handler.can_use_tools() is False

    def test_cannot_use_tools_during_cooldown(self, handler):
        """Cannot use tools during cooldown after recent use."""
        handler._startup_time = time.time() - 1000  # Started long ago
        handler._last_idle_tool_use = time.time()  # Just used
        handler.config.idle_tool_cooldown_seconds = 300.0
        assert handler.can_use_tools() is False

    def test_can_use_tools_after_cooldown(self, handler):
        """Can use tools after cooldown period."""
        handler._startup_time = time.time() - 1000
        handler._last_idle_tool_use = time.time() - 10  # Used 10 seconds ago
        handler.config.idle_tool_cooldown_seconds = 5.0  # 5 second cooldown
        assert handler.can_use_tools() is True


class TestValidateToolCall:
    """Tests for validate_tool_call method."""

    def test_valid_safe_tool(self, handler):
        """Valid safe tool passes validation."""
        tool_call = {"name": "read_file", "arguments": {"file_path": "test.txt"}}
        assert handler.validate_tool_call(tool_call) is None

    def test_invalid_unsafe_tool(self, handler):
        """Unsafe tool fails validation."""
        tool_call = {"name": "execute_bash", "arguments": {"command": "rm -rf /"}}
        error = handler.validate_tool_call(tool_call)
        assert error is not None
        assert "not allowed" in error

    def test_all_safe_tools_allowed(self, handler):
        """All tools in SAFE_IDLE_TOOLS are allowed."""
        for tool_name in SAFE_IDLE_TOOLS:
            tool_call = {"name": tool_name, "arguments": {}}
            assert handler.validate_tool_call(tool_call) is None

    def test_sensitive_path_rejected(self, handler, tmp_path):
        """Sensitive paths are rejected."""
        tool_call = {"name": "read_file", "arguments": {"file_path": "/home/user/.ssh/id_rsa"}}
        error = handler.validate_tool_call(tool_call)
        assert error is not None
        assert "not allowed" in error


class TestIsSafePath:
    """Tests for is_safe_path method."""

    def test_safe_relative_path(self, handler, tmp_path):
        """Safe relative path is allowed."""
        # Create the file in workspace
        test_file = tmp_path / "test.txt"
        test_file.touch()
        assert handler.is_safe_path("test.txt") is True

    def test_safe_absolute_path_in_workspace(self, handler, tmp_path):
        """Safe absolute path within workspace is allowed."""
        test_file = tmp_path / "test.txt"
        test_file.touch()
        assert handler.is_safe_path(str(test_file)) is True

    def test_parent_traversal_blocked(self, handler):
        """Parent directory traversal is blocked."""
        assert handler.is_safe_path("../secret.txt") is False
        assert handler.is_safe_path("foo/../../secret.txt") is False

    def test_sensitive_patterns_blocked(self, handler):
        """Sensitive path patterns are blocked."""
        for pattern in [".ssh", ".aws", ".env", "credentials", "secrets"]:
            assert handler.is_safe_path(f"path/to/{pattern}/file") is False

    def test_path_outside_workspace_blocked(self, handler, tmp_path):
        """Paths outside workspace are blocked."""
        # Use absolute path outside workspace
        assert handler.is_safe_path("/etc/passwd") is False
        assert handler.is_safe_path("/root/.bashrc") is False


class TestGetReflectionPrompt:
    """Tests for get_reflection_prompt method."""

    def test_returns_string(self, handler):
        """Returns a non-empty string."""
        prompt = handler.get_reflection_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_contains_instructions(self, handler):
        """Prompt contains key instructions."""
        prompt = handler.get_reflection_prompt()
        assert "INTERNAL REFLECTION" in prompt
        assert "tool_call" in prompt

    def test_randomness(self, handler):
        """Different prompts can be returned."""
        prompts = set()
        for _ in range(20):
            prompts.add(handler.get_reflection_prompt())
        # Should have at least 2 different prompts (could be same by chance)
        assert len(prompts) >= 1


class TestGenerateThought:
    """Tests for generate_thought method."""

    @pytest.mark.asyncio
    async def test_generate_thought_simple(self, handler, mock_client):
        """Generate a simple thought without tools."""
        async def mock_stream(*args, **kwargs):
            for token in ["This", " is", " a", " thought"]:
                yield token

        mock_client.generate_stream = mock_stream
        tokens_received = []

        result = await handler.generate_thought(
            on_token=lambda t: tokens_received.append(t),
        )

        assert result == "This is a thought"
        assert tokens_received == ["This", " is", " a", " thought"]

    @pytest.mark.asyncio
    async def test_generate_thought_disconnected(self, handler, mock_client):
        """Handle disconnected client."""
        mock_client.is_connected = False

        result = await handler.generate_thought()

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_thought_with_callback(self, handler, mock_client):
        """Thought callback is invoked."""
        async def mock_stream(*args, **kwargs):
            for token in ["A thought"]:
                yield token

        mock_client.generate_stream = mock_stream
        thoughts_received = []

        await handler.generate_thought(
            on_thought=lambda t: thoughts_received.append(t),
        )

        assert len(thoughts_received) == 1
        assert isinstance(thoughts_received[0], ThoughtEvent)
        assert thoughts_received[0].thought_type == "reflection"

    @pytest.mark.asyncio
    async def test_generate_thought_is_thinking_flag(self, handler, mock_client):
        """is_thinking flag is managed correctly."""
        thinking_states = []

        async def mock_stream(*args, **kwargs):
            thinking_states.append(handler.is_thinking)
            for token in ["test"]:
                yield token

        mock_client.generate_stream = mock_stream

        assert handler.is_thinking is False
        await handler.generate_thought()
        assert handler.is_thinking is False  # Reset after completion
        assert True in thinking_states  # Was True during generation


class TestInterrupt:
    """Tests for interrupt handling."""

    def test_interrupt_when_thinking(self, handler):
        """Interrupt returns True when thinking."""
        handler._is_thinking = True
        assert handler.interrupt() is True
        assert handler._interrupt_event.is_set()

    def test_interrupt_when_not_thinking(self, handler):
        """Interrupt returns False when not thinking."""
        handler._is_thinking = False
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
        async def mock_stream(*args, **kwargs):
            for i, token in enumerate(["Hello", " ", "world"]):
                if i == 1:  # Interrupt after second token
                    handler._interrupt_event.set()
                yield token

        mock_client.generate_stream = mock_stream

        result = await handler.generate_thought()

        # Should return None when interrupted
        assert result is None


class TestRecordInteraction:
    """Tests for record_user_interaction method."""

    def test_record_user_interaction_updates_time(self, handler):
        """Recording interaction updates timestamp."""
        old_time = handler._last_user_interaction
        time.sleep(0.01)  # Small delay
        handler.record_user_interaction()
        assert handler._last_user_interaction > old_time

    def test_record_user_interaction_increments_count(self, handler):
        """Recording interaction increments counter."""
        old_count = handler._interaction_count
        handler.record_user_interaction()
        assert handler._interaction_count == old_count + 1


class TestRecordToolUse:
    """Tests for record_tool_use method."""

    def test_record_tool_use_updates_time(self, handler):
        """Recording tool use updates timestamp."""
        old_time = handler._last_idle_tool_use
        time.sleep(0.01)  # Small delay
        handler.record_tool_use()
        assert handler._last_idle_tool_use > old_time


class TestThoughtEvent:
    """Tests for ThoughtEvent dataclass."""

    def test_thought_event_creation(self):
        """Create thought event with all fields."""
        event = ThoughtEvent(
            content="Test thought",
            thought_type="reflection",
            triggered_by="idle",
        )

        assert event.content == "Test thought"
        assert event.thought_type == "reflection"
        assert event.triggered_by == "idle"

    def test_thought_event_optional_triggered_by(self):
        """Create thought event without triggered_by."""
        event = ThoughtEvent(
            content="Test",
            thought_type="planning",
        )

        assert event.triggered_by is None


class TestIdleConfig:
    """Tests for IdleConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IdleConfig()

        assert config.post_interaction_delay == 60.0
        assert config.idle_tool_cooldown_seconds == 300.0
        assert config.startup_warmup_seconds == 120.0
        assert config.max_idle_tool_iterations == 3
        assert config.allow_idle_tools is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = IdleConfig(
            post_interaction_delay=30.0,
            idle_tool_cooldown_seconds=60.0,
            startup_warmup_seconds=0.0,
            allow_idle_tools=False,
        )

        assert config.post_interaction_delay == 30.0
        assert config.idle_tool_cooldown_seconds == 60.0
        assert config.startup_warmup_seconds == 0.0
        assert config.allow_idle_tools is False


class TestSafeIdleTools:
    """Tests for SAFE_IDLE_TOOLS constant."""

    def test_contains_expected_tools(self):
        """Contains expected safe tools."""
        expected = {"read_file", "list_directory", "search_codebase", "recall_memory"}
        assert SAFE_IDLE_TOOLS == expected

    def test_does_not_contain_dangerous_tools(self):
        """Does not contain dangerous tools."""
        dangerous = {"execute_bash", "write_file", "create_file", "edit_file", "delete_file"}
        for tool in dangerous:
            assert tool not in SAFE_IDLE_TOOLS


class TestSensitivePathPatterns:
    """Tests for SENSITIVE_PATH_PATTERNS constant."""

    def test_contains_ssh_patterns(self):
        """Contains SSH-related patterns."""
        assert ".ssh" in SENSITIVE_PATH_PATTERNS
        assert "id_rsa" in SENSITIVE_PATH_PATTERNS
        assert "id_ed25519" in SENSITIVE_PATH_PATTERNS

    def test_contains_cloud_patterns(self):
        """Contains cloud credential patterns."""
        assert ".aws" in SENSITIVE_PATH_PATTERNS
        assert ".azure" in SENSITIVE_PATH_PATTERNS
        assert ".gcloud" in SENSITIVE_PATH_PATTERNS

    def test_contains_env_patterns(self):
        """Contains environment file patterns."""
        assert ".env" in SENSITIVE_PATH_PATTERNS
        assert "credentials" in SENSITIVE_PATH_PATTERNS
        assert "secrets" in SENSITIVE_PATH_PATTERNS
