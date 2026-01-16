"""Unit tests for PsycheCore."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from psyche.core.server import CoreConfig, PsycheCore, REASONING_PROMPT


class MockEmotionalState:
    """Mock emotional state for testing."""

    def __init__(self, valence=0.0, arousal=0.0, quadrant="neutral"):
        self.valence = valence
        self.arousal = arousal
        self.quadrant = quadrant


class MockGenerationResult:
    """Mock generation result for testing."""

    def __init__(self, content="Test response", emotional_state=None):
        self.content = content
        self.emotional_state = emotional_state or MockEmotionalState()


class TestCoreConfig:
    """Tests for CoreConfig dataclass."""

    def test_default_values(self):
        """CoreConfig should have sensible defaults."""
        config = CoreConfig()
        assert config.reasoning_enabled is True
        assert config.auto_storage is True
        assert config.auto_storage_threshold == 0.6
        assert config.emotional_modulation is True
        assert config.context is not None
        assert config.memory is not None

    def test_custom_values(self):
        """CoreConfig should accept custom values."""
        config = CoreConfig(
            reasoning_enabled=False,
            auto_storage=False,
            auto_storage_threshold=0.8,
            emotional_modulation=False,
        )
        assert config.reasoning_enabled is False
        assert config.auto_storage is False
        assert config.auto_storage_threshold == 0.8
        assert config.emotional_modulation is False


class TestPsycheCoreInit:
    """Tests for PsycheCore initialization."""

    def test_initialization_with_elpis_only(self):
        """PsycheCore should initialize with just Elpis client."""
        elpis = MagicMock()
        elpis.is_connected = True

        core = PsycheCore(elpis_client=elpis)

        assert core.elpis is elpis
        assert core.mnemosyne is None
        assert core.config is not None
        assert core.reasoning_enabled is True

    def test_initialization_with_mnemosyne(self):
        """PsycheCore should accept Mnemosyne client."""
        elpis = MagicMock()
        elpis.is_connected = True
        mnemosyne = MagicMock()
        mnemosyne.is_connected = True

        core = PsycheCore(elpis_client=elpis, mnemosyne_client=mnemosyne)

        assert core.elpis is elpis
        assert core.mnemosyne is mnemosyne

    def test_initialization_with_custom_config(self):
        """PsycheCore should use custom config."""
        elpis = MagicMock()
        config = CoreConfig(reasoning_enabled=False, auto_storage_threshold=0.9)

        core = PsycheCore(elpis_client=elpis, config=config)

        assert core.config.reasoning_enabled is False
        assert core.config.auto_storage_threshold == 0.9


class TestSystemPrompt:
    """Tests for system prompt building."""

    def test_build_system_prompt_includes_base(self):
        """System prompt should include base prompt."""
        elpis = MagicMock()
        core = PsycheCore(elpis_client=elpis)
        core.initialize()

        prompt = core._build_system_prompt()
        assert "Psyche" in prompt
        assert "thoughtful AI assistant" in prompt

    def test_build_system_prompt_with_reasoning(self):
        """System prompt should include reasoning when enabled."""
        elpis = MagicMock()
        config = CoreConfig(reasoning_enabled=True)
        core = PsycheCore(elpis_client=elpis, config=config)

        prompt = core._build_system_prompt()
        assert "Reasoning Mode" in prompt
        assert "<reasoning>" in prompt

    def test_build_system_prompt_without_reasoning(self):
        """System prompt should exclude reasoning when disabled."""
        elpis = MagicMock()
        config = CoreConfig(reasoning_enabled=False)
        core = PsycheCore(elpis_client=elpis, config=config)

        prompt = core._build_system_prompt()
        assert "Reasoning Mode" not in prompt

    def test_set_tool_descriptions(self):
        """set_tool_descriptions should add tools to prompt."""
        elpis = MagicMock()
        core = PsycheCore(elpis_client=elpis)

        core.set_tool_descriptions("read_file: Read a file\nwrite_file: Write a file")
        prompt = core._build_system_prompt()

        assert "read_file" in prompt
        assert "Tool Usage" in prompt
        assert "tool_call" in prompt


class TestReasoningMode:
    """Tests for reasoning mode toggle."""

    def test_reasoning_enabled_by_default(self):
        """Reasoning should be enabled by default."""
        elpis = MagicMock()
        core = PsycheCore(elpis_client=elpis)

        assert core.reasoning_enabled is True

    def test_set_reasoning_mode_on(self):
        """set_reasoning_mode should enable reasoning."""
        elpis = MagicMock()
        config = CoreConfig(reasoning_enabled=False)
        core = PsycheCore(elpis_client=elpis, config=config)

        core.set_reasoning_mode(True)

        assert core.reasoning_enabled is True

    def test_set_reasoning_mode_off(self):
        """set_reasoning_mode should disable reasoning."""
        elpis = MagicMock()
        core = PsycheCore(elpis_client=elpis)

        core.set_reasoning_mode(False)

        assert core.reasoning_enabled is False

    def test_set_reasoning_mode_rebuilds_prompt(self):
        """Toggling reasoning mode should rebuild system prompt."""
        elpis = MagicMock()
        core = PsycheCore(elpis_client=elpis)
        core.initialize()

        # Get initial prompt
        initial_prompt = core._build_system_prompt()
        assert "Reasoning Mode" in initial_prompt

        # Toggle off
        core.set_reasoning_mode(False)
        new_prompt = core._build_system_prompt()
        assert "Reasoning Mode" not in new_prompt


class TestAddUserMessage:
    """Tests for adding user messages."""

    @pytest.mark.asyncio
    async def test_add_user_message_without_mnemosyne(self):
        """add_user_message should work without Mnemosyne."""
        elpis = MagicMock()
        elpis.is_connected = True
        core = PsycheCore(elpis_client=elpis)
        core.initialize()

        result = await core.add_user_message("Hello, world!")

        # No memories without Mnemosyne
        assert result is None
        # Message should be in context
        messages = core.get_api_messages()
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1
        assert "Hello, world!" in user_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_add_user_message_with_memory_retrieval(self):
        """add_user_message should retrieve memories from Mnemosyne."""
        elpis = MagicMock()
        elpis.is_connected = True
        mnemosyne = AsyncMock()
        mnemosyne.is_connected = True
        mnemosyne.search_memories.return_value = [
            {"content": "Previous conversation about testing", "memory_type": "episodic"},
        ]

        core = PsycheCore(elpis_client=elpis, mnemosyne_client=mnemosyne)
        core.initialize()

        result = await core.add_user_message("Tell me about testing")

        # Should have retrieved memories
        assert result is not None
        assert "testing" in result.lower()


class TestAddAssistantMessage:
    """Tests for adding assistant messages."""

    @pytest.mark.asyncio
    async def test_add_assistant_message_basic(self):
        """add_assistant_message should add to context."""
        elpis = MagicMock()
        elpis.is_connected = True
        core = PsycheCore(elpis_client=elpis)
        core.initialize()

        await core.add_assistant_message("This is my response")

        messages = core.get_api_messages()
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) == 1
        assert assistant_messages[0]["content"] == "This is my response"

    @pytest.mark.asyncio
    async def test_add_assistant_message_auto_storage_disabled(self):
        """Auto-storage should not occur when disabled."""
        elpis = MagicMock()
        elpis.is_connected = True
        mnemosyne = AsyncMock()
        mnemosyne.is_connected = True

        config = CoreConfig(auto_storage=False)
        core = PsycheCore(elpis_client=elpis, mnemosyne_client=mnemosyne, config=config)
        core.initialize()

        await core.add_assistant_message("Response", user_message="Important message remember this")

        # Should not have stored anything
        mnemosyne.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_assistant_message_auto_storage_triggered(self):
        """Auto-storage should occur for important exchanges."""
        elpis = AsyncMock()
        elpis.is_connected = True
        elpis.get_emotion.return_value = MockEmotionalState(valence=0.8, arousal=0.8)

        mnemosyne = AsyncMock()
        mnemosyne.is_connected = True

        config = CoreConfig(auto_storage=True, auto_storage_threshold=0.5)
        core = PsycheCore(elpis_client=elpis, mnemosyne_client=mnemosyne, config=config)
        core.initialize()

        # Message with "remember" should trigger auto-storage
        long_response = "x" * 1100  # Long response for length score
        await core.add_assistant_message(
            long_response,
            user_message="Please remember this important fact",
        )

        # Should have stored the memory
        mnemosyne.store_memory.assert_called_once()


class TestAddToolResult:
    """Tests for adding tool results."""

    def test_add_tool_result(self):
        """add_tool_result should add formatted result to context."""
        elpis = MagicMock()
        core = PsycheCore(elpis_client=elpis)
        core.initialize()

        core.add_tool_result("read_file", "File contents here")

        messages = core.get_api_messages()
        # Tool results are added as user messages
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1
        assert "read_file" in user_messages[0]["content"]
        assert "File contents here" in user_messages[0]["content"]


class TestGenerate:
    """Tests for generation."""

    @pytest.mark.asyncio
    async def test_generate_basic(self):
        """generate should call Elpis and return response."""
        elpis = AsyncMock()
        elpis.is_connected = True
        elpis.generate.return_value = MockGenerationResult(content="Generated response")

        core = PsycheCore(elpis_client=elpis)
        core.initialize()

        result = await core.generate()

        assert result["content"] == "Generated response"
        assert result["has_thinking"] is False

    @pytest.mark.asyncio
    async def test_generate_with_reasoning_extraction(self):
        """generate should extract reasoning from response."""
        elpis = AsyncMock()
        elpis.is_connected = True
        elpis.generate.return_value = MockGenerationResult(
            content="<reasoning>Let me think...</reasoning>The answer is 42."
        )

        config = CoreConfig(reasoning_enabled=True)
        core = PsycheCore(elpis_client=elpis, config=config)
        core.initialize()

        result = await core.generate()

        assert result["content"] == "The answer is 42."
        assert result["thinking"] == "Let me think..."
        assert result["has_thinking"] is True

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """generate_stream should yield tokens."""
        elpis = AsyncMock()
        elpis.is_connected = True

        async def mock_stream(*args, **kwargs):
            for token in ["Hello", " ", "World"]:
                yield token

        elpis.generate_stream = mock_stream

        core = PsycheCore(elpis_client=elpis)
        core.initialize()

        tokens = []
        async for token in core.generate_stream():
            tokens.append(token)

        assert tokens == ["Hello", " ", "World"]


class TestMemoryOperations:
    """Tests for explicit memory operations."""

    @pytest.mark.asyncio
    async def test_retrieve_memories(self):
        """retrieve_memories should query Mnemosyne."""
        elpis = MagicMock()
        elpis.is_connected = True
        mnemosyne = AsyncMock()
        mnemosyne.is_connected = True
        mnemosyne.search_memories.return_value = [
            {"content": "Memory 1", "memory_type": "episodic"},
            {"content": "Memory 2", "memory_type": "semantic"},
        ]

        core = PsycheCore(elpis_client=elpis, mnemosyne_client=mnemosyne)

        memories = await core.retrieve_memories("test query", n=5)

        assert len(memories) == 2
        mnemosyne.search_memories.assert_called_once_with("test query", n_results=5)

    @pytest.mark.asyncio
    async def test_store_memory(self):
        """store_memory should save to Mnemosyne."""
        elpis = AsyncMock()
        elpis.is_connected = True
        elpis.get_emotion.return_value = MockEmotionalState()

        mnemosyne = AsyncMock()
        mnemosyne.is_connected = True

        core = PsycheCore(elpis_client=elpis, mnemosyne_client=mnemosyne)

        result = await core.store_memory("Important fact to remember", tags=["test"])

        assert result is True
        mnemosyne.store_memory.assert_called_once()
        call_kwargs = mnemosyne.store_memory.call_args.kwargs
        assert "Important fact" in call_kwargs["content"]
        assert "test" in call_kwargs["tags"]

    @pytest.mark.asyncio
    async def test_store_memory_without_mnemosyne(self):
        """store_memory should return False without Mnemosyne."""
        elpis = MagicMock()
        core = PsycheCore(elpis_client=elpis)

        result = await core.store_memory("This won't be stored")

        assert result is False


class TestEmotionalState:
    """Tests for emotional state management."""

    @pytest.mark.asyncio
    async def test_get_emotion(self):
        """get_emotion should return current state from Elpis."""
        elpis = AsyncMock()
        elpis.is_connected = True
        elpis.get_emotion.return_value = MockEmotionalState(
            valence=0.5, arousal=0.3, quadrant="happy"
        )

        core = PsycheCore(elpis_client=elpis)

        emotion = await core.get_emotion()

        assert emotion["valence"] == 0.5
        assert emotion["arousal"] == 0.3
        assert emotion["quadrant"] == "happy"

    @pytest.mark.asyncio
    async def test_get_emotion_disconnected(self):
        """get_emotion should return defaults when disconnected."""
        elpis = MagicMock()
        elpis.is_connected = False

        core = PsycheCore(elpis_client=elpis)

        emotion = await core.get_emotion()

        assert emotion["valence"] == 0.0
        assert emotion["arousal"] == 0.0
        assert emotion["quadrant"] == "neutral"

    @pytest.mark.asyncio
    async def test_update_emotion(self):
        """update_emotion should update state in Elpis."""
        elpis = AsyncMock()
        elpis.is_connected = True
        elpis.update_emotion.return_value = MockEmotionalState(
            valence=0.8, arousal=0.5, quadrant="excited"
        )

        core = PsycheCore(elpis_client=elpis)

        emotion = await core.update_emotion("success", intensity=1.0)

        assert emotion["valence"] == 0.8
        elpis.update_emotion.assert_called_once_with("success", 1.0)


class TestContextManagement:
    """Tests for context management."""

    def test_context_summary(self):
        """context_summary should return context state."""
        elpis = MagicMock()
        core = PsycheCore(elpis_client=elpis)
        core.initialize()

        summary = core.context_summary

        assert "message_count" in summary
        assert "total_tokens" in summary
        assert "has_system_prompt" in summary

    def test_clear_context(self):
        """clear_context should reset context."""
        elpis = MagicMock()
        core = PsycheCore(elpis_client=elpis)
        core.initialize()
        core._context.add_message("user", "Hello")
        core._context.add_message("assistant", "Hi")

        core.clear_context()

        # Should have cleared messages (system prompt may remain)
        messages = core.get_api_messages()
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 0

    def test_is_mnemosyne_available(self):
        """is_mnemosyne_available should check connection."""
        elpis = MagicMock()

        # Without Mnemosyne
        core = PsycheCore(elpis_client=elpis)
        assert core.is_mnemosyne_available is False

        # With disconnected Mnemosyne
        mnemosyne = MagicMock()
        mnemosyne.is_connected = False
        core = PsycheCore(elpis_client=elpis, mnemosyne_client=mnemosyne)
        assert core.is_mnemosyne_available is False

        # With connected Mnemosyne
        mnemosyne.is_connected = True
        core = PsycheCore(elpis_client=elpis, mnemosyne_client=mnemosyne)
        assert core.is_mnemosyne_available is True


class TestShutdown:
    """Tests for shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_calls_consolidate(self):
        """shutdown should flush staged messages."""
        elpis = AsyncMock()
        elpis.is_connected = True
        elpis.get_emotion.return_value = MockEmotionalState()

        mnemosyne = AsyncMock()
        mnemosyne.is_connected = True

        core = PsycheCore(elpis_client=elpis, mnemosyne_client=mnemosyne)
        core.initialize()

        await core.shutdown()

        # Should have attempted to flush staged messages
        # (The actual implementation details depend on memory_handler)

    @pytest.mark.asyncio
    async def test_consolidate(self):
        """consolidate should flush staged messages."""
        elpis = AsyncMock()
        elpis.is_connected = True
        elpis.get_emotion.return_value = MockEmotionalState()

        mnemosyne = AsyncMock()
        mnemosyne.is_connected = True

        core = PsycheCore(elpis_client=elpis, mnemosyne_client=mnemosyne)
        core.initialize()

        # This should not raise
        await core.consolidate()


class TestGetApiMessages:
    """Tests for API message formatting."""

    def test_get_api_messages(self):
        """get_api_messages should return formatted messages."""
        elpis = MagicMock()
        core = PsycheCore(elpis_client=elpis)
        core.initialize()

        # Add some messages
        core._context.add_message("user", "Hello")
        core._context.add_message("assistant", "Hi there")

        messages = core.get_api_messages()

        # Should have system + user + assistant
        assert len(messages) >= 2
        user_msg = next(m for m in messages if m["role"] == "user")
        assert user_msg["content"] == "Hello"


class TestInitialize:
    """Tests for initialization."""

    def test_initialize_sets_system_prompt(self):
        """initialize should set up system prompt in context."""
        elpis = MagicMock()
        core = PsycheCore(elpis_client=elpis)

        core.initialize()

        messages = core.get_api_messages()
        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) >= 1
        assert "Psyche" in system_messages[0]["content"]
