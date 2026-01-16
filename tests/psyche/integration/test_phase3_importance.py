"""
Integration tests for Phase 3 importance scoring and auto-storage.

Tests verify:
- High importance exchanges auto-stored
- Low importance exchanges not stored
- Explicit "remember" triggers storage
- Threshold is configurable
- Graceful handling when Mnemosyne unavailable
- Emotional intensity affects scoring
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from psyche.mcp.client import ElpisClient, EmotionalState, GenerationResult
from psyche.memory.server import MemoryServer, ServerConfig, ServerState
from psyche.memory.importance import calculate_importance, is_worth_storing, ImportanceScore


class MockElpisClient:
    """Mock ElpisClient for testing importance workflow."""

    def __init__(self):
        self.generate_responses = []
        self.response_index = 0
        self.emotion_state = EmotionalState()
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def generate_stream(
        self,
        messages,
        max_tokens=2048,
        temperature=None,
        emotional_modulation=True,
        poll_interval=0.05,
    ):
        """Yield response content as tokens."""
        if self.response_index < len(self.generate_responses):
            content = self.generate_responses[self.response_index]
            self.response_index += 1
        else:
            content = "Default response"

        for char in content:
            yield char

    async def get_emotion(self):
        return self.emotion_state

    async def update_emotion(self, event_type: str, intensity: float = 1.0):
        return self.emotion_state

    def connect(self):
        return MockConnectionContext(self)


class MockMnemosyneClient:
    """Mock MnemosyneClient for testing auto-storage."""

    def __init__(self):
        self.stored_memories = []
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def store_memory(
        self,
        content: str,
        summary: str = None,
        memory_type: str = "episodic",
        emotional_context: dict = None,
        tags: list = None,
        **kwargs,
    ):
        """Track stored memories."""
        self.stored_memories.append({
            "content": content,
            "summary": summary,
            "memory_type": memory_type,
            "emotional_context": emotional_context,
            "tags": tags or [],
        })
        return {"success": True}

    async def recall_memories(self, query: str, n_results: int = 5):
        return []

    def connect(self):
        return MockConnectionContext(self)


class MockConnectionContext:
    """Mock async context manager for client connection."""

    def __init__(self, client):
        self.client = client

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_client():
    """Create a mock Elpis client."""
    return MockElpisClient()


@pytest.fixture
def mock_mnemosyne():
    """Create a mock Mnemosyne client."""
    return MockMnemosyneClient()


@pytest.fixture
def server_config():
    """Create server config for testing with auto-storage enabled."""
    return ServerConfig(
        idle_think_interval=60.0,
        max_context_tokens=4000,
        reserve_tokens=500,
        allow_idle_tools=False,
        auto_storage=True,
        auto_storage_threshold=0.6,
    )


class TestImportanceScoring:
    """Tests for importance calculation."""

    def test_high_importance_code_response(self):
        """Response with code blocks should have high importance."""
        # Need a response > 500 chars with multiple code blocks
        response = """Here's the complete solution to fix your bug:

```python
def fix_bug(input_data):
    \"\"\"Fix the bug by properly validating input.\"\"\"
    if not input_data:
        return None
    # Process the data correctly
    result = process_data(input_data)
    return result
```

Additionally, here's a helper function you'll need:

```python
def process_data(data):
    \"\"\"Process the input data with proper error handling.\"\"\"
    try:
        cleaned = data.strip()
        validated = validate(cleaned)
        return validated
    except ValueError as e:
        logger.error(f"Validation failed: {e}")
        raise
```

This should resolve the issue completely. The key changes are:
1. Added input validation
2. Proper error handling
3. Logging for debugging"""

        score = calculate_importance("Fix this bug", response)

        assert score.code_score >= 0.25  # Multiple code blocks
        assert score.length_score >= 0.15  # > 500 chars
        assert score.total >= 0.4

    def test_low_importance_simple_response(self):
        """Simple short response should have low importance."""
        score = calculate_importance("Hello", "Hi there!")

        assert score.total < 0.2

    def test_explicit_remember_high_importance(self):
        """Explicit 'remember' request should boost importance."""
        score = calculate_importance(
            "Please remember this: my API key is stored in .env",
            "I'll remember that your API key is in .env."
        )

        assert score.explicit_score >= 0.4
        assert score.total >= 0.4

    def test_tool_results_increase_importance(self):
        """Tool execution should increase importance."""
        tool_results = [
            {"tool": "read_file", "result": "file contents"},
            {"tool": "write_file", "result": "success"},
        ]

        score = calculate_importance(
            "Read and update the config",
            "Done, I've updated the config file.",
            tool_results=tool_results,
        )

        assert score.tool_score >= 0.2

    def test_emotional_intensity_affects_score(self):
        """High emotional intensity should increase importance."""
        emotion = {"valence": 0.8, "arousal": 0.6}

        score = calculate_importance(
            "This is frustrating",
            "I understand your frustration.",
            emotion=emotion,
        )

        assert score.emotion_score >= 0.15


class TestAutoStorageIntegration:
    """Tests for auto-storage integration with MemoryServer."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_high_importance_exchange_auto_stored(
        self, mock_client, mock_mnemosyne, server_config
    ):
        """High importance exchange should be auto-stored to Mnemosyne."""
        # Response must be > 1000 chars with multiple code blocks to hit 0.6 threshold
        # (0.35 code + 0.35 length = 0.70)
        mock_client.generate_responses = [
            """Here's the complete fix for your bug with detailed explanation:

```python
def solve_problem(input_data):
    \"\"\"
    Solve the problem by properly processing input data.

    This function handles edge cases and validates input before processing.
    It returns None for invalid input and the processed result otherwise.

    Args:
        input_data: The data to process

    Returns:
        Processed result or None if invalid
    \"\"\"
    if not input_data:
        return None

    # Validate the input format
    if not isinstance(input_data, (str, bytes)):
        raise TypeError("Input must be string or bytes")

    # Process the validated data
    result = process_data(input_data)
    return result
```

Here's the helper function that does the actual processing:

```python
def process_data(data):
    \"\"\"Process the input data with comprehensive error handling.\"\"\"
    try:
        # Clean the input
        cleaned = data.strip() if isinstance(data, str) else data.decode().strip()

        # Validate the cleaned data
        if not cleaned:
            raise ValueError("Empty input after cleaning")

        # Perform the actual processing
        validated = validate_and_transform(cleaned)
        return validated

    except ValueError as e:
        logger.error(f"Validation failed: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during processing: {e}")
        raise RuntimeError("Processing failed") from e
```

This implementation resolves the issue by:
1. Adding proper input validation at multiple levels
2. Implementing comprehensive error handling
3. Adding logging for debugging and monitoring
4. Handling both string and bytes input types"""
        ]

        server = MemoryServer(
            mock_client,
            server_config,
            mnemosyne_client=mock_mnemosyne,
        )
        server.client = mock_client

        await server._process_user_input("Please fix the bug in my code")

        # Check that memory was stored
        assert len(mock_mnemosyne.stored_memories) == 1
        stored = mock_mnemosyne.stored_memories[0]
        assert "auto-stored" in stored["tags"]
        assert "important" in stored["tags"]
        assert "solve_problem" in stored["content"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_low_importance_exchange_not_stored(
        self, mock_client, mock_mnemosyne, server_config
    ):
        """Low importance exchange should not be auto-stored."""
        mock_client.generate_responses = ["Hello! How can I help you today?"]

        server = MemoryServer(
            mock_client,
            server_config,
            mnemosyne_client=mock_mnemosyne,
        )
        server.client = mock_client

        await server._process_user_input("Hi")

        # No memory should be stored for simple greeting
        assert len(mock_mnemosyne.stored_memories) == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_explicit_remember_triggers_storage(
        self, mock_client, mock_mnemosyne, server_config
    ):
        """'Remember this' phrase should trigger storage."""
        # Explicit "remember" (0.4) + length > 500 (0.25) = 0.65, above 0.6 threshold
        mock_client.generate_responses = [
            """I'll remember that your preferred editor is neovim. Here's a summary of your preferences that I'm storing:

1. **Editor**: neovim (nvim)
   - You prefer using neovim as your primary text editor
   - This is a modal editor based on Vim but with better extensibility

2. **Configuration Notes**:
   - You likely have custom configurations in ~/.config/nvim/
   - Common plugins include telescope, treesitter, and LSP support
   - Your colorscheme and keybindings are personalized

3. **Workflow Integration**:
   - You use neovim for coding, writing, and general text editing
   - Terminal-based workflow is important to you
   - You may use tmux or similar for session management

I've stored this preference so I can tailor my suggestions accordingly in future conversations. For example, when showing code examples or discussing editor features, I'll keep your neovim preference in mind."""
        ]

        server = MemoryServer(
            mock_client,
            server_config,
            mnemosyne_client=mock_mnemosyne,
        )
        server.client = mock_client

        await server._process_user_input("Please remember this: I prefer using neovim")

        # Should be stored due to explicit request + length
        assert len(mock_mnemosyne.stored_memories) == 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_auto_storage_disabled_prevents_storage(
        self, mock_client, mock_mnemosyne
    ):
        """When auto_storage=False, no auto-storage should occur."""
        config = ServerConfig(
            auto_storage=False,  # Disabled
            idle_think_interval=60.0,
        )

        mock_client.generate_responses = [
            """Here's important code:
```python
def important():
    pass
```
"""
        ]

        server = MemoryServer(
            mock_client,
            config,
            mnemosyne_client=mock_mnemosyne,
        )
        server.client = mock_client

        await server._process_user_input("Remember this important code")

        # Even with high importance, no storage when disabled
        assert len(mock_mnemosyne.stored_memories) == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_graceful_handling_mnemosyne_unavailable(
        self, mock_client, server_config
    ):
        """Should handle Mnemosyne being unavailable gracefully."""
        mock_client.generate_responses = [
            """Important response with code:
```python
def critical():
    pass
```
"""
        ]

        # No Mnemosyne client
        server = MemoryServer(
            mock_client,
            server_config,
            mnemosyne_client=None,
        )
        server.client = mock_client

        # Should not raise even with high importance exchange
        await server._process_user_input("Remember this critical code")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mnemosyne_disconnected_no_crash(
        self, mock_client, mock_mnemosyne, server_config
    ):
        """Should not crash when Mnemosyne is disconnected."""
        mock_mnemosyne._connected = False  # Simulate disconnection

        mock_client.generate_responses = [
            """Important code:
```python
def important():
    pass
```
"""
        ]

        server = MemoryServer(
            mock_client,
            server_config,
            mnemosyne_client=mock_mnemosyne,
        )
        server.client = mock_client

        # Should not raise
        await server._process_user_input("Remember this")

        # No storage when disconnected
        assert len(mock_mnemosyne.stored_memories) == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_threshold_configuration_respected(
        self, mock_client, mock_mnemosyne
    ):
        """Custom threshold should be respected."""
        # Set very high threshold
        config = ServerConfig(
            auto_storage=True,
            auto_storage_threshold=0.95,  # Very high
            idle_think_interval=60.0,
        )

        mock_client.generate_responses = [
            """Here's code:
```python
def solution():
    pass
```
"""
        ]

        server = MemoryServer(
            mock_client,
            config,
            mnemosyne_client=mock_mnemosyne,
        )
        server.client = mock_client

        await server._process_user_input("Fix this")

        # With very high threshold, even code response shouldn't be stored
        assert len(mock_mnemosyne.stored_memories) == 0
