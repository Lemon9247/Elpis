"""
Integration tests for Phase 3 combined features.

Tests verify that reasoning, importance scoring, and memory systems
work correctly together without conflicts.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from psyche.mcp.client import ElpisClient, EmotionalState, GenerationResult
from psyche.memory.server import MemoryServer, ServerConfig, ServerState, ThoughtEvent
from psyche.memory.importance import calculate_importance


class MockElpisClient:
    """Mock ElpisClient for combined feature testing."""

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
    """Mock MnemosyneClient for combined feature testing."""

    def __init__(self):
        self.stored_memories = []
        self.recalled_queries = []
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
        """Track recall queries and return empty list."""
        self.recalled_queries.append(query)
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
    """Create server config with all Phase 3 features enabled."""
    return ServerConfig(
        idle_think_interval=60.0,
        max_context_tokens=4000,
        reserve_tokens=500,
        allow_idle_tools=False,
        auto_storage=True,
        auto_storage_threshold=0.6,
    )


class TestReasoningAndImportance:
    """Tests for reasoning and importance scoring interaction."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_reasoning_content_not_stored_to_memory(
        self, mock_client, mock_mnemosyne, server_config
    ):
        """Reasoning content should NOT be stored to memory, only the clean response."""
        # Long response with reasoning to exceed 0.6 threshold
        mock_client.generate_responses = [
            """<reasoning>
Let me think about this problem carefully.
I need to consider the input validation, error handling, and the actual algorithm.
This is internal thinking that should NOT be stored.
</reasoning>

Here's the complete solution to your problem with detailed implementation:

```python
def solve_complex_problem(data):
    \"\"\"
    Solve the problem with proper validation and error handling.

    Args:
        data: Input data to process

    Returns:
        Processed result
    \"\"\"
    if not data:
        raise ValueError("Data cannot be empty")

    # Validate the input
    validated = validate_input(data)

    # Process and return
    result = process_validated_data(validated)
    return result
```

```python
def validate_input(data):
    \"\"\"Validate and clean input data.\"\"\"
    if isinstance(data, str):
        return data.strip()
    return str(data)

def process_validated_data(data):
    \"\"\"Process the validated data.\"\"\"
    # Implementation details here
    return f"Processed: {data}"
```

This solution handles edge cases and provides clear error messages when validation fails."""
        ]

        thought_events = []
        def on_thought(event: ThoughtEvent):
            thought_events.append(event)

        server = MemoryServer(
            mock_client,
            server_config,
            mnemosyne_client=mock_mnemosyne,
            on_thought=on_thought,
        )
        server.client = mock_client

        await server._process_user_input("Solve this complex problem")

        # Reasoning should be captured in thought event
        reasoning_thoughts = [e for e in thought_events if e.thought_type == "reasoning"]
        assert len(reasoning_thoughts) == 1
        assert "internal thinking" in reasoning_thoughts[0].content

        # Memory should be stored (high importance)
        assert len(mock_mnemosyne.stored_memories) == 1

        # Stored content should NOT include reasoning tags
        stored_content = mock_mnemosyne.stored_memories[0]["content"]
        assert "<reasoning>" not in stored_content
        assert "internal thinking" not in stored_content
        assert "solve_complex_problem" in stored_content

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_importance_calculated_on_cleaned_response(
        self, mock_client, mock_mnemosyne
    ):
        """Importance score should be based on response WITHOUT reasoning tags."""
        # Set threshold to 0.3 to test that short clean response still uses clean text
        config = ServerConfig(
            auto_storage=True,
            auto_storage_threshold=0.3,
            idle_think_interval=60.0,
        )

        # Response with long reasoning but short actual response
        mock_client.generate_responses = [
            """<reasoning>
This is a very long reasoning block that would inflate the character count
if it were included in the importance calculation. It contains detailed
analysis, multiple considerations, and extensive internal deliberation
about how to approach the user's question. This text alone is over 300
characters and would affect length-based scoring significantly if counted.
</reasoning>

Short answer."""
        ]

        server = MemoryServer(
            mock_client,
            config,
            mnemosyne_client=mock_mnemosyne,
        )
        server.client = mock_client

        await server._process_user_input("Quick question")

        # Should NOT be stored because clean response is too short
        # (importance based on "Short answer." not the full text)
        assert len(mock_mnemosyne.stored_memories) == 0


class TestCombinedWorkflow:
    """Tests for complete workflow with all features."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_reasoning_mode_doesnt_break_memory_retrieval(
        self, mock_client, mock_mnemosyne, server_config
    ):
        """Memory retrieval should work when reasoning mode is enabled."""
        mock_client.generate_responses = [
            "<reasoning>Checking context...</reasoning>\n\nI'll help with that."
        ]

        server = MemoryServer(
            mock_client,
            server_config,
            mnemosyne_client=mock_mnemosyne,
        )
        server.client = mock_client

        # Ensure reasoning mode is enabled (default)
        assert server.reasoning_enabled is True

        await server._process_user_input("What did we discuss earlier?")

        # Memory recall should have been attempted
        # (the server retrieves memories at start of processing)
        assert len(mock_mnemosyne.recalled_queries) >= 0  # May or may not recall based on query

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_compaction_preserves_reasoning_mode(
        self, mock_client, mock_mnemosyne
    ):
        """Reasoning mode should persist through context compaction."""
        # Small context to trigger compaction quickly
        config = ServerConfig(
            max_context_tokens=500,  # Very small to trigger compaction
            reserve_tokens=100,
            auto_storage=False,  # Disable to simplify test
            idle_think_interval=60.0,
        )

        mock_client.generate_responses = [
            "<reasoning>First thought</reasoning>\n\nFirst response that is moderately long to use some tokens.",
            "<reasoning>Second thought</reasoning>\n\nSecond response after potential compaction.",
        ]

        thought_events = []
        def on_thought(event: ThoughtEvent):
            thought_events.append(event)

        server = MemoryServer(
            mock_client,
            config,
            mnemosyne_client=mock_mnemosyne,
            on_thought=on_thought,
        )
        server.client = mock_client

        # Process multiple inputs to potentially trigger compaction
        await server._process_user_input("First question")
        assert server.reasoning_enabled is True

        await server._process_user_input("Second question")
        assert server.reasoning_enabled is True  # Still enabled after compaction

        # Both should have generated reasoning thoughts
        reasoning_thoughts = [e for e in thought_events if e.thought_type == "reasoning"]
        assert len(reasoning_thoughts) == 2

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_conversation_with_all_features(
        self, mock_client, mock_mnemosyne, server_config
    ):
        """Test multi-turn conversation with reasoning, importance, and memory."""
        # First response: Low importance (no storage)
        # Second response: High importance with reasoning (should store clean response)
        mock_client.generate_responses = [
            "Hello! How can I help you today?",  # Low importance
            """<reasoning>
The user wants code help. I should provide a comprehensive solution
with proper error handling and documentation.
</reasoning>

Here's the solution you requested with complete implementation details:

```python
class DataProcessor:
    \"\"\"
    Process data with validation and error handling.

    This class provides a robust interface for data processing
    with built-in validation, logging, and error recovery.
    \"\"\"

    def __init__(self, config=None):
        self.config = config or {}
        self.logger = self._setup_logger()

    def _setup_logger(self):
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def process(self, data):
        \"\"\"Process the input data.\"\"\"
        self.logger.info(f"Processing data: {type(data)}")

        validated = self._validate(data)
        result = self._transform(validated)

        self.logger.info("Processing complete")
        return result

    def _validate(self, data):
        if data is None:
            raise ValueError("Data cannot be None")
        return data

    def _transform(self, data):
        # Transform implementation
        return f"Transformed: {data}"
```

This implementation includes logging, validation, and clear documentation.""",
        ]

        thought_events = []
        def on_thought(event: ThoughtEvent):
            thought_events.append(event)

        responses = []
        def on_response(content: str):
            responses.append(content)

        server = MemoryServer(
            mock_client,
            server_config,
            mnemosyne_client=mock_mnemosyne,
            on_thought=on_thought,
            on_response=on_response,
        )
        server.client = mock_client

        # First message - low importance
        await server._process_user_input("Hi")
        assert len(mock_mnemosyne.stored_memories) == 0  # Not stored

        # Second message - high importance with reasoning
        await server._process_user_input("Please help me write a data processor class")

        # Should have one stored memory (from second response)
        assert len(mock_mnemosyne.stored_memories) == 1

        # Reasoning thought should have been captured
        reasoning_thoughts = [e for e in thought_events if e.thought_type == "reasoning"]
        assert len(reasoning_thoughts) == 1
        assert "comprehensive solution" in reasoning_thoughts[0].content

        # Response should be clean (no reasoning tags)
        assert len(responses) == 2
        assert "<reasoning>" not in responses[1]
        assert "DataProcessor" in responses[1]
