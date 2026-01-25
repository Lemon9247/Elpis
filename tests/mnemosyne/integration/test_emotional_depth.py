"""Integration tests for emotional depth features.

Tests the full stack:
- Hybrid search (BM25 + vector + emotional similarity)
- Quality scoring with configurable weights
- Trajectory tracking with spiral detection
- Dynamic dream query generation
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mnemosyne.config.settings import RetrievalSettings
from mnemosyne.core.models import EmotionalContext, Memory, MemoryStatus, MemoryType


# Skip if chromadb not available
pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")


@pytest.fixture
def temp_db_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for ChromaDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def retrieval_settings() -> RetrievalSettings:
    """Create retrieval settings for testing."""
    return RetrievalSettings(
        emotion_weight=0.3,
        vector_weight=0.5,
        bm25_weight=0.5,
        recency_weight=0.3,
        importance_weight=0.2,
        relevance_weight=0.5,
        recency_decay_rate=0.995,
        max_recency_hours=720,
        max_content_length=500,
        semantic_type_factor=1.2,
        assistant_role_factor=1.1,
        user_role_factor=0.9,
        use_angular_similarity=False,
    )


@pytest.fixture
def chroma_store(temp_db_dir: Path, retrieval_settings: RetrievalSettings):
    """Create a ChromaMemoryStore for testing."""
    from mnemosyne.storage.chroma_store import ChromaMemoryStore

    return ChromaMemoryStore(
        persist_directory=str(temp_db_dir),
        retrieval_settings=retrieval_settings,
    )


class TestHybridSearchIntegration:
    """Test hybrid search with real ChromaDB."""

    def test_hybrid_search_combines_vector_and_bm25(self, chroma_store):
        """Test that hybrid search uses both vector and BM25 results."""
        # Add memories with different characteristics
        memories = [
            Memory(
                id="vec-match",
                content="The quick brown fox jumps over the lazy dog",
                summary="Fox jumping",
                memory_type=MemoryType.EPISODIC,
                importance_score=0.5,
            ),
            Memory(
                id="bm25-match",
                content="Python programming language syntax keywords functions",
                summary="Python syntax",
                memory_type=MemoryType.SEMANTIC,
                importance_score=0.7,
            ),
            Memory(
                id="both-match",
                content="Python is a programming language with simple syntax",
                summary="Python language",
                memory_type=MemoryType.SEMANTIC,
                importance_score=0.8,
            ),
        ]

        for memory in memories:
            chroma_store.add_memory(memory)

        # Search for "Python syntax" - should favor both-match
        results = chroma_store.search_memories_hybrid(
            "Python programming syntax",
            n_results=3,
            vector_weight=0.5,
            bm25_weight=0.5,
        )

        assert len(results) == 3
        # The memory matching both vector and BM25 should rank high
        result_ids = [r.id for r in results]
        assert "both-match" in result_ids[:2]

    def test_bm25_lazy_rebuild(self, chroma_store):
        """Test that BM25 index is rebuilt lazily, not on every add."""
        # Add first memory
        mem1 = Memory(
            id="mem1",
            content="First test memory content",
            memory_type=MemoryType.EPISODIC,
        )
        chroma_store.add_memory(mem1)

        # Check dirty flag is set
        assert chroma_store._bm25_dirty is True

        # Search triggers rebuild
        results = chroma_store.search_memories_hybrid("test memory", n_results=1)
        assert len(results) == 1

        # Dirty flag should be cleared after search
        assert chroma_store._bm25_dirty is False

        # Add another memory - should set dirty again
        mem2 = Memory(
            id="mem2",
            content="Second test memory content",
            memory_type=MemoryType.EPISODIC,
        )
        chroma_store.add_memory(mem2)
        assert chroma_store._bm25_dirty is True

    def test_emotional_similarity_euclidean(self, chroma_store):
        """Test mood-congruent retrieval with Euclidean distance."""
        # Add memories with different emotional contexts
        happy_memory = Memory(
            id="happy",
            content="I felt so joyful and excited today",
            memory_type=MemoryType.EPISODIC,
            emotional_context=EmotionalContext(valence=0.8, arousal=0.7, quadrant="excited"),
        )
        sad_memory = Memory(
            id="sad",
            content="I felt so down and tired today",
            memory_type=MemoryType.EPISODIC,
            emotional_context=EmotionalContext(valence=-0.8, arousal=-0.5, quadrant="depleted"),
        )

        chroma_store.add_memory(happy_memory)
        chroma_store.add_memory(sad_memory)

        # Search with happy emotional context
        happy_ctx = EmotionalContext(valence=0.7, arousal=0.6, quadrant="excited")
        results = chroma_store.search_memories_hybrid(
            "how I felt today",
            n_results=2,
            emotional_context=happy_ctx,
            emotion_weight=0.5,
        )

        # Happy memory should rank first due to emotional similarity
        assert results[0].id == "happy"

    def test_emotional_similarity_angular(self, temp_db_dir):
        """Test mood-congruent retrieval with angular (cosine) similarity."""
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        settings = RetrievalSettings(use_angular_similarity=True)
        store = ChromaMemoryStore(
            persist_directory=str(temp_db_dir / "angular"),
            retrieval_settings=settings,
        )

        # Add memories with same direction but different magnitudes
        strong = Memory(
            id="strong",
            content="Feeling intensely happy and energized",
            memory_type=MemoryType.EPISODIC,
            emotional_context=EmotionalContext(valence=0.9, arousal=0.9, quadrant="excited"),
        )
        mild = Memory(
            id="mild",
            content="Feeling mildly happy and calm",
            memory_type=MemoryType.EPISODIC,
            emotional_context=EmotionalContext(valence=0.3, arousal=0.3, quadrant="calm"),
        )

        store.add_memory(strong)
        store.add_memory(mild)

        # With angular similarity, both should be similar (same direction)
        query_ctx = EmotionalContext(valence=0.5, arousal=0.5, quadrant="excited")
        similarity_strong = store._emotional_similarity(query_ctx, strong.emotional_context)
        similarity_mild = store._emotional_similarity(query_ctx, mild.emotional_context)

        # Both should have similar scores since they're in the same direction
        assert abs(similarity_strong - similarity_mild) < 0.2

    def test_quality_scoring_uses_config(self, chroma_store, retrieval_settings):
        """Test that quality scoring uses configurable weights."""
        # Create memories with different characteristics
        recent_memory = Memory(
            id="recent",
            content="This is a recent memory with good content length",
            memory_type=MemoryType.EPISODIC,
            importance_score=0.5,
            created_at=datetime.now(),
            tags=["assistant"],
        )

        old_memory = Memory(
            id="old",
            content="This is an old memory with good content length too",
            memory_type=MemoryType.EPISODIC,
            importance_score=0.5,
            created_at=datetime.now() - timedelta(days=7),
            tags=["user"],
        )

        chroma_store.add_memory(recent_memory)
        chroma_store.add_memory(old_memory)

        # Quality score should favor recent + assistant
        recent_score = chroma_store._compute_quality_score(recent_memory, 1.0)
        old_score = chroma_store._compute_quality_score(old_memory, 1.0)

        assert recent_score > old_score


class TestTrajectoryIntegration:
    """Test trajectory tracking integration."""

    def test_trajectory_config_from_settings(self):
        """Test creating TrajectoryConfig from EmotionSettings."""
        from elpis.config.settings import EmotionSettings
        from elpis.emotion.state import TrajectoryConfig

        settings = EmotionSettings(
            trajectory_history_size=30,
            momentum_positive_threshold=0.02,
            spiral_history_count=7,
        )

        config = TrajectoryConfig.from_settings(settings)

        assert config.history_size == 30
        assert config.momentum_positive_threshold == 0.02
        assert config.spiral_history_count == 7

    def test_spiral_detection_with_direction(self):
        """Test that spiral detection includes direction."""
        from elpis.emotion.state import EmotionalState, TrajectoryConfig

        state = EmotionalState(
            baseline_valence=0.0,
            baseline_arousal=0.0,
        )
        state._trajectory_config = TrajectoryConfig(
            spiral_history_count=5,
            spiral_increasing_threshold=3,
        )

        # Simulate spiraling toward positive valence
        for i in range(6):
            state.valence = 0.1 * i
            state.arousal = 0.05 * i
            state.record_state()

        trajectory = state.get_trajectory()

        assert trajectory.spiral_detected is True
        assert trajectory.spiral_direction == "positive"

    def test_trajectory_trend_detection(self):
        """Test trend detection with configurable thresholds."""
        from datetime import timezone
        from elpis.emotion.state import EmotionalState, TrajectoryConfig

        state = EmotionalState()
        state._trajectory_config = TrajectoryConfig(
            trend_improving_threshold=0.02,
            trend_declining_threshold=-0.02,
        )

        # Simulate improving trend with controlled timestamps
        # (record_state uses real time which is too fast for velocity detection)
        for i in range(6):
            timestamp = datetime.now(timezone.utc) - timedelta(minutes=5-i)
            valence = -0.5 + 0.2 * i  # -0.5 -> 0.5
            state._history.append((timestamp, valence, 0.0))
        state.valence = 0.5

        trajectory = state.get_trajectory()
        assert trajectory.trend == "improving"


class TestDreamIntegration:
    """Test dream handler integration."""

    def test_dynamic_query_generation(self):
        """Test dynamic query generation from trajectory."""
        from psyche.handlers.dream_handler import DreamConfig, DreamHandler

        core = MagicMock()
        # Use higher dynamic_query_count to allow more queries
        handler = DreamHandler(core, DreamConfig(use_dynamic_queries=True, dynamic_query_count=5))

        # Test with positive spiral trajectory
        trajectory = {
            "spiral_direction": "positive",
            "trend": "improving",
            "time_in_quadrant": 100,
        }

        queries = handler._generate_dynamic_queries(trajectory, ["python", "testing"])

        # Should include spiral-direction queries
        assert any("progress" in q or "growth" in q for q in queries)
        # Should include topic-based queries (with higher count limit)
        assert any("python" in q for q in queries)

    def test_dynamic_query_for_negative_spiral(self):
        """Test dynamic queries adapt to negative spirals."""
        from psyche.handlers.dream_handler import DreamConfig, DreamHandler

        core = MagicMock()
        # Use higher dynamic_query_count to allow more queries
        handler = DreamHandler(core, DreamConfig(use_dynamic_queries=True, dynamic_query_count=5))

        trajectory = {
            "spiral_direction": "negative",
            "trend": "declining",
            "time_in_quadrant": 600,  # Long time in quadrant
        }

        queries = handler._generate_dynamic_queries(trajectory, [])

        # Should seek comfort/support for negative spiral
        assert any("comfort" in q or "support" in q or "helps" in q for q in queries)
        # Long time in quadrant -> seek peace
        assert any("peace" in q for q in queries)


class TestStorageFiltering:
    """Test improved storage filtering logic."""

    def test_high_value_patterns_stored(self):
        """Test that high-value content patterns are always stored."""
        from psyche.core.memory_handler import MemoryHandler
        from psyche.memory.compaction import Message

        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=MagicMock(),
        )

        # High-value content with definition pattern
        msg = Message(
            role="assistant",
            content="A closure means a function that captures its environment",
        )

        should_store, memory_type = handler._should_store_message(msg)
        assert should_store is True
        assert memory_type == "semantic"

    def test_low_value_greetings_filtered(self):
        """Test that short greetings are filtered out."""
        from psyche.core.memory_handler import MemoryHandler
        from psyche.memory.compaction import Message

        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=MagicMock(),
        )

        msg = Message(role="user", content="Hello, thanks for helping!")

        should_store, _ = handler._should_store_message(msg)
        assert should_store is False

    def test_long_question_with_context_stored(self):
        """Test that long questions with context are stored."""
        from psyche.core.memory_handler import MemoryHandler
        from psyche.memory.compaction import Message

        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=MagicMock(),
        )

        # Long question with context
        msg = Message(
            role="user",
            content=(
                "I've been working on a machine learning project using PyTorch "
                "and I'm having trouble with the gradient computation in my custom "
                "loss function. The gradients seem to vanish after a few epochs. "
                "How can I debug this issue?"
            ),
        )

        should_store, memory_type = handler._should_store_message(msg)
        assert should_store is True
        assert memory_type == "episodic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
