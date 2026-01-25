"""Unit tests for hybrid search functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path

from mnemosyne.core.models import EmotionalContext, Memory, MemoryStatus, MemoryType
from mnemosyne.config.settings import RetrievalSettings


# Skip if chromadb not available
pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for ChromaDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def retrieval_settings():
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
def chroma_store(temp_db_dir, retrieval_settings):
    """Create a ChromaMemoryStore for testing."""
    from mnemosyne.storage.chroma_store import ChromaMemoryStore

    return ChromaMemoryStore(
        persist_directory=str(temp_db_dir),
        retrieval_settings=retrieval_settings,
    )


class TestTokenization:
    """Tests for BM25 tokenization."""

    def test_basic_tokenization(self, chroma_store):
        """Test basic word tokenization."""
        tokens = chroma_store._tokenize("Hello World, this is a Test!")
        # Stop words should be removed
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # Stop words should NOT be present
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens

    def test_empty_tokenization(self, chroma_store):
        """Test tokenization of empty string."""
        tokens = chroma_store._tokenize("")
        assert tokens == []

    def test_special_characters(self, chroma_store):
        """Test tokenization ignores special characters."""
        tokens = chroma_store._tokenize("Who is your mother???")
        # "who", "is", "your" are stop words
        assert "mother" in tokens
        assert "who" not in tokens  # Stop word
        assert "is" not in tokens   # Stop word

    def test_tokenize_without_stop_words(self, chroma_store):
        """Test tokenization without stop word removal."""
        tokens = chroma_store._tokenize("Who is your mother", remove_stop_words=False)
        assert "who" in tokens
        assert "is" in tokens
        assert "your" in tokens
        assert "mother" in tokens

    def test_single_char_filtered(self, chroma_store):
        """Test that single character tokens are filtered."""
        tokens = chroma_store._tokenize("I am a programmer")
        # Single char tokens should be filtered
        assert "i" not in tokens
        assert "a" not in tokens
        # "am" is 2 chars and not a stop word, so it's kept
        assert "am" in tokens
        assert "programmer" in tokens


class TestReciprocalRankFusion:
    """Tests for RRF score combination."""

    def test_rrf_basic(self, chroma_store):
        """Test basic RRF combination."""
        vector_results = [("mem1", 0.1), ("mem2", 0.2), ("mem3", 0.3)]
        bm25_results = [("mem2", 10.0), ("mem1", 5.0), ("mem4", 3.0)]

        combined = chroma_store._reciprocal_rank_fusion(
            vector_results, bm25_results,
            k=60, vector_weight=0.5, bm25_weight=0.5
        )

        # mem1 and mem2 should be top since they appear in both
        combined_ids = [mem_id for mem_id, score in combined]
        assert "mem1" in combined_ids[:2]
        assert "mem2" in combined_ids[:2]

    def test_rrf_weights(self, chroma_store):
        """Test RRF respects weights."""
        vector_results = [("vec_only", 0.1)]
        bm25_results = [("bm25_only", 10.0)]

        # With vector_weight=1.0, vector result should win
        combined_vec = chroma_store._reciprocal_rank_fusion(
            vector_results, bm25_results,
            k=60, vector_weight=1.0, bm25_weight=0.0
        )
        assert combined_vec[0][0] == "vec_only"

        # With bm25_weight=1.0, bm25 result should win
        combined_bm25 = chroma_store._reciprocal_rank_fusion(
            vector_results, bm25_results,
            k=60, vector_weight=0.0, bm25_weight=1.0
        )
        assert combined_bm25[0][0] == "bm25_only"

    def test_rrf_empty_inputs(self, chroma_store):
        """Test RRF with empty inputs."""
        combined = chroma_store._reciprocal_rank_fusion([], [], k=60)
        assert combined == []

        combined_one = chroma_store._reciprocal_rank_fusion(
            [("mem1", 0.1)], [],
            k=60, vector_weight=0.5, bm25_weight=0.5
        )
        assert len(combined_one) == 1


class TestEmotionalSimilarity:
    """Tests for emotional similarity computation."""

    def test_identical_emotions(self, chroma_store):
        """Identical emotions should have similarity 1.0."""
        e1 = EmotionalContext(valence=0.5, arousal=0.3, quadrant="calm")
        e2 = EmotionalContext(valence=0.5, arousal=0.3, quadrant="calm")

        sim = chroma_store._emotional_similarity(e1, e2)
        assert sim == pytest.approx(1.0)

    def test_opposite_emotions(self, chroma_store):
        """Opposite corners should have low similarity."""
        e1 = EmotionalContext(valence=1.0, arousal=1.0, quadrant="excited")
        e2 = EmotionalContext(valence=-1.0, arousal=-1.0, quadrant="depleted")

        sim = chroma_store._emotional_similarity(e1, e2)
        assert sim == pytest.approx(0.0, abs=0.01)

    def test_missing_context(self, chroma_store):
        """Missing context should return neutral 0.5."""
        e1 = EmotionalContext(valence=0.5, arousal=0.3, quadrant="calm")

        assert chroma_store._emotional_similarity(e1, None) == 0.5
        assert chroma_store._emotional_similarity(None, e1) == 0.5
        assert chroma_store._emotional_similarity(None, None) == 0.5

    def test_partial_similarity(self, chroma_store):
        """Test intermediate similarity values."""
        e1 = EmotionalContext(valence=0.5, arousal=0.5, quadrant="excited")
        e2 = EmotionalContext(valence=0.0, arousal=0.0, quadrant="neutral")

        sim = chroma_store._emotional_similarity(e1, e2)
        # Distance is sqrt(0.5^2 + 0.5^2) = sqrt(0.5) ~ 0.707
        # Similarity = 1 - 0.707/2.83 ~ 0.75
        assert 0.7 < sim < 0.8

    def test_angular_similarity(self, temp_db_dir):
        """Test angular (cosine) similarity mode."""
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        settings = RetrievalSettings(use_angular_similarity=True)
        store = ChromaMemoryStore(
            persist_directory=str(temp_db_dir / "angular"),
            retrieval_settings=settings,
        )

        # Same direction, different magnitudes should be similar
        e1 = EmotionalContext(valence=0.8, arousal=0.8, quadrant="excited")
        e2 = EmotionalContext(valence=0.4, arousal=0.4, quadrant="excited")

        sim = store._emotional_similarity(e1, e2)
        # Same direction -> high similarity
        assert sim > 0.9


class TestQualityScoring:
    """Tests for quality-weighted scoring."""

    def test_quality_score_factors(self, chroma_store):
        """Test quality score incorporates all factors."""
        # Create a good memory: long content, semantic type, assistant tag, high importance
        good_memory = Memory(
            content="This is a long piece of content that contains important information about the topic at hand. It should receive a higher quality score.",
            memory_type=MemoryType.SEMANTIC,
            tags=["assistant", "compacted"],
            importance_score=0.8,
            created_at=datetime.now(),
        )

        # Create a poor memory: short content, episodic type, user tag, low importance
        poor_memory = Memory(
            content="Short content",
            memory_type=MemoryType.EPISODIC,
            tags=["user", "compacted"],
            importance_score=0.2,
            created_at=datetime.now() - timedelta(days=7),
        )

        good_score = chroma_store._compute_quality_score(good_memory, 0.5)
        poor_score = chroma_store._compute_quality_score(poor_memory, 0.5)

        assert good_score > poor_score

    def test_recency_decay(self, chroma_store):
        """Test recency decay over time."""
        # Recent memory
        recent = Memory(
            content="A" * 100,
            memory_type=MemoryType.EPISODIC,
            importance_score=0.5,
            created_at=datetime.now(),
        )

        # Old memory
        old = Memory(
            content="A" * 100,
            memory_type=MemoryType.EPISODIC,
            importance_score=0.5,
            created_at=datetime.now() - timedelta(days=7),
        )

        recent_score = chroma_store._compute_quality_score(recent, 0.5)
        old_score = chroma_store._compute_quality_score(old, 0.5)

        assert recent_score > old_score

    def test_length_factor(self, chroma_store):
        """Test content length affects score."""
        short = Memory(content="Short", memory_type=MemoryType.EPISODIC)
        long = Memory(content="A" * 500, memory_type=MemoryType.EPISODIC)

        short_score = chroma_store._compute_quality_score(short, 0.5)
        long_score = chroma_store._compute_quality_score(long, 0.5)

        assert long_score > short_score

    def test_quality_score_bounded(self, chroma_store):
        """Test that quality scores are bounded to [0, 1]."""
        # Create a memory with maximum factors
        memory = Memory(
            content="A" * 1000,  # Very long
            memory_type=MemoryType.SEMANTIC,  # Semantic type factor
            tags=["assistant"],  # Assistant role factor
            importance_score=1.0,  # Max importance
            created_at=datetime.now(),  # Most recent
        )

        score = chroma_store._compute_quality_score(memory, 1.0)  # Max base score

        # Score should be clamped to [0, 1]
        assert 0.0 <= score <= 1.0


class TestStorageFiltering:
    """Tests for storage-side message filtering."""

    def test_short_messages_filtered(self):
        """Messages under MIN_MEMORY_LENGTH should be filtered."""
        from psyche.core.memory_handler import MemoryHandler, MIN_MEMORY_LENGTH
        from psyche.memory.compaction import Message

        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=MagicMock(),
        )

        short_msg = Message(role="assistant", content="Hi")
        should_store, _ = handler._should_store_message(short_msg)
        assert not should_store

        long_msg = Message(role="assistant", content="A" * (MIN_MEMORY_LENGTH + 1))
        should_store, _ = handler._should_store_message(long_msg)
        assert should_store

    def test_user_questions_filtered(self):
        """User questions should be filtered."""
        from psyche.core.memory_handler import MemoryHandler
        from psyche.memory.compaction import Message

        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=MagicMock(),
        )

        # Simple question with ?
        question1 = Message(role="user", content="What is the meaning of life? I really want to know.")
        should_store, _ = handler._should_store_message(question1)
        assert not should_store

        # Question with preamble ("So what...")
        question2 = Message(role="user", content="So what do you think about this approach?")
        should_store, _ = handler._should_store_message(question2)
        assert not should_store

        # Question word not at start ("Hmm, what is...")
        question3 = Message(role="user", content="Hmm, what is the best way to do this?")
        should_store, _ = handler._should_store_message(question3)
        assert not should_store

    def test_assistant_declarative_is_semantic(self):
        """Assistant declarative statements should be marked as semantic."""
        from psyche.core.memory_handler import MemoryHandler
        from psyche.memory.compaction import Message

        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=MagicMock(),
        )

        # Declarative statement with knowledge indicator
        declarative = Message(
            role="assistant",
            content="I am Claude, an AI assistant made by Anthropic. I can help with many tasks."
        )
        should_store, memory_type = handler._should_store_message(declarative)
        assert should_store
        assert memory_type == "semantic"

    def test_user_statements_stored(self):
        """Non-question user statements should be stored as episodic."""
        from psyche.core.memory_handler import MemoryHandler
        from psyche.memory.compaction import Message

        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=MagicMock(),
        )

        # Statement (no question)
        statement = Message(
            role="user",
            content="I want to implement a feature that allows users to search through their memories."
        )
        should_store, memory_type = handler._should_store_message(statement)
        assert should_store
        assert memory_type == "episodic"

    def test_long_question_with_context_stored(self):
        """Long questions with context should be stored."""
        from psyche.core.memory_handler import MemoryHandler, MAX_QUESTION_LENGTH
        from psyche.memory.compaction import Message

        handler = MemoryHandler(
            mnemosyne_client=None,
            elpis_client=MagicMock(),
        )

        # Long question with context (exceeds MAX_QUESTION_LENGTH)
        long_question = Message(
            role="user",
            content="A" * (MAX_QUESTION_LENGTH + 50) + " How do I fix this?"
        )
        should_store, _ = handler._should_store_message(long_question)
        assert should_store


class TestBM25Integration:
    """Test BM25 with real store."""

    def test_bm25_lazy_rebuild(self, chroma_store):
        """Test that BM25 index is rebuilt lazily."""
        # Initially dirty (no index built yet)
        assert chroma_store._bm25_dirty is True

        # Add a memory
        mem = Memory(
            id="mem1",
            content="Test memory about Python programming",
            memory_type=MemoryType.EPISODIC,
        )
        chroma_store.add_memory(mem)

        # Still dirty after add
        assert chroma_store._bm25_dirty is True

        # Search triggers rebuild
        results = chroma_store.search_memories_hybrid("Python", n_results=1)

        # Now clean
        assert chroma_store._bm25_dirty is False

        # Add another memory - should set dirty again
        mem2 = Memory(
            id="mem2",
            content="Another memory about testing",
            memory_type=MemoryType.EPISODIC,
        )
        chroma_store.add_memory(mem2)
        assert chroma_store._bm25_dirty is True

    def test_bm25_search_returns_results(self, chroma_store):
        """Test that BM25 search actually works."""
        # Add memories with distinct keywords
        mem1 = Memory(
            id="python-mem",
            content="Python is a programming language with clear syntax",
            memory_type=MemoryType.SEMANTIC,
        )
        mem2 = Memory(
            id="rust-mem",
            content="Rust is a systems programming language focused on safety",
            memory_type=MemoryType.SEMANTIC,
        )
        chroma_store.add_memory(mem1)
        chroma_store.add_memory(mem2)

        # Search for Python-specific terms
        bm25_results = chroma_store._bm25_search("Python syntax", n_results=2)

        # Should find results
        assert len(bm25_results) > 0
        # Python memory should score higher
        result_ids = [r[0] for r in bm25_results]
        assert "python-mem" in result_ids
