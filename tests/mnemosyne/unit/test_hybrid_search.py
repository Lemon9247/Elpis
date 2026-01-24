"""Unit tests for hybrid search functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from mnemosyne.core.models import EmotionalContext, Memory, MemoryStatus, MemoryType


class TestTokenization:
    """Tests for BM25 tokenization."""

    def test_basic_tokenization(self):
        """Test basic word tokenization."""
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        # Create a minimal mock to access _tokenize
        with patch.object(ChromaMemoryStore, '__init__', lambda x: None):
            store = ChromaMemoryStore()
            store._tokenize = lambda text: __import__('re').findall(r'\w+', text.lower())

            tokens = store._tokenize("Hello World, this is a Test!")
            assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_empty_tokenization(self):
        """Test tokenization of empty string."""
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        with patch.object(ChromaMemoryStore, '__init__', lambda x: None):
            store = ChromaMemoryStore()
            store._tokenize = lambda text: __import__('re').findall(r'\w+', text.lower())

            tokens = store._tokenize("")
            assert tokens == []

    def test_special_characters(self):
        """Test tokenization ignores special characters."""
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        with patch.object(ChromaMemoryStore, '__init__', lambda x: None):
            store = ChromaMemoryStore()
            store._tokenize = lambda text: __import__('re').findall(r'\w+', text.lower())

            tokens = store._tokenize("Who is your mother???")
            assert tokens == ["who", "is", "your", "mother"]


class TestReciprocalRankFusion:
    """Tests for RRF score combination."""

    def test_rrf_basic(self):
        """Test basic RRF combination."""
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        with patch.object(ChromaMemoryStore, '__init__', lambda x: None):
            store = ChromaMemoryStore()
            store._reciprocal_rank_fusion = ChromaMemoryStore._reciprocal_rank_fusion.__get__(store)

            vector_results = [("mem1", 0.1), ("mem2", 0.2), ("mem3", 0.3)]
            bm25_results = [("mem2", 10.0), ("mem1", 5.0), ("mem4", 3.0)]

            combined = store._reciprocal_rank_fusion(
                vector_results, bm25_results,
                k=60, vector_weight=0.5, bm25_weight=0.5
            )

            # mem1 and mem2 should be top since they appear in both
            combined_ids = [mem_id for mem_id, score in combined]
            assert "mem1" in combined_ids[:2]
            assert "mem2" in combined_ids[:2]

    def test_rrf_weights(self):
        """Test RRF respects weights."""
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        with patch.object(ChromaMemoryStore, '__init__', lambda x: None):
            store = ChromaMemoryStore()
            store._reciprocal_rank_fusion = ChromaMemoryStore._reciprocal_rank_fusion.__get__(store)

            vector_results = [("vec_only", 0.1)]
            bm25_results = [("bm25_only", 10.0)]

            # With vector_weight=1.0, vector result should win
            combined_vec = store._reciprocal_rank_fusion(
                vector_results, bm25_results,
                k=60, vector_weight=1.0, bm25_weight=0.0
            )
            assert combined_vec[0][0] == "vec_only"

            # With bm25_weight=1.0, bm25 result should win
            combined_bm25 = store._reciprocal_rank_fusion(
                vector_results, bm25_results,
                k=60, vector_weight=0.0, bm25_weight=1.0
            )
            assert combined_bm25[0][0] == "bm25_only"

    def test_rrf_empty_inputs(self):
        """Test RRF with empty inputs."""
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        with patch.object(ChromaMemoryStore, '__init__', lambda x: None):
            store = ChromaMemoryStore()
            store._reciprocal_rank_fusion = ChromaMemoryStore._reciprocal_rank_fusion.__get__(store)

            combined = store._reciprocal_rank_fusion([], [], k=60)
            assert combined == []

            combined_one = store._reciprocal_rank_fusion(
                [("mem1", 0.1)], [],
                k=60, vector_weight=0.5, bm25_weight=0.5
            )
            assert len(combined_one) == 1


class TestEmotionalSimilarity:
    """Tests for emotional similarity computation."""

    def test_identical_emotions(self):
        """Identical emotions should have similarity 1.0."""
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        with patch.object(ChromaMemoryStore, '__init__', lambda x: None):
            store = ChromaMemoryStore()
            store._emotional_similarity = ChromaMemoryStore._emotional_similarity.__get__(store)

            e1 = EmotionalContext(valence=0.5, arousal=0.3, quadrant="calm")
            e2 = EmotionalContext(valence=0.5, arousal=0.3, quadrant="calm")

            sim = store._emotional_similarity(e1, e2)
            assert sim == pytest.approx(1.0)

    def test_opposite_emotions(self):
        """Opposite corners should have low similarity."""
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        with patch.object(ChromaMemoryStore, '__init__', lambda x: None):
            store = ChromaMemoryStore()
            store._emotional_similarity = ChromaMemoryStore._emotional_similarity.__get__(store)

            e1 = EmotionalContext(valence=1.0, arousal=1.0, quadrant="excited")
            e2 = EmotionalContext(valence=-1.0, arousal=-1.0, quadrant="depleted")

            sim = store._emotional_similarity(e1, e2)
            assert sim == pytest.approx(0.0, abs=0.01)

    def test_missing_context(self):
        """Missing context should return neutral 0.5."""
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        with patch.object(ChromaMemoryStore, '__init__', lambda x: None):
            store = ChromaMemoryStore()
            store._emotional_similarity = ChromaMemoryStore._emotional_similarity.__get__(store)

            e1 = EmotionalContext(valence=0.5, arousal=0.3, quadrant="calm")

            assert store._emotional_similarity(e1, None) == 0.5
            assert store._emotional_similarity(None, e1) == 0.5
            assert store._emotional_similarity(None, None) == 0.5

    def test_partial_similarity(self):
        """Test intermediate similarity values."""
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        with patch.object(ChromaMemoryStore, '__init__', lambda x: None):
            store = ChromaMemoryStore()
            store._emotional_similarity = ChromaMemoryStore._emotional_similarity.__get__(store)

            e1 = EmotionalContext(valence=0.5, arousal=0.5, quadrant="excited")
            e2 = EmotionalContext(valence=0.0, arousal=0.0, quadrant="neutral")

            sim = store._emotional_similarity(e1, e2)
            # Distance is sqrt(0.5^2 + 0.5^2) = sqrt(0.5) ≈ 0.707
            # Similarity = 1 - 0.707/2.83 ≈ 0.75
            assert 0.7 < sim < 0.8


class TestQualityScoring:
    """Tests for quality-weighted scoring."""

    def test_quality_score_factors(self):
        """Test quality score incorporates all factors."""
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        with patch.object(ChromaMemoryStore, '__init__', lambda x: None):
            store = ChromaMemoryStore()
            store._compute_quality_score = ChromaMemoryStore._compute_quality_score.__get__(store)

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

            good_score = store._compute_quality_score(good_memory, 0.5)
            poor_score = store._compute_quality_score(poor_memory, 0.5)

            assert good_score > poor_score

    def test_recency_decay(self):
        """Test recency decay over time."""
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        with patch.object(ChromaMemoryStore, '__init__', lambda x: None):
            store = ChromaMemoryStore()
            store._compute_quality_score = ChromaMemoryStore._compute_quality_score.__get__(store)

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

            recent_score = store._compute_quality_score(recent, 0.5)
            old_score = store._compute_quality_score(old, 0.5)

            assert recent_score > old_score

    def test_length_factor(self):
        """Test content length affects score."""
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        with patch.object(ChromaMemoryStore, '__init__', lambda x: None):
            store = ChromaMemoryStore()
            store._compute_quality_score = ChromaMemoryStore._compute_quality_score.__get__(store)

            short = Memory(content="Short", memory_type=MemoryType.EPISODIC)
            long = Memory(content="A" * 500, memory_type=MemoryType.EPISODIC)

            short_score = store._compute_quality_score(short, 0.5)
            long_score = store._compute_quality_score(long, 0.5)

            assert long_score > short_score


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

        # Question with ?
        question1 = Message(role="user", content="What is the meaning of life? I really want to know.")
        should_store, _ = handler._should_store_message(question1)
        assert not should_store

        # Question starting with question word
        question2 = Message(role="user", content="How do I implement this feature in Python code")
        should_store, _ = handler._should_store_message(question2)
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
