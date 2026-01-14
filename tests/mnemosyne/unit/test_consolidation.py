"""Unit tests for memory consolidation feature.

Tests for:
- ChromaMemoryStore consolidation-related methods
- MemoryConsolidator class behavior
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict
import json

from mnemosyne.core.models import (
    Memory,
    MemoryType,
    MemoryStatus,
    ConsolidationConfig,
    ConsolidationReport,
    MemoryCluster,
    EmotionalContext,
)
from mnemosyne.core.consolidator import MemoryConsolidator, cosine_similarity


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    return Memory(
        content="Test memory content about important events",
        memory_type=MemoryType.EPISODIC,
        status=MemoryStatus.SHORT_TERM,
        importance_score=0.7,
    )


@pytest.fixture
def sample_memory_with_emotion():
    """Create a memory with emotional context."""
    return Memory(
        content="An emotionally significant event happened today",
        memory_type=MemoryType.EMOTIONAL,
        status=MemoryStatus.SHORT_TERM,
        importance_score=0.8,
        emotional_context=EmotionalContext(
            valence=0.6,
            arousal=0.7,
            quadrant="excited",
        ),
    )


@pytest.fixture
def old_memory():
    """Create a memory that's old enough for consolidation."""
    memory = Memory(
        content="Old memory from the past",
        memory_type=MemoryType.SEMANTIC,
        status=MemoryStatus.SHORT_TERM,
        importance_score=0.6,
    )
    # Set created_at to 2 hours ago
    memory.created_at = datetime.now() - timedelta(hours=2)
    return memory


@pytest.fixture
def recent_memory():
    """Create a memory that's too recent for consolidation."""
    return Memory(
        content="Recent memory just created",
        memory_type=MemoryType.EPISODIC,
        status=MemoryStatus.SHORT_TERM,
        importance_score=0.5,
        created_at=datetime.now(),
    )


@pytest.fixture
def mock_store():
    """Create a mock ChromaMemoryStore for testing."""
    store = Mock()
    store.get_short_term_count.return_value = 0
    store.get_all_short_term.return_value = []
    store.get_embeddings_batch.return_value = {}
    store.promote_memory.return_value = True
    store.delete_memory.return_value = True
    return store


@pytest.fixture
def consolidation_config():
    """Create a test consolidation config."""
    return ConsolidationConfig(
        buffer_threshold=5,
        min_age_hours=0,  # No age requirement for testing
        importance_threshold=0.5,
        max_batch_size=10,
        similarity_threshold=0.85,
    )


@pytest.fixture
def consolidator(mock_store, consolidation_config):
    """Create a consolidator with mock store and test config."""
    return MemoryConsolidator(mock_store, consolidation_config)


# =============================================================================
# Tests for cosine_similarity helper function
# =============================================================================


class TestCosineSimilarity:
    """Tests for the cosine_similarity helper function."""

    def test_identical_vectors_returns_one(self):
        """Identical vectors should have similarity of 1.0."""
        import numpy as np
        vec = np.array([1.0, 2.0, 3.0])
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 0.0001

    def test_orthogonal_vectors_returns_zero(self):
        """Orthogonal vectors should have similarity of 0.0."""
        import numpy as np
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        similarity = cosine_similarity(vec_a, vec_b)
        assert abs(similarity) < 0.0001

    def test_opposite_vectors_returns_negative_one(self):
        """Opposite vectors should have similarity of -1.0."""
        import numpy as np
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_b = np.array([-1.0, -2.0, -3.0])
        similarity = cosine_similarity(vec_a, vec_b)
        assert abs(similarity - (-1.0)) < 0.0001

    def test_zero_vector_returns_zero(self):
        """Zero vector should return similarity of 0.0."""
        import numpy as np
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_zero = np.array([0.0, 0.0, 0.0])
        similarity = cosine_similarity(vec_a, vec_zero)
        assert similarity == 0.0

    def test_both_zero_vectors_returns_zero(self):
        """Two zero vectors should return similarity of 0.0."""
        import numpy as np
        vec_zero = np.array([0.0, 0.0, 0.0])
        similarity = cosine_similarity(vec_zero, vec_zero)
        assert similarity == 0.0


# =============================================================================
# Tests for MemoryConsolidator.should_consolidate
# =============================================================================


class TestShouldConsolidate:
    """Tests for MemoryConsolidator.should_consolidate method."""

    def test_should_consolidate_returns_true_when_buffer_exceeds_threshold(
        self, mock_store, consolidation_config
    ):
        """Should return True when short-term buffer exceeds threshold."""
        mock_store.get_short_term_count.return_value = 10  # > threshold of 5
        consolidator = MemoryConsolidator(mock_store, consolidation_config)

        should, reason = consolidator.should_consolidate()

        assert should is True
        assert "exceeds threshold" in reason

    def test_should_consolidate_returns_false_when_buffer_small(
        self, mock_store, consolidation_config
    ):
        """Should return False when buffer is below threshold."""
        mock_store.get_short_term_count.return_value = 2  # < threshold of 5
        consolidator = MemoryConsolidator(mock_store, consolidation_config)

        should, reason = consolidator.should_consolidate()

        assert should is False
        assert "below threshold" in reason

    def test_should_consolidate_returns_true_at_exact_threshold(
        self, mock_store, consolidation_config
    ):
        """Should return True when buffer equals threshold."""
        mock_store.get_short_term_count.return_value = 5  # == threshold
        consolidator = MemoryConsolidator(mock_store, consolidation_config)

        should, reason = consolidator.should_consolidate()

        assert should is True

    def test_should_consolidate_returns_false_when_empty(
        self, mock_store, consolidation_config
    ):
        """Should return False when buffer is empty."""
        mock_store.get_short_term_count.return_value = 0
        consolidator = MemoryConsolidator(mock_store, consolidation_config)

        should, reason = consolidator.should_consolidate()

        assert should is False


# =============================================================================
# Tests for MemoryConsolidator.get_consolidation_candidates
# =============================================================================


class TestGetConsolidationCandidates:
    """Tests for MemoryConsolidator.get_consolidation_candidates method."""

    def test_get_consolidation_candidates_filters_by_age(
        self, mock_store, old_memory, recent_memory
    ):
        """Should only return memories older than min_age_hours."""
        config = ConsolidationConfig(
            min_age_hours=1,  # Require at least 1 hour old
            buffer_threshold=5,
        )
        # old_memory is 2 hours old, recent_memory is just created
        mock_store.get_all_short_term.return_value = [old_memory, recent_memory]
        consolidator = MemoryConsolidator(mock_store, config)

        candidates = consolidator.get_consolidation_candidates()

        # Should only include the old memory
        assert len(candidates) == 1
        assert candidates[0].id == old_memory.id

    def test_get_consolidation_candidates_returns_empty_when_no_memories(
        self, mock_store, consolidation_config
    ):
        """Should return empty list when no short-term memories exist."""
        mock_store.get_all_short_term.return_value = []
        consolidator = MemoryConsolidator(mock_store, consolidation_config)

        candidates = consolidator.get_consolidation_candidates()

        assert candidates == []

    def test_get_consolidation_candidates_sorts_by_importance(
        self, mock_store, consolidation_config
    ):
        """Should return candidates sorted by importance (descending).

        Note: get_consolidation_candidates() recomputes importance via compute_importance(),
        which factors in emotional salience, recency, and access frequency.
        We use access_count to create differentiated importance scores.
        """
        # Create memories with different access counts to affect computed importance
        # compute_importance() = salience * 0.4 + recency * 0.3 + access_factor * 0.3
        # With no emotional context: salience = 0.5
        # Recent memories: recency ~ 1.0
        # access_factor = min(1.0, access_count / 10)

        low_importance = Memory(
            content="Low importance memory",
            importance_score=0.3,  # Will be recomputed
            status=MemoryStatus.SHORT_TERM,
            access_count=0,  # access_factor = 0.0
        )
        low_importance.created_at = datetime.now() - timedelta(hours=2)

        high_importance = Memory(
            content="High importance memory",
            importance_score=0.9,  # Will be recomputed
            status=MemoryStatus.SHORT_TERM,
            access_count=10,  # access_factor = 1.0
        )
        high_importance.created_at = datetime.now() - timedelta(hours=2)

        medium_importance = Memory(
            content="Medium importance memory",
            importance_score=0.6,  # Will be recomputed
            status=MemoryStatus.SHORT_TERM,
            access_count=5,  # access_factor = 0.5
        )
        medium_importance.created_at = datetime.now() - timedelta(hours=2)

        mock_store.get_all_short_term.return_value = [
            low_importance,
            high_importance,
            medium_importance,
        ]
        consolidator = MemoryConsolidator(mock_store, consolidation_config)

        candidates = consolidator.get_consolidation_candidates()

        # Should be sorted: high, medium, low by computed importance
        assert len(candidates) == 3
        # Verify order: high (access=10) > medium (access=5) > low (access=0)
        assert candidates[0].access_count == 10
        assert candidates[1].access_count == 5
        assert candidates[2].access_count == 0
        # Also verify computed importance is in descending order
        assert candidates[0].importance_score > candidates[1].importance_score
        assert candidates[1].importance_score > candidates[2].importance_score

    def test_get_consolidation_candidates_respects_max_batch_size(
        self, mock_store
    ):
        """Should limit candidates to max_batch_size.

        Note: get_consolidation_candidates() recomputes importance using compute_importance().
        We use access_count to differentiate importance levels.
        """
        config = ConsolidationConfig(
            max_batch_size=2,
            min_age_hours=0,
            buffer_threshold=5,
        )

        # Create more memories than max_batch_size with different access counts
        memories = []
        for i in range(5):
            mem = Memory(
                content=f"Memory {i}",
                importance_score=0.5 + i * 0.1,  # Will be recomputed
                status=MemoryStatus.SHORT_TERM,
                access_count=i * 2,  # 0, 2, 4, 6, 8
            )
            mem.created_at = datetime.now() - timedelta(hours=2)
            memories.append(mem)

        mock_store.get_all_short_term.return_value = memories
        consolidator = MemoryConsolidator(mock_store, config)

        candidates = consolidator.get_consolidation_candidates()

        # Should be limited to max_batch_size
        assert len(candidates) == 2
        # Should be the highest importance ones (highest access_count)
        # Memory 4 (access=8) and Memory 3 (access=6) should be selected
        assert candidates[0].access_count == 8
        assert candidates[1].access_count == 6


# =============================================================================
# Tests for MemoryConsolidator.cluster_memories
# =============================================================================


class TestClusterMemories:
    """Tests for MemoryConsolidator.cluster_memories method."""

    def test_cluster_memories_returns_empty_for_empty_input(
        self, mock_store, consolidation_config
    ):
        """Should return empty list for empty input."""
        consolidator = MemoryConsolidator(mock_store, consolidation_config)

        clusters = consolidator.cluster_memories([])

        assert clusters == []

    def test_cluster_memories_groups_similar(self, mock_store, consolidation_config):
        """Should group semantically similar memories together."""
        import numpy as np

        # Create two similar memories
        mem1 = Memory(
            id="mem1",
            content="The cat sat on the mat",
            importance_score=0.7,
            status=MemoryStatus.SHORT_TERM,
        )
        mem2 = Memory(
            id="mem2",
            content="A cat is sitting on the mat",  # Similar content
            importance_score=0.6,
            status=MemoryStatus.SHORT_TERM,
        )

        # Mock embeddings that are very similar (cosine similarity > 0.85)
        similar_embedding = [0.5, 0.5, 0.5, 0.5]
        mock_store.get_embeddings_batch.return_value = {
            "mem1": similar_embedding,
            "mem2": [0.51, 0.49, 0.52, 0.48],  # Very similar
        }

        consolidator = MemoryConsolidator(mock_store, consolidation_config)
        clusters = consolidator.cluster_memories([mem1, mem2])

        # Should be grouped into one cluster
        assert len(clusters) == 1
        assert len(clusters[0].memories) == 2

    def test_cluster_memories_separates_dissimilar(
        self, mock_store, consolidation_config
    ):
        """Should put dissimilar memories in separate clusters."""
        # Create two dissimilar memories
        mem1 = Memory(
            id="mem1",
            content="The cat sat on the mat",
            importance_score=0.7,
            status=MemoryStatus.SHORT_TERM,
        )
        mem2 = Memory(
            id="mem2",
            content="Quantum mechanics explains particle behavior",  # Different topic
            importance_score=0.6,
            status=MemoryStatus.SHORT_TERM,
        )

        # Mock embeddings that are very different
        mock_store.get_embeddings_batch.return_value = {
            "mem1": [1.0, 0.0, 0.0, 0.0],
            "mem2": [0.0, 1.0, 0.0, 0.0],  # Orthogonal = dissimilar
        }

        consolidator = MemoryConsolidator(mock_store, consolidation_config)
        clusters = consolidator.cluster_memories([mem1, mem2])

        # Should be in separate clusters
        assert len(clusters) == 2
        assert len(clusters[0].memories) == 1
        assert len(clusters[1].memories) == 1

    def test_cluster_memories_handles_missing_embeddings(
        self, mock_store, consolidation_config
    ):
        """Should handle memories without embeddings gracefully."""
        mem1 = Memory(
            id="mem1",
            content="Memory with embedding",
            importance_score=0.7,
            status=MemoryStatus.SHORT_TERM,
        )
        mem2 = Memory(
            id="mem2",
            content="Memory without embedding",
            importance_score=0.6,
            status=MemoryStatus.SHORT_TERM,
        )

        # Only provide embedding for mem1
        mock_store.get_embeddings_batch.return_value = {
            "mem1": [0.5, 0.5, 0.5, 0.5],
            # mem2 has no embedding
        }

        consolidator = MemoryConsolidator(mock_store, consolidation_config)
        clusters = consolidator.cluster_memories([mem1, mem2])

        # Should only include mem1 in cluster
        assert len(clusters) == 1
        assert clusters[0].memories[0].id == "mem1"

    def test_cluster_memories_returns_singleton_clusters_when_no_embeddings(
        self, mock_store, consolidation_config
    ):
        """Should return singleton clusters when no embeddings are available."""
        mem1 = Memory(
            id="mem1",
            content="Memory 1",
            importance_score=0.7,
            memory_type=MemoryType.EPISODIC,
            status=MemoryStatus.SHORT_TERM,
        )
        mem2 = Memory(
            id="mem2",
            content="Memory 2",
            importance_score=0.6,
            memory_type=MemoryType.SEMANTIC,
            status=MemoryStatus.SHORT_TERM,
        )

        # No embeddings available
        mock_store.get_embeddings_batch.return_value = {}

        consolidator = MemoryConsolidator(mock_store, consolidation_config)
        clusters = consolidator.cluster_memories([mem1, mem2])

        # Each memory should be in its own singleton cluster
        assert len(clusters) == 2
        for cluster in clusters:
            assert len(cluster.memories) == 1

    def test_cluster_memories_calculates_avg_importance(
        self, mock_store, consolidation_config
    ):
        """Should calculate average importance for clusters."""
        mem1 = Memory(
            id="mem1",
            content="Memory 1",
            importance_score=0.8,
            status=MemoryStatus.SHORT_TERM,
        )
        mem2 = Memory(
            id="mem2",
            content="Memory 2",
            importance_score=0.6,
            status=MemoryStatus.SHORT_TERM,
        )

        # Mock similar embeddings so they cluster together
        mock_store.get_embeddings_batch.return_value = {
            "mem1": [0.5, 0.5, 0.5, 0.5],
            "mem2": [0.51, 0.49, 0.52, 0.48],
        }

        consolidator = MemoryConsolidator(mock_store, consolidation_config)
        clusters = consolidator.cluster_memories([mem1, mem2])

        # Average importance should be (0.8 + 0.6) / 2 = 0.7
        assert len(clusters) == 1
        assert abs(clusters[0].avg_importance - 0.7) < 0.01

    def test_cluster_memories_determines_dominant_type(
        self, mock_store, consolidation_config
    ):
        """Should determine the dominant memory type in cluster."""
        mem1 = Memory(
            id="mem1",
            content="Memory 1",
            importance_score=0.7,
            memory_type=MemoryType.EPISODIC,
            status=MemoryStatus.SHORT_TERM,
        )
        mem2 = Memory(
            id="mem2",
            content="Memory 2",
            importance_score=0.7,
            memory_type=MemoryType.EPISODIC,
            status=MemoryStatus.SHORT_TERM,
        )
        mem3 = Memory(
            id="mem3",
            content="Memory 3",
            importance_score=0.7,
            memory_type=MemoryType.SEMANTIC,
            status=MemoryStatus.SHORT_TERM,
        )

        # All cluster together
        mock_store.get_embeddings_batch.return_value = {
            "mem1": [0.5, 0.5, 0.5, 0.5],
            "mem2": [0.51, 0.49, 0.52, 0.48],
            "mem3": [0.52, 0.48, 0.51, 0.49],
        }

        consolidator = MemoryConsolidator(mock_store, consolidation_config)
        clusters = consolidator.cluster_memories([mem1, mem2, mem3])

        # Dominant type should be EPISODIC (2 vs 1)
        assert clusters[0].dominant_type == MemoryType.EPISODIC


# =============================================================================
# Tests for MemoryConsolidator.consolidate
# =============================================================================


class TestConsolidate:
    """Tests for MemoryConsolidator.consolidate method."""

    def test_consolidate_returns_empty_report_when_no_candidates(
        self, mock_store, consolidation_config
    ):
        """Should return empty report when no consolidation candidates."""
        mock_store.get_all_short_term.return_value = []
        consolidator = MemoryConsolidator(mock_store, consolidation_config)

        report = consolidator.consolidate()

        assert isinstance(report, ConsolidationReport)
        assert report.clusters_formed == 0
        assert report.memories_promoted == 0
        assert report.memories_archived == 0
        assert report.total_processed == 0

    def test_consolidate_promotes_high_importance_clusters(
        self, mock_store, consolidation_config
    ):
        """Should promote representative memory from high importance cluster."""
        mem1 = Memory(
            id="mem1",
            content="Important memory",
            importance_score=0.9,  # High importance
            status=MemoryStatus.SHORT_TERM,
        )
        mem1.created_at = datetime.now() - timedelta(hours=2)

        mock_store.get_all_short_term.return_value = [mem1]
        mock_store.get_embeddings_batch.return_value = {
            "mem1": [0.5, 0.5, 0.5, 0.5],
        }
        mock_store.promote_memory.return_value = True

        consolidator = MemoryConsolidator(mock_store, consolidation_config)
        report = consolidator.consolidate()

        assert report.memories_promoted == 1
        mock_store.promote_memory.assert_called_once_with("mem1")

    def test_consolidate_skips_low_importance_clusters(
        self, mock_store
    ):
        """Should skip clusters with importance below threshold."""
        config = ConsolidationConfig(
            importance_threshold=0.8,  # High threshold
            min_age_hours=0,
            buffer_threshold=5,
        )

        mem1 = Memory(
            id="mem1",
            content="Low importance memory",
            importance_score=0.3,  # Below threshold
            status=MemoryStatus.SHORT_TERM,
        )
        mem1.created_at = datetime.now() - timedelta(hours=2)

        mock_store.get_all_short_term.return_value = [mem1]
        mock_store.get_embeddings_batch.return_value = {
            "mem1": [0.5, 0.5, 0.5, 0.5],
        }

        consolidator = MemoryConsolidator(mock_store, config)
        report = consolidator.consolidate()

        assert report.memories_promoted == 0
        assert report.memories_skipped == 1
        mock_store.promote_memory.assert_not_called()

    def test_consolidate_archives_cluster_members(
        self, mock_store, consolidation_config
    ):
        """Should delete other cluster members after promoting representative."""
        # Create a cluster with 3 memories
        mem1 = Memory(
            id="mem1",
            content="Memory 1",
            importance_score=0.9,  # Highest - will be representative
            status=MemoryStatus.SHORT_TERM,
        )
        mem1.created_at = datetime.now() - timedelta(hours=2)

        mem2 = Memory(
            id="mem2",
            content="Memory 2",
            importance_score=0.7,
            status=MemoryStatus.SHORT_TERM,
        )
        mem2.created_at = datetime.now() - timedelta(hours=2)

        mem3 = Memory(
            id="mem3",
            content="Memory 3",
            importance_score=0.6,
            status=MemoryStatus.SHORT_TERM,
        )
        mem3.created_at = datetime.now() - timedelta(hours=2)

        mock_store.get_all_short_term.return_value = [mem1, mem2, mem3]
        # All cluster together
        mock_store.get_embeddings_batch.return_value = {
            "mem1": [0.5, 0.5, 0.5, 0.5],
            "mem2": [0.51, 0.49, 0.52, 0.48],
            "mem3": [0.52, 0.48, 0.51, 0.49],
        }
        mock_store.promote_memory.return_value = True
        mock_store.delete_memory.return_value = True

        consolidator = MemoryConsolidator(mock_store, consolidation_config)
        report = consolidator.consolidate()

        # mem1 promoted, mem2 and mem3 archived
        assert report.memories_promoted == 1
        assert report.memories_archived == 2
        mock_store.promote_memory.assert_called_once_with("mem1")
        # Delete should be called for mem2 and mem3
        delete_calls = mock_store.delete_memory.call_args_list
        deleted_ids = [call[0][0] for call in delete_calls]
        assert "mem2" in deleted_ids
        assert "mem3" in deleted_ids

    def test_consolidate_report_includes_cluster_summaries(
        self, mock_store, consolidation_config
    ):
        """Should include cluster summaries in the report."""
        mem1 = Memory(
            id="mem1",
            content="Memory 1",
            importance_score=0.8,
            status=MemoryStatus.SHORT_TERM,
        )
        mem1.created_at = datetime.now() - timedelta(hours=2)

        mem2 = Memory(
            id="mem2",
            content="Memory 2",
            importance_score=0.6,
            status=MemoryStatus.SHORT_TERM,
        )
        mem2.created_at = datetime.now() - timedelta(hours=2)

        mock_store.get_all_short_term.return_value = [mem1, mem2]
        mock_store.get_embeddings_batch.return_value = {
            "mem1": [0.5, 0.5, 0.5, 0.5],
            "mem2": [0.51, 0.49, 0.52, 0.48],
        }
        mock_store.promote_memory.return_value = True
        mock_store.delete_memory.return_value = True

        consolidator = MemoryConsolidator(mock_store, consolidation_config)
        report = consolidator.consolidate()

        # Should have cluster summary
        assert len(report.cluster_summaries) == 1
        summary = report.cluster_summaries[0]
        assert summary["promoted_id"] == "mem1"
        assert "mem2" in summary["source_ids"]
        assert summary["cluster_size"] == 2

    def test_consolidate_handles_promote_failure(
        self, mock_store, consolidation_config
    ):
        """Should handle promotion failure gracefully."""
        mem1 = Memory(
            id="mem1",
            content="Memory 1",
            importance_score=0.8,
            status=MemoryStatus.SHORT_TERM,
        )
        mem1.created_at = datetime.now() - timedelta(hours=2)

        mock_store.get_all_short_term.return_value = [mem1]
        mock_store.get_embeddings_batch.return_value = {
            "mem1": [0.5, 0.5, 0.5, 0.5],
        }
        mock_store.promote_memory.return_value = False  # Promotion fails

        consolidator = MemoryConsolidator(mock_store, consolidation_config)
        report = consolidator.consolidate()

        # Should be skipped, not promoted
        assert report.memories_promoted == 0
        assert report.memories_skipped == 1

    def test_consolidate_records_duration(
        self, mock_store, consolidation_config
    ):
        """Should record consolidation duration."""
        mock_store.get_all_short_term.return_value = []
        consolidator = MemoryConsolidator(mock_store, consolidation_config)

        report = consolidator.consolidate()

        # Should have some duration recorded
        assert report.duration_seconds >= 0


# =============================================================================
# Tests for ConsolidationConfig
# =============================================================================


class TestConsolidationConfig:
    """Tests for ConsolidationConfig defaults and behavior."""

    def test_default_values(self):
        """Should have sensible default values."""
        config = ConsolidationConfig()

        assert config.importance_threshold == 0.6
        assert config.min_age_hours == 1
        assert config.max_batch_size == 50
        assert config.buffer_threshold == 100
        assert config.similarity_threshold == 0.85
        assert config.min_cluster_size == 2

    def test_custom_values(self):
        """Should accept custom values."""
        config = ConsolidationConfig(
            importance_threshold=0.8,
            min_age_hours=2,
            max_batch_size=25,
            buffer_threshold=50,
            similarity_threshold=0.9,
            min_cluster_size=3,
        )

        assert config.importance_threshold == 0.8
        assert config.min_age_hours == 2
        assert config.max_batch_size == 25
        assert config.buffer_threshold == 50
        assert config.similarity_threshold == 0.9
        assert config.min_cluster_size == 3


# =============================================================================
# Tests for ConsolidationReport
# =============================================================================


class TestConsolidationReport:
    """Tests for ConsolidationReport."""

    def test_default_values(self):
        """Should have zero default values."""
        report = ConsolidationReport()

        assert report.clusters_formed == 0
        assert report.memories_promoted == 0
        assert report.memories_archived == 0
        assert report.memories_skipped == 0
        assert report.total_processed == 0
        assert report.duration_seconds == 0.0
        assert report.cluster_summaries == []

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        report = ConsolidationReport(
            clusters_formed=3,
            memories_promoted=3,
            memories_archived=7,
            memories_skipped=2,
            total_processed=12,
            duration_seconds=1.5,
            cluster_summaries=[{"promoted_id": "test"}],
        )

        d = report.to_dict()

        assert d["clusters_formed"] == 3
        assert d["memories_promoted"] == 3
        assert d["memories_archived"] == 7
        assert d["memories_skipped"] == 2
        assert d["total_processed"] == 12
        assert d["duration_seconds"] == 1.5
        assert d["cluster_summaries"] == [{"promoted_id": "test"}]


# =============================================================================
# Tests for MemoryCluster
# =============================================================================


class TestMemoryCluster:
    """Tests for MemoryCluster dataclass."""

    def test_default_values(self):
        """Should have empty default values."""
        cluster = MemoryCluster()

        assert cluster.memories == []
        assert cluster.centroid_embedding == []
        assert cluster.avg_importance == 0.0
        assert cluster.dominant_type == MemoryType.EPISODIC

    def test_custom_values(self, sample_memory):
        """Should accept custom values."""
        cluster = MemoryCluster(
            memories=[sample_memory],
            centroid_embedding=[0.1, 0.2, 0.3],
            avg_importance=0.75,
            dominant_type=MemoryType.SEMANTIC,
        )

        assert len(cluster.memories) == 1
        assert cluster.memories[0] == sample_memory
        assert cluster.centroid_embedding == [0.1, 0.2, 0.3]
        assert cluster.avg_importance == 0.75
        assert cluster.dominant_type == MemoryType.SEMANTIC


# =============================================================================
# Integration-style Tests (still using mocks but testing full flow)
# =============================================================================


class TestConsolidationIntegration:
    """Integration-style tests for full consolidation workflow."""

    def test_full_consolidation_workflow(self, mock_store):
        """Test complete consolidation workflow."""
        # Setup: Create a realistic scenario
        config = ConsolidationConfig(
            buffer_threshold=3,
            min_age_hours=0,
            importance_threshold=0.5,
            similarity_threshold=0.85,
        )

        # Create memories representing a conversation
        memories = []
        for i in range(5):
            mem = Memory(
                id=f"mem{i}",
                content=f"Conversation message {i}",
                importance_score=0.6 + i * 0.05,  # 0.6, 0.65, 0.7, 0.75, 0.8
                status=MemoryStatus.SHORT_TERM,
            )
            mem.created_at = datetime.now() - timedelta(hours=2)
            memories.append(mem)

        mock_store.get_short_term_count.return_value = 5
        mock_store.get_all_short_term.return_value = memories

        # All similar embeddings (will cluster together)
        embeddings = {
            f"mem{i}": [0.5 + i * 0.01, 0.5, 0.5, 0.5]
            for i in range(5)
        }
        mock_store.get_embeddings_batch.return_value = embeddings
        mock_store.promote_memory.return_value = True
        mock_store.delete_memory.return_value = True

        consolidator = MemoryConsolidator(mock_store, config)

        # Step 1: Check if consolidation is needed
        should, reason = consolidator.should_consolidate()
        assert should is True

        # Step 2: Run consolidation
        report = consolidator.consolidate()

        # Verify results
        assert report.total_processed == 5
        assert report.clusters_formed >= 1
        assert report.memories_promoted >= 1
        assert report.duration_seconds > 0

    def test_consolidation_preserves_most_important_memory(self, mock_store):
        """Most important memory in cluster should be preserved.

        Note: Importance is recomputed using compute_importance() during consolidation.
        We use access_count to create differentiated importance levels.
        """
        config = ConsolidationConfig(
            min_age_hours=0,
            importance_threshold=0.5,
            similarity_threshold=0.85,
        )

        # Create cluster with varying access counts to affect computed importance
        # compute_importance() = salience * 0.4 + recency * 0.3 + access_factor * 0.3
        low = Memory(
            id="low",
            content="Low importance",
            importance_score=0.5,  # Will be recomputed
            status=MemoryStatus.SHORT_TERM,
            access_count=0,  # access_factor = 0.0
        )
        low.created_at = datetime.now() - timedelta(hours=2)

        medium = Memory(
            id="medium",
            content="Medium importance",
            importance_score=0.7,  # Will be recomputed
            status=MemoryStatus.SHORT_TERM,
            access_count=5,  # access_factor = 0.5
        )
        medium.created_at = datetime.now() - timedelta(hours=2)

        high = Memory(
            id="high",
            content="High importance",
            importance_score=0.9,  # Will be recomputed
            status=MemoryStatus.SHORT_TERM,
            access_count=10,  # access_factor = 1.0
        )
        high.created_at = datetime.now() - timedelta(hours=2)

        mock_store.get_all_short_term.return_value = [low, medium, high]
        mock_store.get_embeddings_batch.return_value = {
            "low": [0.5, 0.5, 0.5, 0.5],
            "medium": [0.51, 0.49, 0.52, 0.48],
            "high": [0.52, 0.48, 0.51, 0.49],
        }
        mock_store.promote_memory.return_value = True
        mock_store.delete_memory.return_value = True

        consolidator = MemoryConsolidator(mock_store, config)
        report = consolidator.consolidate()

        # High importance memory (highest access_count) should be promoted
        mock_store.promote_memory.assert_called_once_with("high")
