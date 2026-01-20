"""Tests for shared constants."""

import pytest

from shared.constants import (
    AUTO_STORAGE_THRESHOLD,
    CONSOLIDATION_IMPORTANCE_THRESHOLD,
    CONSOLIDATION_SIMILARITY_THRESHOLD,
    MEMORY_CONTENT_TRUNCATE_LENGTH,
    MEMORY_SUMMARY_LENGTH,
)


class TestSharedConstants:
    """Test that shared constants have expected values."""

    def test_memory_summary_length(self):
        """Test MEMORY_SUMMARY_LENGTH is set correctly."""
        assert MEMORY_SUMMARY_LENGTH == 500
        assert isinstance(MEMORY_SUMMARY_LENGTH, int)

    def test_memory_content_truncate_length(self):
        """Test MEMORY_CONTENT_TRUNCATE_LENGTH is set correctly."""
        assert MEMORY_CONTENT_TRUNCATE_LENGTH == 300
        assert isinstance(MEMORY_CONTENT_TRUNCATE_LENGTH, int)

    def test_auto_storage_threshold(self):
        """Test AUTO_STORAGE_THRESHOLD is in valid range."""
        assert AUTO_STORAGE_THRESHOLD == 0.6
        assert 0.0 <= AUTO_STORAGE_THRESHOLD <= 1.0

    def test_consolidation_importance_threshold(self):
        """Test CONSOLIDATION_IMPORTANCE_THRESHOLD is in valid range."""
        assert CONSOLIDATION_IMPORTANCE_THRESHOLD == 0.6
        assert 0.0 <= CONSOLIDATION_IMPORTANCE_THRESHOLD <= 1.0

    def test_consolidation_similarity_threshold(self):
        """Test CONSOLIDATION_SIMILARITY_THRESHOLD is in valid range."""
        assert CONSOLIDATION_SIMILARITY_THRESHOLD == 0.85
        assert 0.0 <= CONSOLIDATION_SIMILARITY_THRESHOLD <= 1.0

    def test_truncate_lengths_relationship(self):
        """Test that summary length is greater than content truncate length."""
        # This makes sense: summaries can be longer than display snippets
        assert MEMORY_SUMMARY_LENGTH > MEMORY_CONTENT_TRUNCATE_LENGTH
