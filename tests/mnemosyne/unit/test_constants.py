"""Tests for Mnemosyne constants."""

import pytest

from mnemosyne.core.constants import (
    CONSOLIDATION_IMPORTANCE_THRESHOLD,
    CONSOLIDATION_SIMILARITY_THRESHOLD,
    MEMORY_SUMMARY_LENGTH,
)


class TestMnemosyneConstants:
    """Test that Mnemosyne constants have expected values."""

    def test_memory_summary_length(self):
        """Test MEMORY_SUMMARY_LENGTH is set correctly."""
        assert MEMORY_SUMMARY_LENGTH == 500
        assert isinstance(MEMORY_SUMMARY_LENGTH, int)

    def test_consolidation_importance_threshold(self):
        """Test CONSOLIDATION_IMPORTANCE_THRESHOLD is in valid range."""
        assert CONSOLIDATION_IMPORTANCE_THRESHOLD == 0.6
        assert 0.0 <= CONSOLIDATION_IMPORTANCE_THRESHOLD <= 1.0

    def test_consolidation_similarity_threshold(self):
        """Test CONSOLIDATION_SIMILARITY_THRESHOLD is in valid range."""
        assert CONSOLIDATION_SIMILARITY_THRESHOLD == 0.85
        assert 0.0 <= CONSOLIDATION_SIMILARITY_THRESHOLD <= 1.0
