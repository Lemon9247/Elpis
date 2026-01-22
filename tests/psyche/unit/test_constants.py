"""Tests for Psyche constants."""

import pytest

from psyche.config.constants import (
    AUTO_STORAGE_THRESHOLD,
    MEMORY_CONTENT_TRUNCATE_LENGTH,
)


class TestPsycheConstants:
    """Test that Psyche constants have expected values."""

    def test_memory_content_truncate_length(self):
        """Test MEMORY_CONTENT_TRUNCATE_LENGTH is set correctly."""
        assert MEMORY_CONTENT_TRUNCATE_LENGTH == 300
        assert isinstance(MEMORY_CONTENT_TRUNCATE_LENGTH, int)

    def test_auto_storage_threshold(self):
        """Test AUTO_STORAGE_THRESHOLD is in valid range."""
        assert AUTO_STORAGE_THRESHOLD == 0.6
        assert 0.0 <= AUTO_STORAGE_THRESHOLD <= 1.0

    def test_truncate_vs_summary_relationship(self):
        """Test that truncate length is shorter than summary length."""
        from mnemosyne.core.constants import MEMORY_SUMMARY_LENGTH

        # Truncate is for display, summary is for storage - truncate should be shorter
        assert MEMORY_CONTENT_TRUNCATE_LENGTH < MEMORY_SUMMARY_LENGTH
