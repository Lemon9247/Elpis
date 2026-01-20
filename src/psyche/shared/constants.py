"""Shared constants for the Elpis project.

These constants are used across multiple packages (Elpis, Mnemosyne, Psyche, Hermes)
to ensure consistent behavior. Centralizing them here prevents duplication and
makes it easier to tune the system.
"""

# Memory content length limits
MEMORY_SUMMARY_LENGTH = 500
"""Maximum length for memory summaries stored in Mnemosyne."""

MEMORY_CONTENT_TRUNCATE_LENGTH = 300
"""Maximum length for memory content displayed in UI or logs."""

# Importance and storage thresholds
AUTO_STORAGE_THRESHOLD = 0.6
"""Minimum importance score for automatic memory storage."""

CONSOLIDATION_IMPORTANCE_THRESHOLD = 0.6
"""Minimum importance score for memory consolidation/promotion."""

CONSOLIDATION_SIMILARITY_THRESHOLD = 0.85
"""Cosine similarity threshold for clustering similar memories."""
