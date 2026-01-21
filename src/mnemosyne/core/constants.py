"""Memory system constants - Mnemosyne is source of truth.

These constants define the core parameters for the memory system.
Other packages (Psyche) should import from here for memory-related constants.
"""

# Memory content length limits
MEMORY_SUMMARY_LENGTH = 500
"""Maximum length for memory summaries stored in Mnemosyne."""

# Consolidation thresholds
CONSOLIDATION_IMPORTANCE_THRESHOLD = 0.6
"""Minimum importance score for memory consolidation/promotion."""

CONSOLIDATION_SIMILARITY_THRESHOLD = 0.85
"""Cosine similarity threshold for clustering similar memories."""
