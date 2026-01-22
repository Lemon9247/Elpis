"""Psyche-specific constants for memory coordination.

These are constants specific to Psyche's behavior that don't belong in Mnemosyne.
For memory-related constants (consolidation thresholds, summary lengths), import
from mnemosyne.core.constants.
"""

# Display truncation length (for UI/logs, shorter than storage summary)
MEMORY_CONTENT_TRUNCATE_LENGTH = 300
"""Maximum length for memory content displayed in UI or logs."""

# Auto-storage threshold for importance scoring
AUTO_STORAGE_THRESHOLD = 0.6
"""Minimum importance score for automatic memory storage."""
