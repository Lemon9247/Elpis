"""Shared utilities for cross-package dependencies in Elpis.

This package contains code that is used across multiple packages (Elpis, Mnemosyne,
Psyche, Hermes) to avoid circular dependencies.
"""

from shared.constants import (
    AUTO_STORAGE_THRESHOLD,
    CONSOLIDATION_IMPORTANCE_THRESHOLD,
    CONSOLIDATION_SIMILARITY_THRESHOLD,
    MEMORY_CONTENT_TRUNCATE_LENGTH,
    MEMORY_SUMMARY_LENGTH,
)

__all__ = [
    "MEMORY_SUMMARY_LENGTH",
    "MEMORY_CONTENT_TRUNCATE_LENGTH",
    "AUTO_STORAGE_THRESHOLD",
    "CONSOLIDATION_IMPORTANCE_THRESHOLD",
    "CONSOLIDATION_SIMILARITY_THRESHOLD",
]
