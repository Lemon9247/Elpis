"""Shared utilities for cross-package dependencies in Elpis.

This package contains code that is used across multiple packages (Elpis, Mnemosyne,
Psyche, Hermes) to avoid circular dependencies.

Note: Constants have been moved to their respective packages:
- Memory constants (MEMORY_SUMMARY_LENGTH, CONSOLIDATION_*): mnemosyne.core.constants
- Psyche constants (AUTO_STORAGE_THRESHOLD, MEMORY_CONTENT_TRUNCATE_LENGTH): psyche.config.constants
"""

from shared.mcp_patch import apply_mcp_patch

__all__ = [
    "apply_mcp_patch",
]
