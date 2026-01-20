"""Memory management components for Psyche.

This module provides memory-related utilities including:
- ContextCompactor: Working memory management with automatic compaction
- CompactionResult: Result of context compaction operations

For the full architecture, see:
- psyche.core.server.PsycheCore - Memory coordination layer
- psyche.core.context_manager.ContextManager - Context management
- psyche.core.memory_handler.MemoryHandler - Memory handling
- hermes.handlers.psyche_client.RemotePsycheClient - HTTP client for Psyche
"""

from psyche.memory.compaction import CompactionResult, ContextCompactor

__all__ = [
    "ContextCompactor",
    "CompactionResult",
]
