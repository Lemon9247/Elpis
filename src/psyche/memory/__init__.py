"""Memory management components for Psyche.

This module provides memory-related utilities including:
- ContextCompactor: Working memory management with automatic compaction
- CompactionResult: Result of context compaction operations

For the full architecture, see:
- psyche.core.server.PsycheCore - Memory coordination layer
- psyche.core.context_manager.ContextManager - Context management
- psyche.core.memory_handler.MemoryHandler - Memory handling
- hermes.handlers.react_handler.ReactHandler - ReAct loop handling (moved to hermes)
- hermes.handlers.idle_handler.IdleHandler - Idle thinking (moved to hermes)
"""

from psyche.memory.compaction import CompactionResult, ContextCompactor

__all__ = [
    "ContextCompactor",
    "CompactionResult",
]
