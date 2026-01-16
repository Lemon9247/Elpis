"""Memory management components for Psyche.

This module provides memory-related utilities including:
- ContextCompactor: Working memory management with automatic compaction
- CompactionResult: Result of context compaction operations

DEPRECATED exports (will be removed in future release):
- MemoryServer: Use psyche.core.server.PsycheCore instead
- ThoughtEvent: Use psyche.client.idle_handler.ThoughtEvent instead
- ServerState: Use psyche.core for the new architecture
- ServerConfig: Use psyche.core.server.CoreConfig instead

New architecture (recommended):
- psyche.core.server.PsycheCore - Memory coordination layer
- psyche.core.context_manager.ContextManager - Context management
- psyche.core.memory_handler.MemoryHandler - Memory handling
- psyche.client.react_handler.ReactHandler - ReAct loop handling
- psyche.client.idle_handler.IdleHandler - Idle thinking
"""

from psyche.memory.compaction import ContextCompactor, CompactionResult

# Deprecated imports - kept for backward compatibility
# These emit deprecation warnings when MemoryServer is instantiated
from psyche.memory.server import (
    MemoryServer,
    ServerConfig,
    ServerState,
    ThoughtEvent,
)

__all__ = [
    # Current API
    "ContextCompactor",
    "CompactionResult",
    # Deprecated (kept for backward compatibility)
    "MemoryServer",
    "ServerConfig",
    "ServerState",
    "ThoughtEvent",
]
