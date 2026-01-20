"""Psyche - Stateless memory-enriched inference API.

Psyche provides the server-side infrastructure for the Elpis system:
- PsycheCore: Central coordination of inference, memory, and emotion
- DreamHandler: Server-side dreaming when no clients connected
- Memory: Compaction, importance scoring, and reasoning extraction

Client-side handlers (ReactHandler, IdleHandler, PsycheClient) have been
moved to the `hermes` package to make Psyche a stateless API.
"""

__version__ = "0.1.0"

# Core exports
from psyche.core import ContextConfig, CoreConfig, MemoryHandlerConfig, PsycheCore

# Handler exports (only server-side handlers remain)
from psyche.handlers import DreamConfig, DreamHandler

__all__ = [
    # Core
    "PsycheCore",
    "CoreConfig",
    "ContextConfig",
    "MemoryHandlerConfig",
    # Server-side handlers
    "DreamHandler",
    "DreamConfig",
]
