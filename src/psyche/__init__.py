"""Psyche - Core library for Elpis inference with memory and emotion.

Psyche provides the business logic layer for the Elpis system:
- PsycheCore: Central coordination of inference, memory, and emotion
- Handlers: ReactHandler and IdleHandler for processing
- Memory: Compaction, importance scoring, and reasoning extraction

For the TUI client, see the `echo` package.
"""

__version__ = "0.1.0"

# Core exports
from psyche.core import ContextConfig, CoreConfig, MemoryHandlerConfig, PsycheCore

# Handler exports
from psyche.handlers import (
    IdleConfig,
    IdleHandler,
    LocalPsycheClient,
    PsycheClient,
    ReactConfig,
    ReactHandler,
    RemotePsycheClient,
    ThoughtEvent,
    ToolCallResult,
)

__all__ = [
    # Core
    "PsycheCore",
    "CoreConfig",
    "ContextConfig",
    "MemoryHandlerConfig",
    # Handlers
    "ReactHandler",
    "ReactConfig",
    "ToolCallResult",
    "IdleHandler",
    "IdleConfig",
    "ThoughtEvent",
    "PsycheClient",
    "LocalPsycheClient",
    "RemotePsycheClient",
]
