"""Handlers for Psyche business logic.

This package contains the core handlers that implement Psyche's behavior:
- ReactHandler: Processes user input with ReAct loop
- IdleHandler: Manages idle thinking and memory consolidation
- DreamHandler: Server-side dreaming when no clients connected
- PsycheClient: Abstract interface for connecting to PsycheCore
"""

from psyche.handlers.dream_handler import DreamConfig, DreamHandler
from psyche.handlers.idle_handler import IdleConfig, IdleHandler, ThoughtEvent
from psyche.handlers.psyche_client import LocalPsycheClient, PsycheClient, RemotePsycheClient
from psyche.handlers.react_handler import ReactConfig, ReactHandler, ToolCallResult

__all__ = [
    # React handler
    "ReactHandler",
    "ReactConfig",
    "ToolCallResult",
    # Idle handler
    "IdleHandler",
    "IdleConfig",
    "ThoughtEvent",
    # Dream handler
    "DreamHandler",
    "DreamConfig",
    # Client abstractions
    "PsycheClient",
    "LocalPsycheClient",
    "RemotePsycheClient",
]
