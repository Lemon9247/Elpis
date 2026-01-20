"""Handlers for Hermes client orchestration.

This package contains the handlers that were moved from Psyche to Hermes
as part of making Psyche a stateless API:

- ReactHandler: ReAct loop orchestration for user input processing
- IdleHandler: Idle thinking and workspace exploration
- PsycheClient: Abstract interface for connecting to PsycheCore (local/remote)
"""

from hermes.handlers.idle_handler import IdleConfig, IdleHandler, ThoughtEvent
from hermes.handlers.psyche_client import LocalPsycheClient, PsycheClient, RemotePsycheClient
from hermes.handlers.react_handler import ReactConfig, ReactHandler, ToolCallResult

__all__ = [
    # React handler
    "ReactHandler",
    "ReactConfig",
    "ToolCallResult",
    # Idle handler
    "IdleHandler",
    "IdleConfig",
    "ThoughtEvent",
    # Client abstractions
    "PsycheClient",
    "LocalPsycheClient",
    "RemotePsycheClient",
]
