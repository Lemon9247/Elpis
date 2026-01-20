"""Handlers for Psyche server behavior.

This package now contains only server-side handlers:
- DreamHandler: Server-side dreaming when no clients connected

The following handlers have been moved to Hermes (client-side orchestration):
- ReactHandler -> hermes.handlers.react_handler
- IdleHandler -> hermes.handlers.idle_handler
- PsycheClient -> hermes.handlers.psyche_client

This move is part of making Psyche a stateless memory-enriched inference API.
"""

from psyche.handlers.dream_handler import DreamConfig, DreamHandler

__all__ = [
    # Dream handler (server-side behavior)
    "DreamHandler",
    "DreamConfig",
]
