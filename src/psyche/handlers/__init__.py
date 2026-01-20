"""Handlers for Psyche server behavior.

This package contains server-side handlers:
- DreamHandler: Server-side dreaming when no clients connected

Client-side connection is handled by RemotePsycheClient in hermes.handlers.
Psyche operates as a stateless memory-enriched inference API.
"""

from psyche.handlers.dream_handler import DreamConfig, DreamHandler

__all__ = [
    # Dream handler (server-side behavior)
    "DreamHandler",
    "DreamConfig",
]
