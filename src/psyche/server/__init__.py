"""
Psyche Server - External interfaces for the Psyche substrate.

This package provides:
- HTTP server with OpenAI-compatible /v1/chat/completions endpoint
- MCP server for Psyche-aware clients
- Server daemon for lifecycle management and dreaming
"""

from psyche.server.daemon import PsycheDaemon, ServerConfig
from psyche.server.http import PsycheHTTPServer
from psyche.server.mcp import PsycheMCPServer, create_mcp_server

__all__ = [
    "PsycheDaemon",
    "PsycheHTTPServer",
    "PsycheMCPServer",
    "ServerConfig",
    "create_mcp_server",
]
