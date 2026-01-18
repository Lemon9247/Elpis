"""
MCP Server - Psyche-aware client interface.

Provides MCP tools for clients that want direct access to Psyche
capabilities beyond the OpenAI-compatible HTTP API.

Tools exposed:
- chat: Send message and get response
- recall_memory: Explicit memory retrieval
- store_memory: Explicit memory storage
- get_emotion: Get current emotional state
- update_emotion: Report emotional event
- get_status: Get substrate status
- clear_context: Clear working memory
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger
from mcp.server import Server
from mcp.types import TextContent, Tool

if TYPE_CHECKING:
    from psyche.core.server import PsycheCore


def create_mcp_server(core: PsycheCore) -> Server:
    """
    Create an MCP server with Psyche tools.

    Args:
        core: PsycheCore instance for operations

    Returns:
        Configured MCP Server
    """
    server = Server("psyche")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available Psyche tools."""
        return [
            Tool(
                name="chat",
                description="Send a message and get a response with automatic memory retrieval and emotional modulation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to send",
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens to generate (default: 2048)",
                            "default": 2048,
                        },
                    },
                    "required": ["message"],
                },
            ),
            Tool(
                name="add_tool_result",
                description="Add a tool execution result to the conversation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool that was executed",
                        },
                        "result": {
                            "type": "string",
                            "description": "The tool execution result",
                        },
                    },
                    "required": ["tool_name", "result"],
                },
            ),
            Tool(
                name="recall_memory",
                description="Explicitly retrieve memories beyond auto-retrieval",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for memories",
                        },
                        "n": {
                            "type": "integer",
                            "description": "Number of memories to retrieve (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="store_memory",
                description="Explicitly store a memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Memory content to store",
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance score 0.0-1.0 (default: 0.5)",
                            "default": 0.5,
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization",
                        },
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="get_emotion",
                description="Get current emotional state from Elpis",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="update_emotion",
                description="Report an emotional event to Elpis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "event_type": {
                            "type": "string",
                            "description": "Type of emotional event (e.g., 'success', 'error', 'insight')",
                        },
                        "intensity": {
                            "type": "number",
                            "description": "Event intensity multiplier (default: 1.0)",
                            "default": 1.0,
                        },
                    },
                    "required": ["event_type"],
                },
            ),
            Tool(
                name="get_status",
                description="Get substrate status (working memory, tokens, emotion)",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="clear_context",
                description="Clear working memory buffer",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls."""
        try:
            result = await _handle_tool(core, name, arguments)
            return [TextContent(type="text", text=json.dumps(result))]
        except Exception as e:
            logger.error(f"Error in MCP tool {name}: {e}")
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return server


async def _handle_tool(
    core: PsycheCore,
    name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Handle individual tool calls."""

    if name == "chat":
        message = arguments["message"]
        max_tokens = arguments.get("max_tokens", 2048)

        # Add user message (triggers memory retrieval)
        memory_context = await core.add_user_message(message)

        # Generate response
        result = await core.generate(max_tokens=max_tokens)

        # Add to context
        await core.add_assistant_message(
            result["content"],
            user_message=message,
        )

        return {
            "response": result["content"],
            "thinking": result.get("thinking", ""),
            "has_thinking": result.get("has_thinking", False),
            "memory_context": memory_context,
        }

    elif name == "add_tool_result":
        tool_name = arguments["tool_name"]
        result = arguments["result"]
        core.add_tool_result(tool_name, result)
        return {"status": "ok"}

    elif name == "recall_memory":
        query = arguments["query"]
        n = arguments.get("n", 5)
        memories = await core.retrieve_memories(query, n)
        return {"memories": memories}

    elif name == "store_memory":
        content = arguments["content"]
        importance = arguments.get("importance", 0.5)
        tags = arguments.get("tags")
        success = await core.store_memory(content, importance, tags)
        return {"stored": success}

    elif name == "get_emotion":
        emotion = await core.get_emotion()
        return emotion

    elif name == "update_emotion":
        event_type = arguments["event_type"]
        intensity = arguments.get("intensity", 1.0)
        emotion = await core.update_emotion(event_type, intensity)
        return emotion

    elif name == "get_status":
        return {
            "context": core.context_summary,
            "mnemosyne_available": core.is_mnemosyne_available,
            "reasoning_enabled": core.reasoning_enabled,
        }

    elif name == "clear_context":
        core.clear_context()
        return {"status": "ok"}

    else:
        raise ValueError(f"Unknown tool: {name}")


class PsycheMCPServer:
    """
    Wrapper for MCP server with lifecycle management.

    Used by PsycheDaemon to run MCP alongside HTTP.
    """

    def __init__(self, core: PsycheCore):
        """
        Initialize MCP server wrapper.

        Args:
            core: PsycheCore instance
        """
        self.core = core
        self.server = create_mcp_server(core)
        self._running = False

    async def start(self) -> None:
        """Start the MCP server (stdio mode)."""
        import sys

        from mcp.server.stdio import stdio_server

        self._running = True
        logger.info("Starting MCP server on stdio...")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )

    def stop(self) -> None:
        """Stop the MCP server."""
        self._running = False
