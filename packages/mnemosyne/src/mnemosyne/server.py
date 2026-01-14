"""MCP server for memory management."""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

from mnemosyne.core.models import Memory, MemoryType, MemoryStatus, EmotionalContext
from mnemosyne.storage.chroma_store import ChromaMemoryStore


# Global state
memory_store: Optional[ChromaMemoryStore] = None
server = Server("mnemosyne")


def _ensure_initialized() -> None:
    """Ensure server components are initialized."""
    if memory_store is None:
        raise RuntimeError("Server not initialized")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="store_memory",
            description="Store a new memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Memory content"},
                    "summary": {"type": "string", "description": "Brief summary"},
                    "memory_type": {
                        "type": "string",
                        "enum": ["episodic", "semantic", "procedural", "emotional"],
                        "default": "episodic",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Memory tags",
                    },
                    "emotional_context": {
                        "type": "object",
                        "properties": {
                            "valence": {"type": "number"},
                            "arousal": {"type": "number"},
                            "quadrant": {"type": "string"},
                        },
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="search_memories",
            description="Search memories semantically",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "n_results": {
                        "type": "integer",
                        "default": 10,
                        "description": "Number of results",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_memory_stats",
            description="Get memory statistics",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    _ensure_initialized()

    try:
        if name == "store_memory":
            result = await _handle_store_memory(arguments)
        elif name == "search_memories":
            result = await _handle_search_memories(arguments)
        elif name == "get_memory_stats":
            result = await _handle_get_stats()
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.exception(f"Tool call failed: {name}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_store_memory(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle store_memory tool call."""
    # Create emotional context if provided
    emotional_ctx = None
    if args.get("emotional_context"):
        ec = args["emotional_context"]
        emotional_ctx = EmotionalContext(
            valence=ec["valence"],
            arousal=ec["arousal"],
            quadrant=ec["quadrant"],
        )

    # Create memory
    memory = Memory(
        content=args["content"],
        summary=args.get("summary", args["content"][:100]),
        memory_type=MemoryType(args.get("memory_type", "episodic")),
        tags=args.get("tags", []),
        emotional_context=emotional_ctx,
    )

    # Compute importance
    memory.importance_score = memory.compute_importance()

    # Store
    memory_store.add_memory(memory)

    return {
        "id": memory.id,
        "importance_score": memory.importance_score,
        "status": "stored",
    }


async def _handle_search_memories(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle search_memories tool call."""
    query = args["query"]
    n_results = args.get("n_results", 10)

    memories = memory_store.search_memories(query, n_results)

    return {
        "query": query,
        "results": [
            {
                "id": m.id,
                "content": m.content,
                "summary": m.summary,
                "importance_score": m.importance_score,
                "created_at": m.created_at.isoformat(),
            }
            for m in memories
        ],
    }


async def _handle_get_stats() -> Dict[str, Any]:
    """Handle get_memory_stats tool call."""
    return {
        "total_memories": memory_store.count_memories(),
        "short_term": memory_store.count_memories(MemoryStatus.SHORT_TERM),
        "long_term": memory_store.count_memories(MemoryStatus.LONG_TERM),
    }


def initialize(persist_directory: str = "./data/memory") -> None:
    """Initialize server components."""
    global memory_store

    logger.info("Initializing Mnemosyne memory server...")
    memory_store = ChromaMemoryStore(persist_directory=persist_directory)
    logger.info("Memory store initialized")


async def run_server() -> None:
    """Run the MCP server."""
    logger.info("Starting Mnemosyne MCP server...")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
