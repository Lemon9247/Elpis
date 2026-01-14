"""MCP server for memory management."""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

from mnemosyne.core.models import Memory, MemoryType, MemoryStatus, EmotionalContext, ConsolidationConfig
from mnemosyne.core.consolidator import MemoryConsolidator
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
        Tool(
            name="consolidate_memories",
            description="Run memory consolidation. Clusters similar short-term memories and promotes important ones to long-term.",
            inputSchema={
                "type": "object",
                "properties": {
                    "importance_threshold": {
                        "type": "number",
                        "default": 0.6,
                        "description": "Minimum importance score for promotion (0-1)",
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "default": 0.85,
                        "description": "Similarity threshold for clustering (0-1)",
                    },
                },
            },
        ),
        Tool(
            name="should_consolidate",
            description="Check if memory consolidation is recommended based on buffer size",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_memory_context",
            description="Get relevant memories formatted for context injection",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to find relevant memories",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "default": 2000,
                        "description": "Maximum tokens to return",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="delete_memory",
            description="Delete a memory by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to delete",
                    },
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="get_recent_memories",
            description="Get memories from the last N hours",
            inputSchema={
                "type": "object",
                "properties": {
                    "hours": {
                        "type": "integer",
                        "default": 24,
                        "description": "Number of hours to look back",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of memories to return",
                    },
                },
            },
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
        elif name == "consolidate_memories":
            result = await _handle_consolidate_memories(arguments)
        elif name == "should_consolidate":
            result = await _handle_should_consolidate()
        elif name == "get_memory_context":
            result = await _handle_get_memory_context(arguments)
        elif name == "delete_memory":
            result = await _handle_delete_memory(arguments)
        elif name == "get_recent_memories":
            result = await _handle_get_recent_memories(arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.exception(f"Tool call failed: {name}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_store_memory(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle store_memory tool call."""
    # Create emotional context if provided (with safe key access)
    emotional_ctx = None
    if args.get("emotional_context"):
        ec = args["emotional_context"]
        emotional_ctx = EmotionalContext(
            valence=ec.get("valence", 0.0),
            arousal=ec.get("arousal", 0.0),
            quadrant=ec.get("quadrant", "neutral"),
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

    # Store (run in thread pool to avoid blocking event loop)
    await asyncio.to_thread(memory_store.add_memory, memory)

    return {
        "id": memory.id,
        "importance_score": memory.importance_score,
        "status": "stored",
    }


async def _handle_search_memories(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle search_memories tool call."""
    query = args["query"]
    n_results = args.get("n_results", 10)

    # Run in thread pool to avoid blocking event loop
    memories = await asyncio.to_thread(memory_store.search_memories, query, n_results)

    return {
        "query": query,
        "results": [
            {
                "id": m.id,
                "content": m.content,
                "summary": m.summary,
                "memory_type": m.memory_type.value,
                "tags": m.tags,
                "emotional_context": m.emotional_context.to_dict() if m.emotional_context else None,
                "importance_score": m.importance_score,
                "relevance_distance": m.metadata.get("relevance_distance"),
                "created_at": m.created_at.isoformat(),
            }
            for m in memories
        ],
    }


async def _handle_get_stats() -> Dict[str, Any]:
    """Handle get_memory_stats tool call."""
    # Run counts in thread pool to avoid blocking event loop
    total = await asyncio.to_thread(memory_store.count_memories)
    short_term = await asyncio.to_thread(memory_store.count_memories, MemoryStatus.SHORT_TERM)
    long_term = await asyncio.to_thread(memory_store.count_memories, MemoryStatus.LONG_TERM)
    return {
        "total_memories": total,
        "short_term": short_term,
        "long_term": long_term,
    }


async def _handle_consolidate_memories(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle consolidate_memories tool call."""
    # Build config with optional overrides
    config = ConsolidationConfig(
        importance_threshold=args.get("importance_threshold", 0.6),
        similarity_threshold=args.get("similarity_threshold", 0.85),
    )

    # Create consolidator and run in thread pool (long-running operation)
    consolidator = MemoryConsolidator(store=memory_store, config=config)
    report = await asyncio.to_thread(consolidator.consolidate)

    return report.to_dict()


async def _handle_should_consolidate() -> Dict[str, Any]:
    """Handle should_consolidate tool call."""
    consolidator = MemoryConsolidator(store=memory_store)
    should, reason = await asyncio.to_thread(consolidator.should_consolidate)

    short_term = await asyncio.to_thread(memory_store.count_memories, MemoryStatus.SHORT_TERM)
    long_term = await asyncio.to_thread(memory_store.count_memories, MemoryStatus.LONG_TERM)

    return {
        "should_consolidate": should,
        "reason": reason,
        "short_term_count": short_term,
        "long_term_count": long_term,
    }


async def _handle_get_memory_context(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle get_memory_context tool call."""
    query = args["query"]
    max_tokens = args.get("max_tokens", 2000)

    # Search for relevant memories (run in thread pool)
    memories = await asyncio.to_thread(memory_store.search_memories, query, 20)

    # Format memories for context and track token usage
    formatted_memories = []
    total_tokens = 0
    truncated = False

    for memory in memories:
        # Format memory entry
        memory_text = f"[{memory.created_at.isoformat()}] {memory.content}"
        # Rough token estimation: ~4 chars per token
        memory_tokens = len(memory_text) // 4

        if total_tokens + memory_tokens > max_tokens:
            truncated = True
            break

        formatted_memories.append({
            "id": memory.id,
            "content": memory.content,
            "summary": memory.summary,
            "created_at": memory.created_at.isoformat(),
            "importance_score": memory.importance_score,
        })
        total_tokens += memory_tokens

    return {
        "query": query,
        "memories": formatted_memories,
        "token_count": total_tokens,
        "truncated": truncated,
    }


async def _handle_delete_memory(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle delete_memory tool call."""
    memory_id = args["memory_id"]
    deleted = await asyncio.to_thread(memory_store.delete_memory, memory_id)

    return {
        "deleted": deleted,
        "memory_id": memory_id,
    }


async def _handle_get_recent_memories(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle get_recent_memories tool call."""
    hours = args.get("hours", 24)
    limit = args.get("limit", 20)

    cutoff = datetime.now() - timedelta(hours=hours)

    # Get all short-term memories (run in thread pool)
    short_term_memories = await asyncio.to_thread(
        memory_store.get_all_short_term, limit * 2
    )

    # Also search long-term for recent additions
    # Note: We'll combine and filter both collections
    all_memories = short_term_memories

    # Filter by cutoff time and sort by created_at
    recent_memories = [
        m for m in all_memories
        if m.created_at >= cutoff
    ]
    recent_memories.sort(key=lambda m: m.created_at, reverse=True)
    recent_memories = recent_memories[:limit]

    return {
        "memories": [
            {
                "id": m.id,
                "content": m.content,
                "summary": m.summary,
                "created_at": m.created_at.isoformat(),
                "importance_score": m.importance_score,
                "status": m.status.value,
            }
            for m in recent_memories
        ],
        "count": len(recent_memories),
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
