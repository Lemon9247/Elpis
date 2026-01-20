"""Memory tools for interacting with Mnemosyne."""

from typing import Any, Dict, List, Optional

from loguru import logger

from psyche.mcp.client import EmotionalState, MnemosyneClient
from psyche.shared.constants import MEMORY_SUMMARY_LENGTH


class MemoryTools:
    """Tools for memory storage and retrieval via Mnemosyne."""

    def __init__(
        self,
        mnemosyne_client: MnemosyneClient,
        get_emotion_fn: Optional[callable] = None,
    ):
        """
        Initialize memory tools.

        Args:
            mnemosyne_client: Connected Mnemosyne client instance
            get_emotion_fn: Optional async function to get current emotional state
        """
        self.client = mnemosyne_client
        self.get_emotion_fn = get_emotion_fn

    async def recall_memory(
        self,
        query: str,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Search and recall memories from Mnemosyne.

        Args:
            query: Search query to find relevant memories
            n_results: Number of memories to retrieve

        Returns:
            Dict with success status and list of memories
        """
        try:
            memories = await self.client.search_memories(
                query=query,
                n_results=n_results,
            )

            # Format memories for display
            formatted = []
            for mem in memories:
                formatted.append({
                    "content": mem.get("content", ""),
                    "summary": mem.get("summary", ""),
                    "memory_type": mem.get("memory_type", "unknown"),
                    "tags": mem.get("tags", []),
                    "emotional_context": mem.get("emotional_context"),
                    "timestamp": mem.get("created_at"),
                    "relevance": mem.get("distance", 0),
                })

            logger.debug(f"Recalled {len(formatted)} memories for query: {query[:50]}")

            return {
                "success": True,
                "query": query,
                "count": len(formatted),
                "memories": formatted,
            }

        except Exception as e:
            logger.error(f"Memory recall failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def store_memory(
        self,
        content: str,
        summary: Optional[str] = None,
        memory_type: str = "episodic",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Store a new memory in Mnemosyne.

        Args:
            content: Content of the memory to store
            summary: Brief summary (auto-generated if not provided)
            memory_type: Type of memory (episodic, semantic, procedural, emotional)
            tags: Optional tags to categorize the memory

        Returns:
            Dict with success status and memory_id
        """
        try:
            # Auto-generate summary if not provided
            if not summary:
                summary = content[:MEMORY_SUMMARY_LENGTH] + (
                    "..." if len(content) > MEMORY_SUMMARY_LENGTH else ""
                )

            # Get current emotional context if available
            emotional_context = None
            if self.get_emotion_fn:
                try:
                    emotion = await self.get_emotion_fn()
                    emotional_context = {
                        "valence": emotion.valence,
                        "arousal": emotion.arousal,
                        "quadrant": emotion.quadrant,
                    }
                except Exception as e:
                    logger.debug(f"Could not get emotional context: {e}")

            result = await self.client.store_memory(
                content=content,
                summary=summary,
                memory_type=memory_type,
                tags=tags or [],
                emotional_context=emotional_context,
            )

            logger.debug(f"Stored memory: {summary[:50]}")

            return {
                "success": True,
                "memory_id": result.get("memory_id"),
                "summary": summary,
                "memory_type": memory_type,
                "tags": tags or [],
            }

        except Exception as e:
            logger.error(f"Memory storage failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }
