"""
Memory System Sketches for Psyche
=================================

A biologically-inspired memory architecture with:
- Short-term memory (context window / rolling buffer)
- Long-term memory (ChromaDB with embeddings)
- Sleep consolidation (batch processing, clustering, compression)
- Emotional salience (memories tagged and weighted by emotional state)

This integrates with the existing Elpis emotional regulation system.

Contents:
1. Memory data structures
2. ChromaDB memory store
3. Memory formation (encoding)
4. Memory retrieval
5. Sleep consolidation system
6. Integration with Psyche
7. Example usage and tests
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import json

# Placeholder imports - adjust to your actual structure
# import chromadb
# from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer


# =============================================================================
# 1. MEMORY DATA STRUCTURES
# =============================================================================

class MemoryType(Enum):
    """Classification of memory types."""
    EPISODIC = "episodic"      # Specific events/conversations
    SEMANTIC = "semantic"       # General knowledge/facts
    PROCEDURAL = "procedural"   # How to do things (learned skills)
    EMOTIONAL = "emotional"     # Emotional associations/patterns


class MemoryStatus(Enum):
    """Memory lifecycle status."""
    SHORT_TERM = "short_term"   # Recent, not yet consolidated
    CONSOLIDATING = "consolidating"  # Being processed during sleep
    LONG_TERM = "long_term"     # Consolidated into permanent storage
    ARCHIVED = "archived"       # Old, rarely accessed, may be pruned


@dataclass
class EmotionalContext:
    """Emotional state at time of memory encoding."""
    valence: float  # -1 to 1
    arousal: float  # -1 to 1
    quadrant: str   # excited, frustrated, calm, depleted

    @property
    def salience(self) -> float:
        """
        Compute emotional salience score.

        Higher arousal = more salient (like in humans, emotional
        moments are remembered more vividly).
        Extreme valence also increases salience.
        """
        arousal_factor = (abs(self.arousal) + 1) / 2  # 0.5 to 1.0
        valence_factor = (abs(self.valence) + 1) / 2  # 0.5 to 1.0

        # Arousal matters more than valence for salience
        return (arousal_factor * 0.7) + (valence_factor * 0.3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "quadrant": self.quadrant,
            "salience": self.salience,
        }

    @classmethod
    def from_emotional_state(cls, state: "EmotionalState") -> "EmotionalContext":
        """Create from an EmotionalState instance."""
        return cls(
            valence=state.valence,
            arousal=state.arousal,
            quadrant=state.get_quadrant(),
        )


@dataclass
class Memory:
    """A single memory unit."""

    # Core content
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""  # The actual memory content
    summary: str = ""  # Compressed version for context injection

    # Classification
    memory_type: MemoryType = MemoryType.EPISODIC
    status: MemoryStatus = MemoryStatus.SHORT_TERM

    # Temporal info
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # Emotional context at encoding
    emotional_context: Optional[EmotionalContext] = None

    # Relationships
    related_memory_ids: List[str] = field(default_factory=list)
    source_memory_ids: List[str] = field(default_factory=list)  # For consolidated memories

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Computed importance (updated during consolidation)
    importance_score: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "memory_type": self.memory_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "emotional_context": self.emotional_context.to_dict() if self.emotional_context else None,
            "related_memory_ids": self.related_memory_ids,
            "source_memory_ids": self.source_memory_ids,
            "tags": self.tags,
            "metadata": self.metadata,
            "importance_score": self.importance_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Reconstruct from dictionary."""
        emotional_ctx = None
        if data.get("emotional_context"):
            ec = data["emotional_context"]
            emotional_ctx = EmotionalContext(
                valence=ec["valence"],
                arousal=ec["arousal"],
                quadrant=ec["quadrant"],
            )

        return cls(
            id=data["id"],
            content=data["content"],
            summary=data.get("summary", ""),
            memory_type=MemoryType(data["memory_type"]),
            status=MemoryStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
            emotional_context=emotional_ctx,
            related_memory_ids=data.get("related_memory_ids", []),
            source_memory_ids=data.get("source_memory_ids", []),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            importance_score=data.get("importance_score", 0.5),
        )

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = datetime.now()
        self.access_count += 1


# =============================================================================
# 2. CHROMADB MEMORY STORE
# =============================================================================

class MemoryStore:
    """
    Persistent memory storage using ChromaDB.

    Handles embedding generation, storage, and retrieval.
    """

    def __init__(
        self,
        persist_directory: str = "./data/memory",
        collection_name: str = "psyche_memories",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the memory store.

        Args:
            persist_directory: Where to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model: SentenceTransformer model for embeddings
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize ChromaDB
        # self.client = chromadb.Client(Settings(
        #     chroma_db_impl="duckdb+parquet",
        #     persist_directory=persist_directory,
        # ))
        # self.collection = self.client.get_or_create_collection(
        #     name=collection_name,
        #     metadata={"description": "Psyche's long-term memories"}
        # )

        # Initialize embedding model
        # self.embedder = SentenceTransformer(embedding_model)

        # Placeholder for sketch
        self.client = None
        self.collection = None
        self.embedder = None

        # In-memory cache for short-term memories not yet persisted
        self.short_term_buffer: List[Memory] = []
        self.buffer_max_size: int = 50

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        if self.embedder is None:
            # Placeholder - return dummy embedding
            return [0.0] * 384
        return self.embedder.encode(text).tolist()

    # -------------------------------------------------------------------------
    # Storage operations
    # -------------------------------------------------------------------------

    def store(self, memory: Memory) -> None:
        """
        Store a memory.

        Short-term memories go to buffer, long-term to ChromaDB.
        """
        if memory.status == MemoryStatus.SHORT_TERM:
            self._store_short_term(memory)
        else:
            self._store_long_term(memory)

    def _store_short_term(self, memory: Memory) -> None:
        """Add to short-term buffer."""
        self.short_term_buffer.append(memory)

        # If buffer is full, trigger consolidation warning
        if len(self.short_term_buffer) >= self.buffer_max_size:
            # In practice, this might trigger a sleep cycle
            pass  # logger.warning("Short-term buffer full, consolidation needed")

    def _store_long_term(self, memory: Memory) -> None:
        """Persist to ChromaDB."""
        if self.collection is None:
            # Placeholder - just log
            return

        # Generate embedding from content (or summary for efficiency)
        text_to_embed = memory.summary or memory.content
        embedding = self._generate_embedding(text_to_embed)

        # Prepare metadata (ChromaDB requires flat dict with simple types)
        metadata = {
            "memory_type": memory.memory_type.value,
            "status": memory.status.value,
            "created_at": memory.created_at.isoformat(),
            "importance_score": memory.importance_score,
            "access_count": memory.access_count,
            "tags": json.dumps(memory.tags),
        }

        if memory.emotional_context:
            metadata["emotional_valence"] = memory.emotional_context.valence
            metadata["emotional_arousal"] = memory.emotional_context.arousal
            metadata["emotional_quadrant"] = memory.emotional_context.quadrant
            metadata["emotional_salience"] = memory.emotional_context.salience

        # Store in ChromaDB
        self.collection.add(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.content],
            metadatas=[metadata],
        )

    def update(self, memory: Memory) -> None:
        """Update an existing memory."""
        if self.collection is None:
            return

        # Re-embed if content changed
        text_to_embed = memory.summary or memory.content
        embedding = self._generate_embedding(text_to_embed)

        metadata = {
            "memory_type": memory.memory_type.value,
            "status": memory.status.value,
            "created_at": memory.created_at.isoformat(),
            "last_accessed": memory.last_accessed.isoformat(),
            "importance_score": memory.importance_score,
            "access_count": memory.access_count,
            "tags": json.dumps(memory.tags),
        }

        if memory.emotional_context:
            metadata["emotional_valence"] = memory.emotional_context.valence
            metadata["emotional_arousal"] = memory.emotional_context.arousal
            metadata["emotional_quadrant"] = memory.emotional_context.quadrant
            metadata["emotional_salience"] = memory.emotional_context.salience

        self.collection.update(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.content],
            metadatas=[metadata],
        )

    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        # Check short-term buffer first
        self.short_term_buffer = [m for m in self.short_term_buffer if m.id != memory_id]

        # Delete from ChromaDB
        if self.collection:
            self.collection.delete(ids=[memory_id])

    # -------------------------------------------------------------------------
    # Retrieval operations
    # -------------------------------------------------------------------------

    def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by ID."""
        # Check short-term buffer
        for memory in self.short_term_buffer:
            if memory.id == memory_id:
                memory.touch()
                return memory

        # Query ChromaDB
        if self.collection:
            results = self.collection.get(ids=[memory_id])
            if results["ids"]:
                return self._result_to_memory(results, 0)

        return None

    def query_similar(
        self,
        query_text: str,
        n_results: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: float = 0.0,
        include_short_term: bool = True,
    ) -> List[Tuple[Memory, float]]:
        """
        Query for memories similar to the given text.

        Args:
            query_text: Text to find similar memories for
            n_results: Maximum number of results
            memory_types: Filter by memory type (None = all types)
            min_importance: Minimum importance score threshold
            include_short_term: Whether to search short-term buffer

        Returns:
            List of (Memory, similarity_score) tuples, sorted by relevance
        """
        results: List[Tuple[Memory, float]] = []

        # Search short-term buffer (simple keyword matching for now)
        if include_short_term:
            query_lower = query_text.lower()
            for memory in self.short_term_buffer:
                if memory_types and memory.memory_type not in memory_types:
                    continue
                if memory.importance_score < min_importance:
                    continue

                # Simple relevance: keyword overlap
                content_lower = memory.content.lower()
                overlap = sum(1 for word in query_lower.split() if word in content_lower)
                if overlap > 0:
                    score = overlap / len(query_lower.split())
                    results.append((memory, score))

        # Search ChromaDB
        if self.collection:
            query_embedding = self._generate_embedding(query_text)

            # Build where clause for filters
            where = {}
            if min_importance > 0:
                where["importance_score"] = {"$gte": min_importance}

            where_document = None
            if memory_types:
                # ChromaDB where clause for memory type
                type_values = [mt.value for mt in memory_types]
                where["memory_type"] = {"$in": type_values}

            chroma_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where if where else None,
            )

            # Convert to Memory objects with scores
            if chroma_results["ids"] and chroma_results["ids"][0]:
                for i, memory_id in enumerate(chroma_results["ids"][0]):
                    memory = self._result_to_memory(chroma_results, i)
                    if memory:
                        # ChromaDB returns distances, convert to similarity
                        distance = chroma_results["distances"][0][i] if chroma_results.get("distances") else 0
                        similarity = 1.0 / (1.0 + distance)  # Simple conversion
                        results.append((memory, similarity))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Touch retrieved memories (updates access count/time)
        for memory, _ in results[:n_results]:
            memory.touch()

        return results[:n_results]

    def query_recent(
        self,
        n_results: int = 10,
        include_short_term: bool = True,
    ) -> List[Memory]:
        """Get most recent memories."""
        results: List[Memory] = []

        if include_short_term:
            results.extend(self.short_term_buffer)

        # ChromaDB doesn't have great support for sorting by metadata,
        # so this might need a different approach in practice
        # (e.g., separate index, or just load recent IDs from a log)

        # Sort by created_at descending
        results.sort(key=lambda m: m.created_at, reverse=True)

        return results[:n_results]

    def query_by_emotion(
        self,
        quadrant: str,
        n_results: int = 5,
        min_salience: float = 0.5,
    ) -> List[Memory]:
        """
        Query memories by emotional quadrant.

        Useful for finding memories formed in similar emotional states.
        """
        if self.collection is None:
            return []

        results = self.collection.query(
            query_texts=[""],  # Empty query, just filtering
            n_results=n_results,
            where={
                "emotional_quadrant": quadrant,
                "emotional_salience": {"$gte": min_salience},
            },
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                memory = self._result_to_memory(results, i)
                if memory:
                    memories.append(memory)

        return memories

    def _result_to_memory(self, results: Dict, index: int) -> Optional[Memory]:
        """Convert ChromaDB result to Memory object."""
        try:
            metadata = results["metadatas"][0][index] if results.get("metadatas") else {}
            document = results["documents"][0][index] if results.get("documents") else ""
            memory_id = results["ids"][0][index] if results.get("ids") else str(uuid4())

            emotional_ctx = None
            if metadata.get("emotional_quadrant"):
                emotional_ctx = EmotionalContext(
                    valence=metadata.get("emotional_valence", 0),
                    arousal=metadata.get("emotional_arousal", 0),
                    quadrant=metadata["emotional_quadrant"],
                )

            return Memory(
                id=memory_id,
                content=document,
                memory_type=MemoryType(metadata.get("memory_type", "episodic")),
                status=MemoryStatus(metadata.get("status", "long_term")),
                created_at=datetime.fromisoformat(metadata["created_at"]) if metadata.get("created_at") else datetime.now(),
                importance_score=metadata.get("importance_score", 0.5),
                access_count=metadata.get("access_count", 0),
                emotional_context=emotional_ctx,
                tags=json.loads(metadata.get("tags", "[]")),
            )
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Bulk operations for sleep consolidation
    # -------------------------------------------------------------------------

    def get_short_term_buffer(self) -> List[Memory]:
        """Get all memories in short-term buffer."""
        return self.short_term_buffer.copy()

    def clear_short_term_buffer(self) -> None:
        """Clear the short-term buffer (after consolidation)."""
        self.short_term_buffer = []

    def get_memories_for_consolidation(
        self,
        max_age_hours: float = 24,
    ) -> List[Memory]:
        """Get short-term memories ready for consolidation."""
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)

        return [
            m for m in self.short_term_buffer
            if m.created_at.timestamp() < cutoff
        ]


# Continuing in next message due to length...
