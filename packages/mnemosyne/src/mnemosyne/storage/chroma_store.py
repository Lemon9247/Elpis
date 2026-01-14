"""ChromaDB-based memory storage."""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from loguru import logger

from mnemosyne.core.models import Memory, MemoryStatus


class ChromaMemoryStore:
    """
    Memory storage using ChromaDB vector database.
    
    Stores memories with embeddings for semantic search.
    """

    def __init__(
        self,
        persist_directory: str = "./data/memory",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the memory store.
        
        Args:
            persist_directory: Where to store the database
            embedding_model: SentenceTransformer model name
        """
        if not CHROMADB_AVAILABLE:
            raise RuntimeError(
                "chromadb is required for ChromaMemoryStore. "
                "Install with: pip install chromadb"
            )
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collections
        self.short_term = self.client.get_or_create_collection(
            name="short_term_memory",
            metadata={"description": "Recent memories, not yet consolidated"},
        )

        self.long_term = self.client.get_or_create_collection(
            name="long_term_memory",
            metadata={"description": "Consolidated long-term memories"},
        )

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info("Memory store initialized")

    def add_memory(self, memory: Memory) -> None:
        """
        Add a memory to the store.
        
        Args:
            memory: Memory to add
        """
        # Generate embedding
        embedding = self.embedding_model.encode(memory.content).tolist()

        # Select collection based on status
        collection = (
            self.long_term
            if memory.status == MemoryStatus.LONG_TERM
            else self.short_term
        )

        # Add to ChromaDB
        collection.add(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.content],
            metadatas=[{
                "summary": memory.summary,
                "memory_type": memory.memory_type.value,
                "status": memory.status.value,
                "importance_score": memory.importance_score,
                "created_at": memory.created_at.isoformat(),
                "tags": json.dumps(memory.tags),
                "metadata_json": json.dumps(memory.metadata),
                "emotional_context": json.dumps(
                    memory.emotional_context.to_dict()
                    if memory.emotional_context
                    else None
                ),
            }],
        )

        logger.debug(f"Added memory {memory.id} to {collection.name}")

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: Memory ID to retrieve
            
        Returns:
            Memory if found, None otherwise
        """
        # Try short-term first
        try:
            result = self.short_term.get(ids=[memory_id])
            if result["ids"]:
                return self._result_to_memory(result, 0)
        except:
            pass

        # Try long-term
        try:
            result = self.long_term.get(ids=[memory_id])
            if result["ids"]:
                return self._result_to_memory(result, 0)
        except:
            pass

        return None

    def search_memories(
        self,
        query: str,
        n_results: int = 10,
        status_filter: Optional[MemoryStatus] = None,
    ) -> List[Memory]:
        """
        Semantic search for memories.
        
        Args:
            query: Search query
            n_results: Number of results to return
            status_filter: Filter by memory status
            
        Returns:
            List of matching memories
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Determine which collections to search
        collections = []
        if status_filter == MemoryStatus.LONG_TERM:
            collections = [self.long_term]
        elif status_filter == MemoryStatus.SHORT_TERM:
            collections = [self.short_term]
        else:
            collections = [self.short_term, self.long_term]

        # Search and combine results
        all_memories = []
        for collection in collections:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, collection.count()),
            )

            for i in range(len(results["ids"][0])):
                memory = self._query_result_to_memory(results, 0, i)
                all_memories.append(memory)

        # Sort by distance and return top N
        all_memories.sort(
            key=lambda m: m.importance_score,
            reverse=True
        )
        
        return all_memories[:n_results]

    def _result_to_memory(self, result: Dict, idx: int) -> Memory:
        """Convert ChromaDB get result to Memory object."""
        metadata = result["metadatas"][idx]
        
        memory_dict = {
            "id": result["ids"][idx],
            "content": result["documents"][idx],
            "summary": metadata.get("summary", ""),
            "memory_type": metadata.get("memory_type", "episodic"),
            "status": metadata.get("status", "short_term"),
            "created_at": metadata.get("created_at"),
            "last_accessed": metadata.get("created_at"),
            "importance_score": metadata.get("importance_score", 0.5),
            "tags": json.loads(metadata.get("tags", "[]")),
            "metadata": json.loads(metadata.get("metadata_json", "{}")),
            "emotional_context": json.loads(
                metadata.get("emotional_context", "null")
            ),
        }
        
        return Memory.from_dict(memory_dict)

    def _query_result_to_memory(
        self,
        results: Dict,
        query_idx: int,
        result_idx: int
    ) -> Memory:
        """Convert ChromaDB query result to Memory object."""
        metadata = results["metadatas"][query_idx][result_idx]
        
        memory_dict = {
            "id": results["ids"][query_idx][result_idx],
            "content": results["documents"][query_idx][result_idx],
            "summary": metadata.get("summary", ""),
            "memory_type": metadata.get("memory_type", "episodic"),
            "status": metadata.get("status", "short_term"),
            "created_at": metadata.get("created_at"),
            "last_accessed": metadata.get("created_at"),
            "importance_score": metadata.get("importance_score", 0.5),
            "tags": json.loads(metadata.get("tags", "[]")),
            "metadata": json.loads(metadata.get("metadata_json", "{}")),
            "emotional_context": json.loads(
                metadata.get("emotional_context", "null")
            ),
        }
        
        return Memory.from_dict(memory_dict)

    def count_memories(self, status: Optional[MemoryStatus] = None) -> int:
        """Count memories, optionally filtered by status."""
        if status == MemoryStatus.LONG_TERM:
            return self.long_term.count()
        elif status == MemoryStatus.SHORT_TERM:
            return self.short_term.count()
        else:
            return self.short_term.count() + self.long_term.count()
