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
        except Exception as e:
            logger.debug(f"Memory {memory_id} not in short_term: {e}")

        # Try long-term
        try:
            result = self.long_term.get(ids=[memory_id])
            if result["ids"]:
                return self._result_to_memory(result, 0)
        except Exception as e:
            logger.debug(f"Memory {memory_id} not in long_term: {e}")

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

        # Search and combine results with distances
        results_with_distance: List[tuple[Memory, float]] = []
        for collection in collections:
            count = collection.count()
            if count == 0:
                continue  # Skip empty collections
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, count),
                include=["metadatas", "documents", "distances"],
            )

            for i in range(len(results["ids"][0])):
                memory = self._query_result_to_memory(results, 0, i)
                distance = results["distances"][0][i]
                results_with_distance.append((memory, distance))

        # Sort by distance (lower = more relevant)
        results_with_distance.sort(key=lambda x: x[1])

        # Return top N with relevance stored in metadata
        top_results = []
        for memory, distance in results_with_distance[:n_results]:
            memory.metadata["relevance_distance"] = distance
            top_results.append(memory)

        return top_results

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

    def get_all_short_term(self, limit: int = 1000) -> List[Memory]:
        """
        Get all short-term memories for consolidation evaluation.

        Args:
            limit: Maximum number of memories to retrieve

        Returns:
            List of short-term memories
        """
        try:
            count = self.short_term.count()
            if count == 0:
                return []

            # Get all short-term memories up to limit
            result = self.short_term.get(
                limit=min(count, limit),
                include=["documents", "metadatas"]
            )

            memories = []
            for i in range(len(result["ids"])):
                try:
                    memory = self._result_to_memory(result, i)
                    memories.append(memory)
                except Exception as e:
                    logger.warning(f"Failed to parse memory {result['ids'][i]}: {e}")

            logger.debug(f"Retrieved {len(memories)} short-term memories")
            return memories
        except Exception as e:
            logger.warning(f"Failed to get short-term memories: {e}")
            return []

    def get_short_term_count(self) -> int:
        """
        Efficient count of short-term buffer size.

        Returns:
            Number of memories in short-term storage
        """
        try:
            return self.short_term.count()
        except Exception as e:
            logger.warning(f"Failed to count short-term memories: {e}")
            return 0

    def get_embeddings_batch(self, memory_ids: List[str]) -> Dict[str, List[float]]:
        """
        Get embeddings for multiple memories by ID.

        Used for clustering similarity computation during consolidation.

        Args:
            memory_ids: List of memory IDs to get embeddings for

        Returns:
            Dictionary mapping memory_id to embedding vector
        """
        if not memory_ids:
            return {}

        embeddings_map: Dict[str, List[float]] = {}

        try:
            # Query short_term collection with embeddings
            result = self.short_term.get(
                ids=memory_ids,
                include=["embeddings"]
            )

            if result["ids"] and result["embeddings"]:
                for i, memory_id in enumerate(result["ids"]):
                    if result["embeddings"][i] is not None:
                        embeddings_map[memory_id] = result["embeddings"][i]

            logger.debug(f"Retrieved embeddings for {len(embeddings_map)} memories")
        except Exception as e:
            logger.warning(f"Failed to get embeddings batch: {e}")

        return embeddings_map

    def promote_memory(self, memory_id: str) -> bool:
        """
        Move a memory from short_term to long_term collection.

        Args:
            memory_id: ID of the memory to promote

        Returns:
            True on success, False otherwise
        """
        try:
            # Get from short_term with all data
            result = self.short_term.get(
                ids=[memory_id],
                include=["embeddings", "documents", "metadatas"]
            )

            if not result["ids"]:
                logger.warning(f"Memory {memory_id} not found in short_term")
                return False

            # Extract data
            embedding = result["embeddings"][0]
            document = result["documents"][0]
            metadata = result["metadatas"][0].copy()

            # Update status to LONG_TERM
            metadata["status"] = MemoryStatus.LONG_TERM.value

            # Add to long_term collection
            self.long_term.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )

            # Delete from short_term
            self.short_term.delete(ids=[memory_id])

            logger.debug(f"Promoted memory {memory_id} to long-term storage")
            return True
        except Exception as e:
            logger.warning(f"Failed to promote memory {memory_id}: {e}")
            return False

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from either collection.

        Tries short_term first, then long_term.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if deleted, False if not found
        """
        # Try short_term first
        try:
            result = self.short_term.get(ids=[memory_id])
            if result["ids"]:
                self.short_term.delete(ids=[memory_id])
                logger.debug(f"Deleted memory {memory_id} from short_term")
                return True
        except Exception as e:
            logger.debug(f"Memory {memory_id} not in short_term: {e}")

        # Try long_term
        try:
            result = self.long_term.get(ids=[memory_id])
            if result["ids"]:
                self.long_term.delete(ids=[memory_id])
                logger.debug(f"Deleted memory {memory_id} from long_term")
                return True
        except Exception as e:
            logger.debug(f"Memory {memory_id} not in long_term: {e}")

        logger.warning(f"Memory {memory_id} not found in any collection")
        return False

    def batch_delete(self, memory_ids: List[str]) -> int:
        """
        Delete multiple memories.

        Args:
            memory_ids: List of memory IDs to delete

        Returns:
            Count of successfully deleted memories
        """
        if not memory_ids:
            return 0

        deleted_count = 0

        for memory_id in memory_ids:
            if self.delete_memory(memory_id):
                deleted_count += 1

        logger.debug(f"Batch deleted {deleted_count}/{len(memory_ids)} memories")
        return deleted_count
