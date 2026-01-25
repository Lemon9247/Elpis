"""ChromaDB-based memory storage with hybrid search."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

from loguru import logger

from mnemosyne.core.models import EmotionalContext, Memory, MemoryStatus, MemoryType

if TYPE_CHECKING:
    from mnemosyne.config.settings import RetrievalSettings, StorageSettings


class ChromaMemoryStore:
    """
    Memory storage using ChromaDB vector database.

    Stores memories with embeddings for semantic search.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None,
        settings: Optional[StorageSettings] = None,
        retrieval_settings: Optional["RetrievalSettings"] = None,
    ):
        """
        Initialize the memory store.

        Args:
            persist_directory: Where to store the database (overrides settings)
            embedding_model: SentenceTransformer model name (overrides settings)
            settings: StorageSettings instance (provides defaults)
            retrieval_settings: RetrievalSettings instance for search configuration
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

        # Load settings if not provided
        if settings is None:
            from mnemosyne.config.settings import StorageSettings
            settings = StorageSettings()

        if retrieval_settings is None:
            from mnemosyne.config.settings import RetrievalSettings
            retrieval_settings = RetrievalSettings()

        self.retrieval_settings = retrieval_settings

        # Use explicit args if provided, otherwise fall back to settings
        _persist_directory = persist_directory or settings.persist_directory
        _embedding_model = embedding_model or settings.embedding_model
        _short_term_collection = settings.short_term_collection
        _long_term_collection = settings.long_term_collection

        self.persist_directory = Path(_persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collections
        self.short_term = self.client.get_or_create_collection(
            name=_short_term_collection,
            metadata={"description": "Recent memories, not yet consolidated"},
        )

        self.long_term = self.client.get_or_create_collection(
            name=_long_term_collection,
            metadata={"description": "Consolidated long-term memories"},
        )

        # Initialize embedding model
        logger.info(f"Loading embedding model: {_embedding_model}")
        self.embedding_model = SentenceTransformer(_embedding_model)

        # Initialize BM25 index for hybrid search
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_corpus: List[List[str]] = []
        self._bm25_doc_ids: List[str] = []
        self._bm25_available = BM25_AVAILABLE
        self._bm25_dirty = True  # Lazy rebuild flag

        if self._bm25_available:
            # Defer initial build until first search (lazy initialization)
            logger.info("BM25 hybrid search available (lazy initialization)")
        else:
            logger.warning("rank-bm25 not available, hybrid search disabled")

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

        # Mark BM25 index as dirty (lazy rebuild on next search)
        if self._bm25_available:
            self._bm25_dirty = True

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
                if self._bm25_available:
                    self._bm25_dirty = True
                logger.debug(f"Deleted memory {memory_id} from short_term")
                return True
        except Exception as e:
            logger.debug(f"Memory {memory_id} not in short_term: {e}")

        # Try long_term
        try:
            result = self.long_term.get(ids=[memory_id])
            if result["ids"]:
                self.long_term.delete(ids=[memory_id])
                if self._bm25_available:
                    self._bm25_dirty = True
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

        # BM25 already marked dirty by individual delete_memory calls
        logger.debug(f"Batch deleted {deleted_count}/{len(memory_ids)} memories")
        return deleted_count

    # -------------------------------------------------------------------------
    # BM25 and Hybrid Search Methods
    # -------------------------------------------------------------------------

    # Common English stop words to filter from BM25 indexing
    _STOP_WORDS = frozenset([
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
        "can", "could", "did", "do", "does", "doing", "done", "for", "from",
        "had", "has", "have", "having", "he", "her", "here", "hers", "him",
        "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
        "me", "might", "more", "most", "must", "my", "no", "nor", "not", "of",
        "on", "or", "our", "ours", "out", "own", "same", "she", "should", "so",
        "some", "such", "than", "that", "the", "their", "theirs", "them", "then",
        "there", "these", "they", "this", "those", "through", "to", "too", "under",
        "up", "very", "was", "we", "were", "what", "when", "where", "which",
        "while", "who", "whom", "why", "will", "with", "would", "you", "your",
        "yours", "yourself", "yourselves",
    ])

    def _tokenize(self, text: str, remove_stop_words: bool = True) -> List[str]:
        """
        Tokenization for BM25 with optional stop word removal.

        Converts to lowercase, extracts word tokens, and optionally
        removes common English stop words for better BM25 relevance.

        Args:
            text: Text to tokenize
            remove_stop_words: Whether to remove stop words (default True)

        Returns:
            List of lowercase word tokens
        """
        tokens = re.findall(r'\w+', text.lower())
        if remove_stop_words:
            tokens = [t for t in tokens if t not in self._STOP_WORDS and len(t) > 1]
        return tokens

    def _rebuild_bm25_index(self) -> None:
        """
        Rebuild the BM25 index from all memories.

        Called on init and after add/delete operations.
        """
        if not self._bm25_available:
            return

        corpus: List[List[str]] = []
        doc_ids: List[str] = []

        # Collect documents from both collections
        for collection in [self.short_term, self.long_term]:
            count = collection.count()
            if count == 0:
                continue

            try:
                result = collection.get(include=["documents"])
                for i, doc_id in enumerate(result["ids"]):
                    doc = result["documents"][i]
                    if doc:
                        corpus.append(self._tokenize(doc))
                        doc_ids.append(doc_id)
            except Exception as e:
                logger.warning(f"Failed to get documents from {collection.name}: {e}")

        # Build BM25 index
        if corpus:
            self._bm25_index = BM25Okapi(corpus)
            self._bm25_corpus = corpus
            self._bm25_doc_ids = doc_ids
            logger.debug(f"BM25 index rebuilt with {len(corpus)} documents")
        else:
            self._bm25_index = None
            self._bm25_corpus = []
            self._bm25_doc_ids = []
            logger.debug("BM25 index empty (no documents)")

    def _bm25_search(
        self,
        query: str,
        n_results: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Search using BM25 keyword matching.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of (memory_id, bm25_score) tuples, sorted by score descending
        """
        # Lazy rebuild if index is dirty
        if self._bm25_dirty:
            self._rebuild_bm25_index()
            self._bm25_dirty = False

        if not self._bm25_index or not self._bm25_doc_ids:
            return []

        # Tokenize query and get scores
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25_index.get_scores(query_tokens)

        # Pair scores with doc IDs and sort
        scored_docs = list(zip(self._bm25_doc_ids, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:n_results]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        k: int = 60,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Combine rankings using Reciprocal Rank Fusion.

        RRF score = sum(weight / (k + rank)) for each ranking list

        Args:
            vector_results: List of (memory_id, distance) from vector search
            bm25_results: List of (memory_id, score) from BM25 search
            k: RRF constant (default 60, standard value)
            vector_weight: Weight for vector search contribution
            bm25_weight: Weight for BM25 search contribution

        Returns:
            List of (memory_id, rrf_score) tuples, sorted by score descending
        """
        rrf_scores: Dict[str, float] = {}

        # Add vector search contributions (already sorted by distance, lower = better)
        for rank, (memory_id, _distance) in enumerate(vector_results):
            rrf_scores[memory_id] = rrf_scores.get(memory_id, 0) + vector_weight / (k + rank + 1)

        # Add BM25 contributions (already sorted by score, higher = better)
        for rank, (memory_id, _score) in enumerate(bm25_results):
            rrf_scores[memory_id] = rrf_scores.get(memory_id, 0) + bm25_weight / (k + rank + 1)

        # Sort by combined RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def _compute_quality_score(
        self,
        memory: Memory,
        base_score: float,
    ) -> float:
        """
        Compute quality-adjusted score for a memory.

        Combines:
        - Base relevance score (from RRF or semantic distance)
        - Recency decay (exponential, human-like forgetting)
        - Importance score
        - Content quality signals (length, type, role)

        All weights and factors are configurable via RetrievalSettings.

        Args:
            memory: Memory to score
            base_score: Base relevance score (higher = better)

        Returns:
            Quality-adjusted score (higher = better)
        """
        rs = self.retrieval_settings

        # Recency decay (configurable rate, e.g., 0.995^hours)
        try:
            age_hours = (datetime.now() - memory.created_at).total_seconds() / 3600
            recency = rs.recency_decay_rate ** min(age_hours, rs.max_recency_hours)
        except Exception:
            recency = 0.5  # Default if created_at is invalid

        # Importance score (already 0-1)
        importance = memory.importance_score

        # Content quality multipliers
        content_length = len(memory.content)
        length_factor = min(content_length, rs.max_content_length) / rs.max_content_length

        # Memory type factor (semantic memories are more valuable for recall)
        type_factor = rs.semantic_type_factor if memory.memory_type == MemoryType.SEMANTIC else 1.0

        # Role factor (configurable per role)
        if "assistant" in memory.tags:
            role_factor = rs.assistant_role_factor
        elif "user" in memory.tags:
            role_factor = rs.user_role_factor
        else:
            role_factor = 1.0

        # Normalize weights to sum to 1.0 for consistent scoring
        total_weight = rs.relevance_weight + rs.recency_weight + rs.importance_weight
        if total_weight > 0:
            norm_relevance = rs.relevance_weight / total_weight
            norm_recency = rs.recency_weight / total_weight
            norm_importance = rs.importance_weight / total_weight
        else:
            norm_relevance = norm_recency = norm_importance = 1.0 / 3.0

        # Weighted combination (normalized to [0, 1] range)
        weighted_score = (
            norm_relevance * min(1.0, base_score) +  # Clamp base_score to [0, 1]
            norm_recency * recency +
            norm_importance * importance
        )

        # Apply quality multipliers (capped to avoid runaway scores)
        # Multipliers adjust relative ranking but don't blow up scores
        combined_multiplier = min(2.0, length_factor * type_factor * role_factor)

        # Final score clamped to [0, 1] for consistent comparison
        return min(1.0, weighted_score * combined_multiplier)

    def _emotional_similarity(
        self,
        query_emotion: Optional[EmotionalContext],
        memory_emotion: Optional[EmotionalContext],
    ) -> float:
        """
        Compute emotional similarity between query and memory.

        Supports two modes (configured via retrieval_settings.use_angular_similarity):
        - Euclidean: Distance in valence-arousal space (default)
        - Angular: Cosine similarity of direction (ignores magnitude)

        Args:
            query_emotion: Current emotional context for query
            memory_emotion: Emotional context stored with memory

        Returns:
            Similarity score 0-1 (higher = more similar)
        """
        if not query_emotion or not memory_emotion:
            return 0.5  # Neutral if either is missing

        qv, qa = query_emotion.valence, query_emotion.arousal
        mv, ma = memory_emotion.valence, memory_emotion.arousal

        if self.retrieval_settings.use_angular_similarity:
            # Cosine similarity: focuses on direction, not magnitude
            # Useful when (0.8, 0.8) and (0.4, 0.4) should be "similar"
            dot_product = qv * mv + qa * ma
            q_mag = (qv**2 + qa**2) ** 0.5
            m_mag = (mv**2 + ma**2) ** 0.5

            if q_mag < 1e-6 or m_mag < 1e-6:
                return 0.5  # One or both are near origin (neutral)

            cosine = dot_product / (q_mag * m_mag)
            # Cosine is in [-1, 1], normalize to [0, 1]
            similarity = (cosine + 1.0) / 2.0
        else:
            # Euclidean distance (original behavior)
            valence_diff = qv - mv
            arousal_diff = qa - ma
            distance = (valence_diff**2 + arousal_diff**2) ** 0.5

            # Max distance in 2D space [-1, 1] x [-1, 1] is sqrt(8) ~ 2.83
            max_distance = 2.83
            similarity = 1.0 - (distance / max_distance)

        return similarity

    def search_memories_hybrid(
        self,
        query: str,
        n_results: int = 10,
        status_filter: Optional[MemoryStatus] = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        use_quality_scoring: bool = True,
        emotional_context: Optional[EmotionalContext] = None,
        emotion_weight: float = 0.3,
    ) -> List[Memory]:
        """
        Hybrid search combining vector similarity and BM25 keyword matching.

        Uses Reciprocal Rank Fusion (RRF) to combine rankings, then applies
        quality scoring and optional emotional reranking.

        Args:
            query: Search query
            n_results: Number of results to return
            status_filter: Filter by memory status
            vector_weight: Weight for vector search in RRF (default 0.5)
            bm25_weight: Weight for BM25 search in RRF (default 0.5)
            use_quality_scoring: Apply quality-based reranking (default True)
            emotional_context: Current emotional state for mood-congruent retrieval
            emotion_weight: Weight for emotional similarity (0-1, default 0.3)

        Returns:
            List of memories, ranked by combined score
        """
        # Determine candidate pool size (more for reranking)
        candidate_multiplier = 2 if (use_quality_scoring or emotional_context) else 1
        n_candidates = n_results * candidate_multiplier

        # Generate query embedding for vector search
        query_embedding = self.embedding_model.encode(query).tolist()

        # Determine which collections to search
        if status_filter == MemoryStatus.LONG_TERM:
            collections = [self.long_term]
        elif status_filter == MemoryStatus.SHORT_TERM:
            collections = [self.short_term]
        else:
            collections = [self.short_term, self.long_term]

        # Vector search across collections - fetch full documents to avoid double-query
        vector_results: List[Tuple[str, float]] = []
        memory_cache: Dict[str, Memory] = {}  # Cache to avoid re-fetching

        for collection in collections:
            count = collection.count()
            if count == 0:
                continue
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_candidates, count),
                include=["distances", "documents", "metadatas"],
            )
            for i, memory_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                vector_results.append((memory_id, distance))
                # Cache the memory object from query results
                try:
                    memory = self._query_result_to_memory(results, 0, i)
                    memory_cache[memory_id] = memory
                except Exception as e:
                    logger.debug(f"Failed to parse memory {memory_id}: {e}")

        # Sort vector results by distance (lower = better)
        vector_results.sort(key=lambda x: x[1])
        vector_results = vector_results[:n_candidates]

        # BM25 search (if available)
        if self._bm25_available and bm25_weight > 0:
            bm25_results = self._bm25_search(query, n_candidates)
        else:
            bm25_results = []

        # Combine with RRF (or use vector only if BM25 unavailable)
        if bm25_results:
            combined_results = self._reciprocal_rank_fusion(
                vector_results,
                bm25_results,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
            )
        else:
            # Convert vector distances to scores (invert so higher = better)
            combined_results = [
                (mem_id, 1.0 / (1.0 + dist))
                for mem_id, dist in vector_results
            ]

        # Build memory list from cache, falling back to get_memory for BM25-only results
        memories_with_scores: List[Tuple[Memory, float]] = []
        for memory_id, rrf_score in combined_results:
            memory = memory_cache.get(memory_id)
            if not memory:
                # BM25 found this but vector search didn't - fetch it
                memory = self.get_memory(memory_id)
            if memory:
                memories_with_scores.append((memory, rrf_score))

        # Apply quality scoring if enabled
        if use_quality_scoring:
            memories_with_scores = [
                (memory, self._compute_quality_score(memory, score))
                for memory, score in memories_with_scores
            ]

        # Apply emotional reranking if context provided
        if emotional_context and emotion_weight > 0:
            semantic_weight = 1.0 - emotion_weight
            memories_with_scores = [
                (
                    memory,
                    semantic_weight * score +
                    emotion_weight * self._emotional_similarity(emotional_context, memory.emotional_context)
                )
                for memory, score in memories_with_scores
            ]

        # Sort by final score (higher = better)
        memories_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top N with score in metadata
        results = []
        for memory, score in memories_with_scores[:n_results]:
            memory.metadata["relevance_score"] = score
            results.append(memory)

        return results
