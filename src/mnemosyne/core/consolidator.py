"""Memory consolidation from short-term to long-term storage."""

import time
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
from loguru import logger

from mnemosyne.core.models import (
    ConsolidationConfig,
    ConsolidationReport,
    Memory,
    MemoryCluster,
    MemoryStatus,
    MemoryType,
)
from mnemosyne.storage.chroma_store import ChromaMemoryStore


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in range [-1, 1], or 0.0 if either vector has zero norm
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class MemoryConsolidator:
    """
    Handles memory consolidation from short-term to long-term storage.

    Uses clustering to group semantically similar memories and promotes
    representative memories to long-term storage.
    """

    def __init__(
        self,
        store: ChromaMemoryStore,
        config: ConsolidationConfig = None,
    ):
        """
        Initialize the consolidator.

        Args:
            store: The memory store to operate on
            config: Consolidation configuration (uses defaults if None)
        """
        self.store = store
        self.config = config or ConsolidationConfig()

    def should_consolidate(self) -> Tuple[bool, str]:
        """
        Check if consolidation is recommended.

        Consolidation is recommended when the short-term memory buffer
        exceeds the configured threshold.

        Returns:
            Tuple of (should_consolidate, reason)
        """
        count = self.store.get_short_term_count()

        if count >= self.config.buffer_threshold:
            return (
                True,
                f"Buffer size ({count}) exceeds threshold ({self.config.buffer_threshold})",
            )

        return (False, f"Buffer size ({count}) below threshold ({self.config.buffer_threshold})")

    def get_consolidation_candidates(self) -> List[Memory]:
        """
        Get memories eligible for consolidation.

        Criteria:
        - Status is SHORT_TERM
        - Age exceeds min_age_hours
        - Recompute importance score

        Returns:
            List of candidate memories, sorted by importance (descending),
            limited to max_batch_size
        """
        # Get more memories than batch size to account for filtering
        memories = self.store.get_all_short_term(limit=self.config.max_batch_size * 2)

        if not memories:
            logger.debug("No short-term memories found")
            return []

        # Filter by minimum age
        cutoff = datetime.now() - timedelta(hours=self.config.min_age_hours)
        candidates = []

        for memory in memories:
            if memory.created_at <= cutoff:
                # Recompute importance score with current factors
                memory.importance_score = memory.compute_importance()
                candidates.append(memory)

        logger.debug(
            f"Found {len(candidates)} candidates out of {len(memories)} short-term memories"
        )

        # Sort by importance (highest first) and limit to batch size
        candidates.sort(key=lambda m: m.importance_score, reverse=True)
        return candidates[: self.config.max_batch_size]

    def cluster_memories(self, memories: List[Memory]) -> List[MemoryCluster]:
        """
        Cluster semantically similar memories using embeddings.

        Algorithm:
        1. Get embeddings for all candidate memories
        2. Greedy clustering: start with first memory as cluster seed
        3. Add memories with similarity >= threshold to current cluster
        4. Continue until all memories assigned to a cluster

        Args:
            memories: List of memories to cluster

        Returns:
            List of MemoryCluster objects (including singleton clusters)
        """
        if not memories:
            return []

        # Get embeddings from ChromaDB
        memory_ids = [m.id for m in memories]
        embeddings = self.store.get_embeddings_batch(memory_ids)

        if not embeddings:
            logger.warning("No embeddings retrieved for clustering")
            # Return each memory as its own singleton cluster
            return [
                MemoryCluster(
                    memories=[m],
                    centroid_embedding=[],
                    avg_importance=m.importance_score,
                    dominant_type=m.memory_type,
                )
                for m in memories
            ]

        # Filter to only memories with embeddings
        memories_with_embeddings = [m for m in memories if m.id in embeddings]

        if not memories_with_embeddings:
            logger.warning("No memories have valid embeddings")
            return []

        clusters: List[MemoryCluster] = []
        assigned: set = set()

        for memory in memories_with_embeddings:
            if memory.id in assigned:
                continue

            # Start a new cluster with this memory
            cluster_members = [memory]
            cluster_embedding = np.array(embeddings[memory.id])
            assigned.add(memory.id)

            # Find similar memories to add to this cluster
            for other in memories_with_embeddings:
                if other.id in assigned:
                    continue

                other_emb = np.array(embeddings[other.id])
                similarity = cosine_similarity(cluster_embedding, other_emb)

                if similarity >= self.config.similarity_threshold:
                    cluster_members.append(other)
                    assigned.add(other.id)
                    # Update centroid as running average
                    n = len(cluster_members)
                    cluster_embedding = (
                        cluster_embedding * (n - 1) + other_emb
                    ) / n

            # Calculate cluster properties
            avg_importance = sum(m.importance_score for m in cluster_members) / len(
                cluster_members
            )

            # Find dominant memory type in cluster
            type_counts: dict = {}
            for m in cluster_members:
                type_counts[m.memory_type] = type_counts.get(m.memory_type, 0) + 1
            dominant_type = max(type_counts.keys(), key=lambda t: type_counts[t])

            clusters.append(
                MemoryCluster(
                    memories=cluster_members,
                    centroid_embedding=cluster_embedding.tolist(),
                    avg_importance=avg_importance,
                    dominant_type=dominant_type,
                )
            )

        logger.debug(
            f"Formed {len(clusters)} clusters from {len(memories_with_embeddings)} memories"
        )
        return clusters

    def consolidate(self) -> ConsolidationReport:
        """
        Run consolidation cycle.

        Algorithm:
        1. Get candidates (filtered by age)
        2. Cluster similar memories
        3. For each cluster with avg_importance >= threshold:
           - Promote highest-importance memory as representative
           - Store source_memory_ids for lineage
           - Delete other cluster members
        4. Return report

        Returns:
            ConsolidationReport with statistics about the consolidation
        """
        start_time = time.time()

        # Step 1: Get candidates
        candidates = self.get_consolidation_candidates()

        if not candidates:
            logger.info("No consolidation candidates found")
            return ConsolidationReport(
                clusters_formed=0,
                memories_promoted=0,
                memories_archived=0,
                memories_skipped=0,
                total_processed=0,
                duration_seconds=time.time() - start_time,
                cluster_summaries=[],
            )

        # Step 2: Cluster similar memories
        clusters = self.cluster_memories(candidates)

        promoted = 0
        archived = 0
        skipped = 0
        cluster_summaries: List[dict] = []

        # Step 3: Process each cluster
        for cluster in clusters:
            if cluster.avg_importance >= self.config.importance_threshold:
                # Find the representative (highest importance in cluster)
                representative = max(
                    cluster.memories, key=lambda m: m.importance_score
                )

                # Record source memory IDs for lineage tracking
                source_ids = [
                    m.id for m in cluster.memories if m.id != representative.id
                ]

                # Promote the representative
                if self.store.promote_memory(representative.id):
                    promoted += 1

                    # Delete other cluster members (they're now archived in the representative)
                    for memory in cluster.memories:
                        if memory.id != representative.id:
                            if self.store.delete_memory(memory.id):
                                archived += 1

                    cluster_summaries.append(
                        {
                            "promoted_id": representative.id,
                            "source_ids": source_ids,
                            "cluster_size": len(cluster.memories),
                            "avg_importance": cluster.avg_importance,
                        }
                    )

                    logger.debug(
                        f"Promoted memory {representative.id} representing "
                        f"{len(cluster.memories)} memories"
                    )
                else:
                    # Failed to promote, skip this cluster
                    skipped += len(cluster.memories)
                    logger.warning(
                        f"Failed to promote representative {representative.id}"
                    )
            else:
                # Cluster importance below threshold, skip
                skipped += len(cluster.memories)
                logger.debug(
                    f"Skipped cluster with importance {cluster.avg_importance:.3f} "
                    f"(threshold: {self.config.importance_threshold})"
                )

        duration = time.time() - start_time

        logger.info(
            f"Consolidation complete: {promoted} promoted, {archived} archived, "
            f"{skipped} skipped in {duration:.2f}s"
        )

        return ConsolidationReport(
            clusters_formed=len(clusters),
            memories_promoted=promoted,
            memories_archived=archived,
            memories_skipped=skipped,
            total_processed=len(candidates),
            duration_seconds=duration,
            cluster_summaries=cluster_summaries,
        )
