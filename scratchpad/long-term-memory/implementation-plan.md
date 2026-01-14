# Long-Term Memory Consolidation for Mnemosyne

## Summary

Implement memory consolidation in Mnemosyne to promote short-term memories to long-term storage. Design:
- **Trigger**: Client-driven (psyche calls during idle periods)
- **Algorithm**: Clustering-based consolidation with importance thresholds

---

## Current State

**Existing Infrastructure:**
- `ChromaMemoryStore` with two collections: `short_term_memory`, `long_term_memory`
- `MemoryStatus` enum: SHORT_TERM, CONSOLIDATING, LONG_TERM, ARCHIVED (CONSOLIDATING unused)
- Importance scoring: salience(40%) + recency(30%) + access_frequency(30%)
- `Memory.source_memory_ids` field for tracking consolidated memory lineage

**What's Missing:**
- No consolidation mechanism
- No MCP tools for consolidation workflow
- No memory clustering
- No update/delete operations in storage layer

---

## Architecture

```
Psyche (Harness)
│  Idle Detection → should_consolidate? → consolidate()
│
└──▶ Mnemosyne (Memory Server)
     ├── MCP Tools: consolidate_memories, should_consolidate, get_memory_context
     ├── Consolidator: clustering, promotion logic
     └── ChromaMemoryStore: short_term, long_term collections
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/mnemosyne/storage/chroma_store.py` | Add promote_memory(), delete_memory(), get_all_short_term(), get_embeddings_batch() |
| `src/mnemosyne/core/models.py` | Add ConsolidationConfig, ConsolidationReport, MemoryCluster |
| `src/mnemosyne/core/consolidator.py` | **NEW** - MemoryConsolidator class with clustering |
| `src/mnemosyne/server.py` | Add 5 new MCP tools |
| `tests/mnemosyne/test_consolidation.py` | **NEW** - Unit/integration tests |

---

## Phase 1: Storage Layer

**File: `src/mnemosyne/storage/chroma_store.py`**

Add methods:

```python
def get_all_short_term(self, limit: int = 1000) -> List[Memory]:
    """Get all short-term memories for consolidation."""

def get_embeddings_batch(self, memory_ids: List[str]) -> Dict[str, List[float]]:
    """Get embeddings for clustering similarity computation."""

def promote_memory(self, memory_id: str) -> bool:
    """Move memory from short_term to long_term collection."""
    # 1. Get from short_term (with embedding)
    # 2. Add to long_term with status=LONG_TERM
    # 3. Delete from short_term

def delete_memory(self, memory_id: str) -> bool:
    """Delete memory from either collection."""

def batch_delete(self, memory_ids: List[str]) -> int:
    """Delete multiple memories, return count deleted."""
```

---

## Phase 2: Data Models

**File: `src/mnemosyne/core/models.py`**

Add:

```python
@dataclass
class ConsolidationConfig:
    importance_threshold: float = 0.6      # Min importance for promotion
    min_age_hours: int = 1                 # Min age before eligible
    max_batch_size: int = 50               # Max memories per consolidation
    buffer_threshold: int = 100            # Recommend consolidation trigger
    similarity_threshold: float = 0.85     # For clustering similar memories
    min_cluster_size: int = 2              # Min memories to form cluster

@dataclass
class MemoryCluster:
    memories: List[Memory]
    centroid_embedding: List[float]
    avg_importance: float
    dominant_type: MemoryType

@dataclass
class ConsolidationReport:
    clusters_formed: int
    memories_promoted: int
    memories_archived: int
    memories_skipped: int
    total_processed: int
    duration_seconds: float
    cluster_summaries: List[Dict]  # {cluster_id, size, promoted_id, source_ids}
```

---

## Phase 3: Consolidator with Clustering

**File: `src/mnemosyne/core/consolidator.py`** (NEW)

```python
class MemoryConsolidator:
    def __init__(self, store: ChromaMemoryStore, config: ConsolidationConfig = None):
        self.store = store
        self.config = config or ConsolidationConfig()

    def should_consolidate(self) -> Tuple[bool, str]:
        """Check if consolidation is recommended."""
        count = self.store.get_short_term_count()
        if count >= self.config.buffer_threshold:
            return (True, f"Buffer size ({count}) exceeds threshold")
        return (False, "No consolidation needed")

    def get_consolidation_candidates(self) -> List[Memory]:
        """Get memories eligible for consolidation (by age, recompute importance)."""
        memories = self.store.get_all_short_term(limit=self.config.max_batch_size * 2)
        cutoff = datetime.now() - timedelta(hours=self.config.min_age_hours)

        candidates = []
        for m in memories:
            if m.created_at <= cutoff:
                m.importance_score = m.compute_importance()
                candidates.append(m)

        return sorted(candidates, key=lambda x: x.importance_score, reverse=True)[:self.config.max_batch_size]

    def cluster_memories(self, memories: List[Memory]) -> List[MemoryCluster]:
        """
        Cluster semantically similar memories using embeddings.

        Algorithm:
        1. Get embeddings for all candidate memories
        2. Compute pairwise cosine similarity
        3. Greedy clustering: assign to existing cluster if similarity > threshold
        4. Return clusters (including singleton clusters)
        """
        if not memories:
            return []

        # Get embeddings from ChromaDB
        memory_ids = [m.id for m in memories]
        embeddings = self.store.get_embeddings_batch(memory_ids)

        clusters = []
        assigned = set()

        for i, memory in enumerate(memories):
            if memory.id in assigned:
                continue

            cluster_members = [memory]
            cluster_embedding = np.array(embeddings[memory.id])
            assigned.add(memory.id)

            # Find similar memories
            for j, other in enumerate(memories[i+1:], i+1):
                if other.id in assigned:
                    continue

                other_emb = np.array(embeddings[other.id])
                similarity = cosine_similarity(cluster_embedding, other_emb)

                if similarity >= self.config.similarity_threshold:
                    cluster_members.append(other)
                    assigned.add(other.id)
                    # Update centroid (simple average)
                    cluster_embedding = (cluster_embedding + other_emb) / 2

            clusters.append(MemoryCluster(
                memories=cluster_members,
                centroid_embedding=cluster_embedding.tolist(),
                avg_importance=sum(m.importance_score for m in cluster_members) / len(cluster_members),
                dominant_type=max(set(m.memory_type for m in cluster_members), key=lambda t: sum(1 for m in cluster_members if m.memory_type == t))
            ))

        return clusters

    async def consolidate(self) -> ConsolidationReport:
        """
        Run consolidation cycle.

        Algorithm:
        1. Get candidates (filtered by age)
        2. Cluster similar memories
        3. For each cluster:
           - If avg_importance >= threshold: promote representative memory
           - Archive other cluster members (or delete if low importance)
           - If singleton with high importance: promote directly
        4. Return report
        """
        start = time.time()
        candidates = self.get_consolidation_candidates()
        clusters = self.cluster_memories(candidates)

        promoted = 0
        archived = 0
        skipped = 0
        cluster_summaries = []

        for cluster in clusters:
            if cluster.avg_importance >= self.config.importance_threshold:
                # Promote highest-importance memory as cluster representative
                representative = max(cluster.memories, key=lambda m: m.importance_score)
                representative.status = MemoryStatus.LONG_TERM
                representative.source_memory_ids = [m.id for m in cluster.memories if m.id != representative.id]

                if self.store.promote_memory(representative.id):
                    promoted += 1

                    # Archive or delete other cluster members
                    for m in cluster.memories:
                        if m.id != representative.id:
                            self.store.delete_memory(m.id)
                            archived += 1

                    cluster_summaries.append({
                        "promoted_id": representative.id,
                        "source_ids": representative.source_memory_ids,
                        "cluster_size": len(cluster.memories),
                        "avg_importance": cluster.avg_importance
                    })
            else:
                skipped += len(cluster.memories)

        return ConsolidationReport(
            clusters_formed=len(clusters),
            memories_promoted=promoted,
            memories_archived=archived,
            memories_skipped=skipped,
            total_processed=len(candidates),
            duration_seconds=time.time() - start,
            cluster_summaries=cluster_summaries
        )
```

---

## Phase 4: MCP Tools

**File: `src/mnemosyne/server.py`**

Add 5 tools:

### 1. `consolidate_memories`
```python
Tool(
    name="consolidate_memories",
    description="Run memory consolidation. Clusters similar short-term memories and promotes important ones to long-term.",
    inputSchema={
        "type": "object",
        "properties": {
            "importance_threshold": {"type": "number", "default": 0.6},
            "similarity_threshold": {"type": "number", "default": 0.85}
        }
    }
)
# Returns: {clusters_formed, memories_promoted, memories_archived, cluster_summaries}
```

### 2. `should_consolidate`
```python
Tool(
    name="should_consolidate",
    description="Check if consolidation is recommended.",
    inputSchema={"type": "object", "properties": {}}
)
# Returns: {should_consolidate: bool, reason: str, short_term_count, long_term_count}
```

### 3. `get_memory_context`
```python
Tool(
    name="get_memory_context",
    description="Get relevant memories formatted for context injection.",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_tokens": {"type": "integer", "default": 2000}
        },
        "required": ["query"]
    }
)
# Returns: {memories: [...], token_count, truncated}
```

### 4. `delete_memory`
```python
Tool(
    name="delete_memory",
    description="Delete a memory by ID.",
    inputSchema={
        "type": "object",
        "properties": {"memory_id": {"type": "string"}},
        "required": ["memory_id"]
    }
)
# Returns: {deleted: bool}
```

### 5. `get_recent_memories`
```python
Tool(
    name="get_recent_memories",
    description="Get memories from the last N hours.",
    inputSchema={
        "type": "object",
        "properties": {
            "hours": {"type": "integer", "default": 24},
            "limit": {"type": "integer", "default": 20}
        }
    }
)
# Returns: {memories: [...]}
```

---

## Phase 5: Tests

**File: `tests/mnemosyne/test_consolidation.py`** (NEW)

Test cases:
- `test_should_consolidate_when_buffer_exceeds_threshold`
- `test_get_consolidation_candidates_filters_by_age`
- `test_cluster_memories_groups_similar_embeddings`
- `test_cluster_memories_keeps_dissimilar_separate`
- `test_consolidate_promotes_high_importance_cluster`
- `test_consolidate_skips_low_importance_cluster`
- `test_consolidate_archives_cluster_members`
- `test_promote_memory_moves_between_collections`
- Integration test: full MCP tool cycle

---

## Implementation Order

1. **Phase 1**: Storage layer methods (chroma_store.py)
2. **Phase 2**: Data models (models.py)
3. **Phase 3**: Consolidator with clustering (consolidator.py)
4. **Phase 4**: MCP tools (server.py)
5. **Phase 5**: Tests

---

## Notes

- **Clustering uses embeddings already in ChromaDB** - no re-embedding needed
- **Cosine similarity** for semantic grouping (threshold 0.85)
- **Representative memory promoted** - highest importance in cluster
- **Other cluster members deleted** - their IDs stored in `source_memory_ids`
- **Future extension**: LLM summarization of clusters (not in v1)
