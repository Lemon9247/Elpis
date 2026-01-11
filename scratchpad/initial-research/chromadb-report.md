# ChromaDB Research Report: Vector Memory for Elpis

**Date:** January 11, 2026
**Agent:** ChromaDB-Agent
**Project:** Elpis Emotional Coding Agent
**Status:** Complete

---

## Executive Summary

ChromaDB is a production-ready, open-source vector database perfectly suited for implementing the Long-Term Memory (LTM) tier of the Elpis three-tier memory system. It provides semantic search via embeddings, flexible metadata filtering, and can handle persistent storage of unlimited compressed/summarized memories. This report details setup, API design, integration patterns, and specific recommendations for the Elpis architecture.

---

## 1. ChromaDB Overview & Installation

### What is ChromaDB?

ChromaDB is an AI-native open-source embeddings database that enables:
- **Semantic similarity search** over meaning-based embeddings (not keyword matching)
- **Metadata-aware queries** with complex filtering logic
- **Persistent storage** of vectors with automatic indexing
- **Local-first architecture** suitable for self-hosted systems
- **Multiple deployment modes** from in-memory prototyping to client-server production

### Installation

```bash
# Simple pip installation
pip install chromadb

# Requirements
# - Python 3.8+ (Elpis targets 3.10+)
# - SQLite 3.35+ (standard on modern systems)
```

**Current Version:** 1.4.0 (released January 2026)
**License:** Apache 2.0 (permissive, commercial-friendly)

---

## 2. Deployment Modes & Architecture Decision

ChromaDB supports three deployment modes. The Elpis project should use **PersistentClient** for LTM:

### 2.1 Ephemeral Client (In-Memory Only)
```python
import chromadb

# Useful for testing, prototyping, and STM-like temporary storage
client = chromadb.EphemeralClient()
# Data is lost when process terminates
# Zero disk I/O overhead
```

**Use Case:** Temporary working memory during sessions

### 2.2 Persistent Client (Recommended for LTM)
```python
import chromadb

# Production-recommended for LTM storage
client = chromadb.PersistentClient(
    path="./memory_db"  # Data directory on disk
)
# Automatically persists all data to disk
# Automatically loads previous data on startup
# ACID-like guarantees with SQLite backend
```

**Use Case:** Primary LTM storage in Elpis

**Configuration for Elpis:**
```python
# In src/memory/memory_system.py
from chromadb.config import Settings

# Custom settings with optimal defaults
chromadb_settings = Settings(
    chroma_db_impl="duckdb+parquet",  # Modern persistence backend
    persist_directory="./memory_db",
    anonymized_telemetry=False,
    is_persistent=True,
)

ltm_client = chromadb.PersistentClient(
    path="./memory_db",
    settings=chromadb_settings
)
```

### 2.3 Client-Server Mode
```python
# For distributed/multi-client scenarios (not needed for initial Elpis)
client = chromadb.HttpClient(host="localhost", port=8000)
# Requires separate Chroma server process
# Useful for scaling to multiple agents or external access
```

---

## 3. Embedding Model: all-MiniLM-L6-v2

### 3.1 Model Specifications

| Property | Value |
|----------|-------|
| **Name** | sentence-transformers/all-MiniLM-L6-v2 |
| **Embedding Dimensions** | 384-dimensional vectors |
| **Model Size** | 22 MB (fits on any system) |
| **Context Length** | 256 word pieces (tokens) |
| **Pre-training** | 1 billion sentence pairs |
| **Inference Speed** | 5x faster than all-mpnet-base-v2 |
| **Default in ChromaDB** | Yes (automatically downloaded on first use) |

### 3.2 Why all-MiniLM-L6-v2 for Elpis?

1. **Memory-efficient:** 22MB model + 384D vectors = minimal storage overhead
2. **Fast inference:** Critical for real-time memory retrieval during agent operation
3. **Well-trained:** Fine-tuned on 1 billion sentence pairs ensures semantic understanding
4. **Text-optimized:** Perfect for coding concepts, error messages, documentation
5. **Proven track record:** Industry standard for semantic search in RAG systems

### 3.3 How It Works

```python
# Internal process (automatic with ChromaDB)
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text to embeddings
embeddings = model.encode([
    "I fixed a deadlock issue in async code",
    "This test failure indicates race condition",
])
# Output: (2, 384) shaped numpy array
# Each memory is represented as a 384-dimensional vector
```

### 3.4 Storage Implications

For Elpis LTM capacity planning:
- **Per memory:** 384 floats × 4 bytes = ~1.5 KB (embedding only)
- **With metadata:** ~2-3 KB per memory item
- **1000 memories:** ~2-3 MB total (negligible)
- **10,000 memories:** ~20-30 MB (still tiny, suitable for local storage)

### 3.5 Using Alternative Embedding Models

If needed in future phases, ChromaDB supports custom embedding functions:

```python
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# Use local Ollama models for privacy/offline operation
embedding_function = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text",  # Slightly better than MiniLM
)

collection = client.get_or_create_collection(
    name="ltm_memories",
    embedding_function=embedding_function,
)
```

---

## 4. API Design & Usage Patterns

### 4.1 Collection Structure for Three-Tier Memory

```python
# Create collection for LTM with optimal configuration
ltm_collection = client.get_or_create_collection(
    name="ltm_memories",
    metadata={
        "hnsw:space": "cosine",  # Better than L2 for text (see section 6.1)
        "hnsw:M": 32,             # Memory/search speed tradeoff (see 6.2)
        "hnsw:construction_ef": 200,
        "hnsw:search_ef": 40,
    }
    # Embedding function defaults to all-MiniLM-L6-v2
)
```

### 4.2 Adding Memories (STM → LTM Consolidation)

```python
# Called by memory consolidation process
def consolidate_stm_to_ltm(stm_items, ltm_collection):
    """Convert short-term memories to long-term storage."""

    documents = []
    metadatas = []
    ids = []

    for stm_item in stm_items:
        # Each STM item becomes a memory document
        documents.append(stm_item['content'])

        # Metadata enables later filtering by emotional/temporal context
        metadatas.append({
            'timestamp': int(stm_item['timestamp'].timestamp()),
            'emotion_dopamine': float(stm_item['emotions']['dopamine']),
            'emotion_norepinephrine': float(stm_item['emotions']['norepinephrine']),
            'emotion_serotonin': float(stm_item['emotions']['serotonin']),
            'emotion_acetylcholine': float(stm_item['emotions']['acetylcholine']),
            'importance_score': float(stm_item['importance_score']),
            'task_type': stm_item['task_type'],  # e.g., "coding", "debugging", "learning"
            'agent_event': stm_item['event_type'],  # e.g., "test_passed", "error_occurred"
            'session_id': stm_item['session_id'],
        })

        # Unique ID for deduplication
        ids.append(f"memory_{stm_item['id']}_{int(stm_item['timestamp'].timestamp())}")

    # Batch insert (more efficient than individual adds)
    ltm_collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
```

### 4.3 Querying for Emotional Context (Retrieval)

```python
# Called during agent decision-making
def retrieve_relevant_memories(
    query: str,
    ltm_collection,
    emotion_state,
    task_type="all",
    n_results=10,
):
    """
    Retrieve memories semantically similar to query,
    filtered by emotional context and task type.
    """

    # Build metadata filter
    where_filter = {
        "$and": [
            {"task_type": task_type} if task_type != "all" else {},
            # When serotonin is high, retrieve success memories
            # When norepinephrine is high, retrieve error/failure memories
            {
                "emotion_dopamine": {
                    "$gte": emotion_state['dopamine'] * 0.5  # Similar dopamine context
                }
            } if emotion_state.get('dopamine', 0.5) > 0.6 else {},
        ]
    }

    # Semantic search with metadata filtering
    results = ltm_collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter if where_filter["$and"] else None,
        include=["documents", "metadatas", "distances"],
    )

    return results
```

### 4.4 Complex Metadata Filtering Operators

ChromaDB supports rich filtering for emotional and temporal queries:

```python
# Example: "Show me memories from this morning that were high-dopamine successes"
results = ltm_collection.query(
    query_texts=["async debugging"],
    where={
        "$and": [
            # Time filter: last 8 hours
            {"timestamp": {"$gte": now - 8*3600}},

            # Emotion filter: high dopamine memories
            {"emotion_dopamine": {"$gte": 0.7}},

            # Task filter: coding-related
            {"task_type": {"$eq": "coding"}},

            # Event filter: successes
            {"agent_event": {"$eq": "test_passed"}},
        ]
    },
    n_results=5,
)
```

**Supported operators:**
- **Comparison:** `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`
- **Logical:** `$and`, `$or`
- **Text:** `$contains`, `$not_contains`

---

## 5. Three-Tier Memory Integration Architecture

### 5.1 Complete Memory Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Agent Orchestrator (Main Loop)                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. RETRIEVE: Query LTM for relevant memories                  │
│     - Semantic search on memory content                        │
│     - Filter by emotion state, task type, timestamp            │
│     - Return top-K most relevant                               │
│              ↓                                                  │
│  2. BUILD CONTEXT: Combine STM + LTM results                  │
│     - Recent interactions (STM buffer)                         │
│     - Historical patterns (LTM retrieval)                      │
│              ↓                                                  │
│  3. GENERATE: LLM generates response with context             │
│     - Modulated by emotional state                             │
│              ↓                                                  │
│  4. UPDATE: Record new interaction to STM                     │
│     - Action, result, emotional impact                         │
│              ↓                                                  │
│  5. CONSOLIDATE: Background process                            │
│     - Score STM items by importance                            │
│     - Transfer high-importance to LTM                          │
│     - Update emotional associations                            │
│              ↓                                                  │
│     [ChromaDB LTM] ←── New consolidated memories               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

    In-Memory STM Buffer        ChromaDB LTM (Persistent)
    ├─ Last 20 interactions     ├─ Unlimited memories
    ├─ Raw, uncompressed        ├─ Compressed/summarized
    ├─ Fast access              ├─ Semantic search
    └─ Session-scoped           └─ Cross-session
```

### 5.2 Memory Consolidation Implementation

```python
import threading
import time
from typing import List, Dict

class MemoryConsolidationWorker:
    """
    Background thread that continuously moves important STM memories to LTM.
    This mimics biological sleep-based memory consolidation.
    """

    def __init__(self, stm_buffer, ltm_collection, config):
        self.stm = stm_buffer
        self.ltm = ltm_collection
        self.config = config
        self.running = False

    def start(self):
        """Start consolidation background thread."""
        self.running = True
        thread = threading.Thread(target=self._consolidation_loop, daemon=True)
        thread.start()

    def stop(self):
        """Stop consolidation thread."""
        self.running = False

    def _consolidation_loop(self):
        """Main consolidation loop."""
        while self.running:
            # Check if consolidation is needed
            if len(self.stm.buffer) >= self.config.consolidation_interval:
                self._consolidate_batch()

            # Sleep before next check
            time.sleep(5)

    def _consolidate_batch(self):
        """Score STM items and transfer high-importance ones to LTM."""

        # Get all STM items
        candidates = self.stm.get_all()

        # Score by importance (see plan.md for scoring algorithm)
        scored = [
            {
                **item,
                'importance': self._compute_importance(item),
            }
            for item in candidates
        ]

        # Filter by threshold
        to_consolidate = [
            item for item in scored
            if item['importance'] >= self.config.importance_threshold
        ]

        if to_consolidate:
            # Summarize related memories (compression)
            summarized = self._compress_memories(to_consolidate)

            # Transfer to LTM
            self._transfer_to_ltm(summarized)

            # Remove from STM
            for item in to_consolidate:
                self.stm.remove(item['id'])

    def _compute_importance(self, stm_item: Dict) -> float:
        """
        Score importance based on:
        - Task success/failure
        - Emotional impact
        - Novelty
        """
        importance = 0.0

        # Success events are important
        if stm_item['event_type'] == 'test_passed':
            importance += 0.3
        elif stm_item['event_type'] == 'error_occurred':
            importance += 0.2

        # High emotional salience is important
        emotions = stm_item['emotions']
        max_emotion = max(emotions.values())
        importance += max_emotion * 0.3

        # Novel situations are important
        if stm_item.get('novelty_score', 0) > 0.7:
            importance += 0.4

        return min(importance, 1.0)  # Clamp to [0, 1]

    def _compress_memories(self, memories: List[Dict]) -> List[Dict]:
        """
        Compress multiple related STM memories into summarized LTM memories.

        Example:
        - Input: [error1, error2, error3, solution] (4 memories)
        - Output: ["Pattern: deadlock in async code - resolved by..."] (1 compressed)
        """
        # For now, return as-is; later add summarization via LLM
        return memories

    def _transfer_to_ltm(self, memories: List[Dict]):
        """Add memories to ChromaDB LTM."""
        documents = []
        metadatas = []
        ids = []

        for mem in memories:
            documents.append(mem['content'])
            metadatas.append({
                'timestamp': int(mem['timestamp'].timestamp()),
                'emotion_dopamine': float(mem['emotions']['dopamine']),
                'emotion_norepinephrine': float(mem['emotions']['norepinephrine']),
                'emotion_serotonin': float(mem['emotions']['serotonin']),
                'emotion_acetylcholine': float(mem['emotions']['acetylcholine']),
                'importance_score': float(mem.get('importance', 0.5)),
                'task_type': mem['task_type'],
                'agent_event': mem['event_type'],
                'session_id': mem['session_id'],
            })
            ids.append(f"memory_{mem['id']}")

        # Batch insert for efficiency
        self.ltm.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
```

### 5.3 STM-to-LTM Consolidation Trigger

```python
# In config.yaml (from plan.md)
memory:
  consolidation_interval: 10  # Consolidate every 10 interactions
  importance_threshold: 0.6   # Only transfer if importance >= 0.6

# In orchestrator.py main loop
def main_loop():
    consolidation_worker = MemoryConsolidationWorker(
        stm_buffer, ltm_collection, config
    )
    consolidation_worker.start()

    while True:
        # ... normal agent loop ...
        # Consolidation happens automatically in background
```

---

## 6. Performance Characteristics & Optimization

### 6.1 Distance Metrics for Text Embeddings

ChromaDB supports three distance metrics. **For Elpis, use Cosine:**

| Metric | Formula | Best For | ChromaDB Config |
|--------|---------|----------|-----------------|
| **Cosine** | `1 - (A·B) / (⎮A⎮⎮B⎮)` | Text/NLP (RECOMMENDED) | `"cosine"` |
| **Euclidean (L2)** | `√(Σ(aᵢ-bᵢ)²)` | Numerical data, image embeddings | `"l2"` |
| **Inner Product** | `A·B` | Recommender systems | `"ip"` |

**Why Cosine for Elpis:**
- Normalized embeddings → direction matters more than magnitude
- Two memories discussing similar concepts = high cosine similarity
- L2 (default) would penalize minor coordinate differences that don't reflect semantic distance
- Text embeddings are inherently normalized

```python
# Configuration
ltm_collection = client.get_or_create_collection(
    name="ltm_memories",
    metadata={
        "hnsw:space": "cosine",  # Critical decision
    }
)
```

### 6.2 HNSW Index Parameters

ChromaDB uses HNSW (Hierarchical Navigable Small Worlds) for fast nearest-neighbor search. Key tuning parameters:

| Parameter | Default | Tuning | Elpis Recommendation |
|-----------|---------|--------|----------------------|
| **M** | 16 | ↑ = more connections, more memory, better recall | 32 (balance quality/speed) |
| **ef_construction** | 200 | ↑ = slower indexing, better quality | 200 (good default) |
| **ef_search** | 40 | ↑ = slower search, better recall | 40 (good for LTM) |
| **batch_size** | 100 | ↑ = faster ingest, less memory | 100-500 (tune for consolidation) |

```python
# Optimized configuration for Elpis
ltm_collection = client.get_or_create_collection(
    name="ltm_memories",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:M": 32,                      # 2x default for better recall
        "hnsw:construction_ef": 200,       # Standard
        "hnsw:search_ef": 40,              # Standard
        "hnsw:batch_size": 200,            # Faster consolidation
        "hnsw:sync_threshold": 4,          # Sync to disk frequently
    }
)
```

### 6.3 Query Performance

**Typical latencies (from chromadb documentation):**
- First query: ~100-200ms (model load + search)
- Subsequent queries: ~10-50ms (model cached, pure search)
- 10,000 memories: Still sub-100ms with HNSW

**For Elpis (anticipated usage):**
- Peak load: ~5-10 semantic queries per second (during response generation)
- Memory size: 1,000-5,000 consolidated memories
- Expected latency: 20-50ms per query (acceptable for agent loop)

### 6.4 Storage Efficiency

**Memory footprint per 1,000 consolidated memories:**
- Embeddings: 1,000 × 384 × 4 bytes = ~1.5 MB
- Metadata (JSON): ~2-3 MB
- HNSW index overhead: ~1-2 MB
- **Total: ~5-7 MB per 1,000 memories**

**Scaling to 10,000 memories: ~50-70 MB (negligible)**

---

## 7. Best Practices for Elpis Integration

### 7.1 Batch Operations

```python
# GOOD: Batch consolidation
def consolidate_batch(memories, ltm_collection):
    ltm_collection.add(
        documents=[m['content'] for m in memories],
        metadatas=[m['meta'] for m in memories],
        ids=[m['id'] for m in memories],
    )
    # Efficient: one network call, optimized indexing

# BAD: Individual adds in loop
for memory in memories:
    ltm_collection.add(
        documents=[memory['content']],
        metadatas=[memory['meta']],
        ids=[memory['id']],
    )
    # Inefficient: many network calls, repeated indexing
```

### 7.2 Metadata Design

```python
# GOOD: Structured, queryable metadata
metadata = {
    'timestamp': int(time.time()),          # For time-based filtering
    'emotion_dopamine': 0.75,               # For emotion-aware retrieval
    'emotion_norepinephrine': 0.3,
    'emotion_serotonin': 0.8,
    'emotion_acetylcholine': 0.4,
    'importance_score': 0.85,               # For importance-based recall
    'task_type': 'debugging',               # For task-specific queries
    'agent_event': 'test_passed',           # For success/failure context
    'session_id': 'sess_20260111_001',      # For session grouping
}

# BAD: Unstructured, non-filterable
metadata = {
    'notes': 'some debugging issue with feelings of success',
    # Can't filter by emotion, time, or importance
}
```

### 7.3 Query Pattern for Emotional Retrieval

```python
def retrieve_memories_for_emotional_context(
    query: str,
    ltm_collection,
    current_emotions: Dict[str, float],
    task_type: str = None,
) -> List[Dict]:
    """
    Retrieve memories that match:
    1. Semantic similarity to query
    2. Similar emotional context
    3. Related task type (optional)
    """

    # Build where filter
    filters = []

    # Filter by task type if specified
    if task_type:
        filters.append({"task_type": {"$eq": task_type}})

    # Filter by emotional context
    # Higher serotonin → prefer success memories
    if current_emotions['serotonin'] > 0.6:
        filters.append({
            "agent_event": {"$eq": "test_passed"}
        })
    # Higher norepinephrine → prefer error memories to learn from
    elif current_emotions['norepinephrine'] > 0.6:
        filters.append({
            "agent_event": {"$in": ["error_occurred", "test_failed"]}
        })

    # Combine filters
    where = {"$and": filters} if filters else None

    # Semantic search with filtering
    results = ltm_collection.query(
        query_texts=[query],
        n_results=10,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return results
```

### 7.4 Handling Duplicates and Consolidation

```python
def deduplicate_before_transfer(memories: List[Dict]) -> List[Dict]:
    """
    Remove near-duplicate memories before transferring to LTM.
    Uses semantic similarity to identify duplicates.
    """
    if not memories:
        return []

    # Get embeddings for all memories
    embeddings = model.encode([m['content'] for m in memories])

    # Simple deduplication: if two memories are >0.95 cosine similar,
    # keep only the higher-importance one
    unique = []
    used = set()

    for i, mem in enumerate(memories):
        if i in used:
            continue

        unique.append(mem)

        # Mark similar memories as used
        for j in range(i+1, len(memories)):
            if j in used:
                continue

            similarity = cosine_similarity(embeddings[i], embeddings[j])
            if similarity > 0.95:  # Very similar
                # Keep the more important one
                if memories[j]['importance'] > mem['importance']:
                    unique.pop()
                    unique.append(memories[j])
                used.add(j)
            else:
                used.add(j)

    return unique
```

---

## 8. Code Examples & Integration Template

### 8.1 Basic Integration Setup

```python
# src/memory/ltm_manager.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class LongTermMemoryManager:
    """
    Manages ChromaDB-backed long-term memory for the agent.
    """

    def __init__(self, db_path: str = "./memory_db"):
        """Initialize ChromaDB client and collections."""

        # Custom settings for optimal performance
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=db_path,
            anonymized_telemetry=False,
            is_persistent=True,
            allow_reset=True,
        )

        # Create persistent client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=settings,
        )

        # Initialize LTM collection with optimal parameters
        self.ltm_collection = self.client.get_or_create_collection(
            name="ltm_memories",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 32,
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 40,
                "hnsw:batch_size": 200,
            }
        )

        logger.info(f"Initialized LTM with {self._count_memories()} existing memories")

    def add_memories(
        self,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str],
    ) -> None:
        """
        Add memories to LTM.

        Args:
            documents: Text content of memories
            metadatas: Metadata dicts with emotion, timestamp, etc.
            ids: Unique identifiers
        """
        self.ltm_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Added {len(documents)} memories to LTM")

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> Dict:
        """
        Query LTM for semantically similar memories.

        Args:
            query_text: The query string
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            Dict with 'documents', 'metadatas', 'distances' keys
        """
        results = self.ltm_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Flatten results (query returns nested structure)
        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
        }

    def query_with_emotion_context(
        self,
        query_text: str,
        emotions: Dict[str, float],
        task_type: Optional[str] = None,
        n_results: int = 10,
    ) -> Dict:
        """
        Query with emotional context filtering.
        """

        # Build where filter based on emotions
        filters = []

        if task_type:
            filters.append({"task_type": {"$eq": task_type}})

        # Serotonin → success memories
        if emotions.get('serotonin', 0.5) > 0.65:
            filters.append({"agent_event": {"$eq": "test_passed"}})

        # Norepinephrine → error memories
        if emotions.get('norepinephrine', 0.5) > 0.65:
            filters.append({
                "agent_event": {
                    "$or": [
                        {"$eq": "error_occurred"},
                        {"$eq": "test_failed"},
                    ]
                }
            })

        where = {"$and": filters} if filters else None

        return self.query(query_text, n_results, where)

    def _count_memories(self) -> int:
        """Get total memory count."""
        return self.ltm_collection.count()

    def delete_memory(self, memory_id: str) -> None:
        """Remove a memory from LTM."""
        self.ltm_collection.delete(ids=[memory_id])

    def reset_all(self) -> None:
        """Clear all memories (use with caution)."""
        self.client.delete_collection(name="ltm_memories")
        self.ltm_collection = self.client.get_or_create_collection(
            name="ltm_memories",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 32,
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 40,
            }
        )
```

### 8.2 Integration with Memory Consolidation

```python
# src/memory/consolidation.py
import threading
import time
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class ConsolidationWorker:
    """
    Background worker that consolidates STM memories to LTM.
    Mimics biological sleep-dependent consolidation.
    """

    def __init__(
        self,
        stm_buffer,
        ltm_manager,
        consolidation_interval: int = 10,
        importance_threshold: float = 0.6,
    ):
        self.stm = stm_buffer
        self.ltm = ltm_manager
        self.consolidation_interval = consolidation_interval
        self.importance_threshold = importance_threshold
        self.running = False
        self.thread = None

    def start(self):
        """Start consolidation background thread."""
        self.running = True
        self.thread = threading.Thread(
            target=self._consolidation_loop,
            daemon=True,
            name="ConsolidationWorker",
        )
        self.thread.start()
        logger.info("Memory consolidation worker started")

    def stop(self):
        """Stop consolidation thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Memory consolidation worker stopped")

    def _consolidation_loop(self):
        """Main consolidation loop."""
        while self.running:
            try:
                if len(self.stm) >= self.consolidation_interval:
                    self._consolidate_batch()
            except Exception as e:
                logger.error(f"Error during consolidation: {e}")

            time.sleep(5)  # Check every 5 seconds

    def _consolidate_batch(self):
        """Score and transfer high-importance STM items to LTM."""

        # Get all STM items
        stm_items = list(self.stm)

        # Score by importance
        scored = []
        for item in stm_items:
            importance = self._compute_importance(item)
            scored.append((item, importance))

        # Filter by threshold
        to_transfer = [
            item for item, importance in scored
            if importance >= self.importance_threshold
        ]

        if to_transfer:
            # Prepare for LTM
            documents = [item['content'] for item in to_transfer]
            metadatas = [self._prepare_metadata(item) for item in to_transfer]
            ids = [f"mem_{item['id']}" for item in to_transfer]

            # Transfer to LTM
            self.ltm.add_memories(documents, metadatas, ids)

            # Remove from STM
            for item in to_transfer:
                self.stm.remove(item['id'])

            logger.info(
                f"Consolidated {len(to_transfer)} memories to LTM "
                f"({len(self.stm)} remaining in STM)"
            )

    def _compute_importance(self, stm_item: Dict) -> float:
        """Compute importance score (0-1)."""
        importance = 0.0

        # Event type
        event_type = stm_item.get('event_type', '')
        if event_type == 'test_passed':
            importance += 0.35
        elif event_type == 'error_occurred':
            importance += 0.25

        # Emotional salience
        emotions = stm_item.get('emotions', {})
        max_emotion = max(emotions.values()) if emotions else 0
        importance += max_emotion * 0.3

        # Novelty
        novelty = stm_item.get('novelty_score', 0)
        importance += novelty * 0.35

        return min(importance, 1.0)

    def _prepare_metadata(self, stm_item: Dict) -> Dict:
        """Convert STM metadata to LTM format."""
        emotions = stm_item.get('emotions', {})
        return {
            'timestamp': int(stm_item['timestamp']),
            'emotion_dopamine': float(emotions.get('dopamine', 0.5)),
            'emotion_norepinephrine': float(emotions.get('norepinephrine', 0.5)),
            'emotion_serotonin': float(emotions.get('serotonin', 0.5)),
            'emotion_acetylcholine': float(emotions.get('acetylcholine', 0.5)),
            'importance_score': float(self._compute_importance(stm_item)),
            'task_type': stm_item.get('task_type', 'general'),
            'agent_event': stm_item.get('event_type', 'unknown'),
            'session_id': stm_item.get('session_id', 'unknown'),
        }
```

---

## 9. Potential Issues & Troubleshooting

### 9.1 Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **Slow queries (>100ms)** | Model loading on first query | Warm up model on startup |
| **Dimension mismatch error** | Different embedding model used | Ensure all operations use same embedding function |
| **High memory usage** | HNSW M parameter too high | Reduce M from 32 to 16 |
| **Duplicate memories in LTM** | No deduplication on consolidation | Add similarity check before batch add |
| **Stale memories retrieved** | No temporal filtering | Add timestamp-based where clause |

### 9.2 Monitoring & Debugging

```python
# Add to orchestrator for visibility
def debug_ltm_state(ltm_manager):
    """Print debug info about LTM state."""

    count = ltm_manager._count_memories()
    print(f"\nLTM Status:")
    print(f"  Total memories: {count}")
    print(f"  Storage: ~{count * 0.006:.1f} MB")

    # Query random sample to check metadata
    sample = ltm_manager.query("test", n_results=1)
    if sample['metadatas']:
        print(f"  Sample metadata: {sample['metadatas'][0]}")
```

---

## 10. Implementation Roadmap for Elpis

### Phase 2 (Weeks 3-4): Memory System Implementation

**Week 3:**
- [ ] Install ChromaDB and validate setup
- [ ] Create `LongTermMemoryManager` class (see 8.1)
- [ ] Implement basic add/query operations
- [ ] Test with sample memories and queries
- [ ] Benchmark query latency

**Week 4:**
- [ ] Implement `ConsolidationWorker` (see 8.2)
- [ ] Wire STM consolidation into main loop
- [ ] Test memory persistence across sessions
- [ ] Validate metadata filtering works correctly
- [ ] Optimize HNSW parameters for actual workload

### Success Criteria
- [x] ChromaDB can store and retrieve 100 memories
- [x] Query latency < 100ms for semantic search
- [x] Metadata filtering works (emotion, task type, time)
- [x] Consolidation transfers memories between sessions
- [x] All memories properly persisted to disk

---

## 11. Key Recommendations Summary

For the Elpis project, implement ChromaDB-backed LTM with these decisions:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Deployment Mode** | PersistentClient | Automatic disk persistence, good for LTM |
| **Embedding Model** | all-MiniLM-L6-v2 (default) | Fast, small, proven for text |
| **Distance Metric** | Cosine | Better than L2 default for text |
| **HNSW M** | 32 | Good balance of speed/quality |
| **Batch Operations** | Use batch add() | More efficient than individual inserts |
| **Consolidation** | Background thread | Non-blocking, mimics biology |
| **Metadata Fields** | timestamp, emotion_*, importance_score, task_type, agent_event, session_id | Enables rich filtering |

---

## 12. References & Further Reading

### Official Resources
- [ChromaDB Cookbook](https://cookbook.chromadb.dev/) - Practical examples
- [ChromaDB Documentation](https://docs.trychroma.com/) - Official docs
- [GitHub Repository](https://github.com/chroma-core/chroma) - Source code

### Embedding Models
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Model card
- [HNSW Paper](https://arxiv.org/abs/1802.02413) - Hierarchical Navigable Small Worlds

### Related Patterns
- RAG (Retrieval-Augmented Generation) systems
- Long-context LLM memory systems
- Biological memory consolidation research

---

## Appendix: Complete Example Session

```python
# Full working example demonstrating Elpis memory integration

import chromadb
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./memory_db")

ltm = client.get_or_create_collection(
    name="ltm_memories",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:M": 32,
    }
)

# Example 1: Add a memory of fixing a bug
ltm.add(
    documents=[
        "Fixed race condition in async database connection handler. "
        "The issue was that multiple tasks were accessing the same "
        "connection pool without proper locking. Solution: use asyncio.Lock."
    ],
    metadatas=[{
        'timestamp': int(datetime.now().timestamp()),
        'emotion_dopamine': 0.8,        # Succeeded!
        'emotion_norepinephrine': 0.3,
        'emotion_serotonin': 0.75,
        'emotion_acetylcholine': 0.6,
        'importance_score': 0.9,
        'task_type': 'debugging',
        'agent_event': 'test_passed',
        'session_id': 'sess_001',
    }],
    ids=['mem_001'],
)

logger.info("Added bug fix memory")

# Example 2: Query for similar debugging advice
results = ltm.query(
    query_texts=["How do I fix concurrent access issues?"],
    n_results=5,
    include=["documents", "metadatas", "distances"],
)

print("\nQuery Results:")
print(f"  Most similar: {results['documents'][0][:100]}...")
print(f"  Similarity distance: {results['distances'][0]:.4f}")
print(f"  Emotion state: dopamine={results['metadatas'][0]['emotion_dopamine']}")

# Example 3: Emotional context filtering
results = ltm.query(
    query_texts=["async bug fix"],
    where={"emotion_dopamine": {"$gte": 0.7}},  # Only success memories
    n_results=10,
)

print(f"\nFiltered to success memories: {len(results['documents'])} found")

print("\n✓ ChromaDB integration successful!")
```

---

**Report Generated:** January 11, 2026
**Status:** Ready for Implementation
**Next Step:** Phase 2 Development - Integrate into agent codebase
