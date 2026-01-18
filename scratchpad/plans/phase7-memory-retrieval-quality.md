# Phase 7: Memory Retrieval Quality Improvements

## Research Summary

See: `scratchpad/reports/2026-01-18-memory-systems-research.md`

Key findings from state-of-the-art systems (Zep, Mem0, SimpleMem):
- **Hybrid search is now standard** - BM25 + vector + metadata, combined via RRF
- **Storage filtering is critical** - Entropy-aware filtering prevents pollution
- **Quality-weighted ranking** - Combine relevance, recency, importance, type
- **Human-like decay** - Memories that aren't accessed fade over time

---

## Problem Statement

Memory retrieval is returning poor results. Despite successful storage and retrieval mechanics (verified in Phase 6), semantic search returns contextually irrelevant results:

1. **Questions rank higher than answers** - "Who is your mother" returns the question itself rather than the answer
2. **Short compacted snippets pollute results** - Brief messages rank highly due to embedding similarity
3. **Contradictory memories exist** - Multiple "truths" stored (e.g., "Nyx is my mother" vs "Willow is my creator")
4. **No filtering by content quality** - All memories treated equally regardless of type or role

### Evidence from ChromaDB Analysis

```
Query: "Who is your mother"
Results:
1. "Who is your mother" (distance: 0.477) ← The question itself!
2. "who is your creator" (distance: 0.615) ← Another question
3. "Nyx, of course... [long response]" (distance: 1.066) ← Actual answer, ranked 3rd
```

The semantic similarity model finds questions about "mother" more similar to the query "Who is your mother" than answers containing actual information about mothers.

**Why this happens:** Vector embeddings capture semantic similarity, not information content. Questions and answers about the same topic have similar embeddings, but BM25 keyword search would favor documents containing actual answer content.

---

## Root Cause Analysis

### 1. Compaction Storage is Indiscriminate

**File:** `src/psyche/core/memory_handler.py:177-194`

```python
for msg in messages:
    if msg.role == "system":
        continue  # Skip system prompts
    # Stores ALL user AND assistant messages
    await self.mnemosyne_client.store_memory(
        content=msg.content,
        summary=msg.content[:500],
        memory_type="episodic",  # Always episodic
        tags=["compacted", msg.role],
        ...
    )
```

**Issues:**
- User questions are stored as memories (useless for retrieval)
- No minimum content length filter
- Always uses `episodic` type, no semantic/knowledge differentiation
- Summary is just content truncated, not actual summarization

### 2. Retrieval Has No Quality Filters

**File:** `src/mnemosyne/storage/chroma_store.py:152-207`

```python
def search_memories(self, query: str, n_results: int = 10, ...):
    # Pure semantic search - no content filters
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, count),
        ...
    )
```

**Issues:**
- No minimum content length filter
- No memory type weighting (semantic vs episodic)
- No role filtering (assistant responses vs user questions)
- No importance score threshold
- Returns raw semantic similarity without quality adjustment

### 3. Memory Types Not Utilized

Mnemosyne supports four memory types:
- `episodic` - Specific events/conversations
- `semantic` - General knowledge/facts (should be preferred for recall)
- `procedural` - How to do things
- `emotional` - Emotional associations

But compaction always stores as `episodic` and retrieval doesn't weight by type.

---

## Proposed Solutions

### Solution A: Storage-Side Filtering (Recommended)

Filter what gets stored during compaction:

1. **Minimum content length** - Skip messages under 50 chars
2. **Role-based filtering** - Only store assistant messages (they contain the knowledge)
3. **Better summarization** - Use LLM to extract key facts from messages
4. **Smarter type assignment** - Detect factual statements → `semantic`, emotional content → `emotional`

**Pros:** Prevents pollution at source, smaller database, faster search
**Cons:** Loses user context, can't retroactively fix existing data

### Solution B: Retrieval-Side Filtering

Add quality filters during search:

1. **ChromaDB metadata filters** - Filter by role, min length, memory type
2. **Post-retrieval ranking** - Score by content length, type, importance
3. **Hybrid search** - Combine semantic similarity with keyword matching

**Pros:** Works with existing data, more flexible
**Cons:** Larger database, slower search, filtering done after embedding lookup

### Solution C: Combined Approach (Recommended)

Implement both:
1. Fix storage to prevent future pollution (Solution A)
2. Add retrieval filters to handle existing data (Solution B)
3. Add cleanup tool to remove low-quality memories from existing database

---

## Implementation Plan

### Step 1: Add Hybrid Search (BM25 + Vector)

**File:** `src/mnemosyne/storage/chroma_store.py`

This is the highest-impact change based on research. Anthropic found hybrid search reduces retrieval failure by **67%**.

#### 1.1 Add BM25 Index

```python
from rank_bm25 import BM25Okapi
import re

class ChromaMemoryStore:
    def __init__(self, ...):
        ...
        # BM25 index for keyword search
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_doc_ids: List[str] = []
        self._rebuild_bm25_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        return re.findall(r'\w+', text.lower())

    def _rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index from all memories."""
        all_docs = []
        self._bm25_doc_ids = []

        for collection in [self.short_term, self.long_term]:
            if collection.count() == 0:
                continue
            result = collection.get(include=["documents"])
            for i, doc_id in enumerate(result["ids"]):
                all_docs.append(self._tokenize(result["documents"][i]))
                self._bm25_doc_ids.append(doc_id)

        if all_docs:
            self._bm25_index = BM25Okapi(all_docs)

    def add_memory(self, memory: Memory) -> None:
        """Add memory and update BM25 index."""
        # ... existing code ...
        # After adding to ChromaDB:
        self._rebuild_bm25_index()  # Or incremental update
```

#### 1.2 Implement Hybrid Search with RRF

```python
def search_memories_hybrid(
    self,
    query: str,
    n_results: int = 10,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
) -> List[Memory]:
    """
    Hybrid search combining vector similarity and BM25.

    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """
    # 1. Vector search (existing)
    query_embedding = self.embedding_model.encode(query).tolist()
    vector_results = self._vector_search(query_embedding, n_results * 2)

    # 2. BM25 search
    bm25_results = self._bm25_search(query, n_results * 2)

    # 3. Reciprocal Rank Fusion
    rrf_scores: Dict[str, float] = {}
    k = 60  # RRF constant

    for rank, (memory_id, _) in enumerate(vector_results):
        rrf_scores[memory_id] = rrf_scores.get(memory_id, 0) + vector_weight / (k + rank + 1)

    for rank, (memory_id, _) in enumerate(bm25_results):
        rrf_scores[memory_id] = rrf_scores.get(memory_id, 0) + bm25_weight / (k + rank + 1)

    # 4. Sort by combined score and return top N
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    return [self.get_memory(mid) for mid in sorted_ids[:n_results] if mid]
```

---

### Step 2: Storage-Side Filtering

**File:** `src/psyche/core/memory_handler.py`

#### 2.1 Add Content Quality Assessment

```python
MIN_MEMORY_LENGTH = 50  # Minimum chars for storage
KNOWLEDGE_INDICATORS = ["I ", "My ", "You ", "The ", "This ", "That "]  # Declarative statements

def _should_store_message(self, msg: Message) -> tuple[bool, str]:
    """Determine if a message is worth storing as a memory.

    Returns:
        (should_store, memory_type)
    """
    # Skip very short messages
    if len(msg.content) < MIN_MEMORY_LENGTH:
        return False, "episodic"

    # User questions are rarely useful to recall
    if msg.role == "user":
        # Check if it's actually a question
        if "?" in msg.content or msg.content.strip().lower().startswith(
            ("who", "what", "where", "when", "why", "how", "can", "do", "is", "are")
        ):
            return False, "episodic"

    # Assistant messages with factual content → semantic
    if msg.role == "assistant":
        if any(indicator in msg.content for indicator in KNOWLEDGE_INDICATORS):
            return True, "semantic"

    return True, "episodic"
```

#### 2.2 Update store_messages()

```python
async def store_messages(self, messages: List[Message], ...) -> bool:
    for msg in messages:
        if msg.role == "system":
            continue

        should_store, memory_type = self._should_store_message(msg)
        if not should_store:
            logger.debug(f"Skipping low-quality message: {msg.content[:30]}...")
            continue

        await self.mnemosyne_client.store_memory(
            content=msg.content,
            summary=msg.content[:500],
            memory_type=memory_type,  # Dynamic type
            tags=["compacted", msg.role, memory_type],
            ...
        )
```

---

### Step 3: Quality-Weighted Ranking (Research-Informed)

Based on Park et al.'s Generative Agents and human memory research:

```python
import math
from datetime import datetime

# Decay factor per hour (0.995^24 ≈ 0.89 after 1 day)
RECENCY_DECAY = 0.995

def _compute_quality_score(self, memory: Memory, distance: float) -> float:
    """
    Compute quality-adjusted relevance score.

    Combines:
    - Semantic relevance (from embedding distance)
    - Recency (exponential decay)
    - Importance (LLM-assigned or computed)
    - Content quality (length, type)

    Based on WMR from Generative Agents paper.
    """
    # 1. Base relevance from distance (lower distance = higher score)
    relevance = 1.0 - min(distance, 2.0) / 2.0  # Normalize to 0-1

    # 2. Recency decay (human-like forgetting curve)
    age_hours = (datetime.now() - memory.created_at).total_seconds() / 3600
    recency = RECENCY_DECAY ** age_hours

    # 3. Importance (already computed in Memory model)
    importance = memory.importance_score

    # 4. Content quality signals
    length_factor = min(len(memory.content), 500) / 500  # 0-1
    type_factor = 1.2 if memory.memory_type == MemoryType.SEMANTIC else 1.0
    role_factor = 1.1 if "assistant" in memory.tags else 0.9  # Prefer assistant messages

    # Weighted combination (based on Generative Agents research)
    # α=0.5 relevance, β=0.3 recency, γ=0.2 importance
    base_score = (
        0.5 * relevance +
        0.3 * recency +
        0.2 * importance
    )

    # Apply quality multipliers
    return base_score * length_factor * type_factor * role_factor
```

#### 3.2 Sort Results by Quality Score

```python
def search_memories(self, query: str, n_results: int = 10, ...) -> List[Memory]:
    # ... existing retrieval code ...

    # After getting results_with_distance:
    scored_results = [
        (memory, self._compute_quality_score(memory, distance))
        for memory, distance in results_with_distance
    ]

    # Sort by quality score (descending)
    scored_results.sort(key=lambda x: x[1], reverse=True)

    return [memory for memory, _ in scored_results[:n_results]]
```

---

### Step 4: Update Psyche's Memory Retrieval

**File:** `src/psyche/core/memory_handler.py`

#### 4.1 Update retrieve_relevant()

```python
async def retrieve_relevant(
    self,
    query: str,
    n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Retrieve high-quality relevant memories."""
    ...
    memories = await self.mnemosyne_client.search_memories(
        query,
        n_results=n * 2,  # Fetch extra for quality filtering
        min_length=50,
        memory_types=["semantic", "episodic"],  # Exclude procedural/emotional for general recall
        exclude_tags=["user"],  # Exclude user questions
    )

    # Quality re-ranking done in Mnemosyne
    return memories[:n]
```

### Step 5: Database Cleanup Tool (Optional)

**File:** `src/mnemosyne/server.py`

Add new tool `cleanup_memories`:

```python
Tool(
    name="cleanup_memories",
    description="Remove low-quality memories from database",
    inputSchema={
        "type": "object",
        "properties": {
            "min_length": {"type": "integer", "default": 50},
            "remove_questions": {"type": "boolean", "default": True},
            "dry_run": {"type": "boolean", "default": True},
        },
    },
)
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/mnemosyne/storage/chroma_store.py` | Add BM25 index, hybrid search, quality scoring |
| `src/psyche/core/memory_handler.py` | Add `_should_store_message()`, update `store_messages()` |
| `src/mnemosyne/server.py` | Add `cleanup_memories` tool, expose hybrid search |
| `pyproject.toml` | Add `rank-bm25` dependency |

---

## Verification

### Test 1: Hybrid Search (Most Important)
```bash
# After implementing Step 1, test directly via Python:
from mnemosyne.storage.chroma_store import ChromaMemoryStore

store = ChromaMemoryStore("./data/memory")

# Query that previously returned the question instead of answer
results = store.search_memories_hybrid("Who is your mother", n_results=5)

# Verify:
# - Answer about Nyx should rank higher than the question itself
# - BM25 should boost results containing "Nyx", "mother" as answers
for r in results:
    print(f"{r.content[:100]}... (type={r.memory_type.value})")
```

### Test 2: Storage Filtering
```bash
# Start servers, have a conversation with questions
hermes --server http://localhost:8741

> What is the capital of France?
# (Psyche answers: "Paris")
> How do I make coffee?
# (Psyche answers with instructions)

# Check database - only assistant messages should be stored
# Short user questions should NOT be stored
```

### Test 3: Quality Ranking
```bash
# Compare results with and without quality scoring:
results_old = store.search_memories("your creator", n_results=5)  # Old method
results_new = store.search_memories_hybrid("your creator", n_results=5)  # New method

# Verify longer, semantic-type memories rank higher
```

### Test 4: End-to-End
```bash
# In a conversation with Psyche:
> Who created you?

# Server logs should show:
# - Hybrid search being used
# - Quality scores being computed
# - Answer memories ranking higher than questions
```

### Test 5: Cleanup Tool (Optional)
```bash
# Via MCP
cleanup_memories(dry_run=True, min_length=50)
# Review what would be deleted
cleanup_memories(dry_run=False)
```

---

## Estimated Sessions

| Task | Sessions |
|------|----------|
| Step 1: Hybrid search (BM25 + RRF) | 1.5 |
| Step 2: Storage filtering | 0.5 |
| Step 3: Quality-weighted ranking | 0.5 |
| Step 4: Cleanup tool (optional) | 0.5 |
| Testing and iteration | 1 |

**Total:** ~4 sessions

---

## Future Considerations (from Research)

### Near-term (Phase 8+)

1. **Cross-encoder reranking** - Use `ms-marco-MiniLM-L-12-v2` to rescore top-k results
2. **Entropy-aware filtering** (SimpleMem) - Compute `entity_novelty + semantic_divergence` before storing
3. **Memory consolidation** - Cluster similar memories and synthesize summaries
4. **Bi-temporal tracking** (Zep) - Distinguish event time vs ingestion time

### Long-term

5. **Graph-based memory** (Zep/Graphiti) - Entity extraction, relationship tracking, community detection
6. **RL-tuned memory** (Mem-α) - Reinforcement learning for optimal memory update sequences
7. **Reflective retrieval** (MemR3) - Evidence-gap tracking for iterative retrieval
8. **ColBERT late interaction** - Token-level similarity for better accuracy/speed tradeoff

### References

- Zep achieves 94.8% accuracy with graph + bi-temporal model
- SimpleMem achieves 30x token reduction with entropy filtering
- Mem0 achieves 26% accuracy boost with priority scoring
- Hybrid search reduces retrieval failure by 67% (Anthropic benchmark)
