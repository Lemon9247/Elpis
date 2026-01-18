# Research Report: LLM Memory Systems

**Date:** 2026-01-18
**Topic:** Memory architectures, retrieval techniques, and quality improvements for LLM agents
**Purpose:** Inform Phase 7 (Memory Retrieval Quality) improvements for Psyche/Mnemosyne

---

## Executive Summary

Modern LLM memory systems have evolved significantly beyond simple vector search. Key innovations include:

1. **Multi-modal retrieval** - Combining semantic embeddings, keyword search (BM25), and graph traversal
2. **Importance scoring** - Weighting memories by recency, relevance, and importance
3. **Memory consolidation** - Compressing and abstracting memories over time
4. **Temporal awareness** - Tracking when facts were true vs when they were recorded
5. **Quality filtering** - Entropy-aware filtering to prevent memory pollution

This research identifies several techniques applicable to Mnemosyne that could significantly improve retrieval quality.

---

## State-of-the-Art Memory Architectures

### Zep (Temporal Knowledge Graph)

**Source:** [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956)

Zep is the current state-of-the-art, outperforming MemGPT on benchmarks with **94.8% accuracy** and **90% latency reduction**.

**Architecture:**
- **Episode Subgraph** - Raw conversational data stored non-lossy
- **Semantic Entity Subgraph** - Extracted entities and relationships
- **Community Subgraph** - Clusters of related entities with summaries

**Key Innovations:**
1. **Bi-temporal modeling** - Tracks both event time (when fact was true) and ingestion time (when recorded)
2. **Triple retrieval method:**
   - Cosine semantic similarity
   - BM25 full-text search
   - Breadth-first graph traversal
3. **Contradiction detection** - Invalidates edges when new information conflicts

**Applicable to Mnemosyne:** The bi-temporal model could help with memory versioning. The triple retrieval approach (semantic + BM25 + graph) is directly applicable.

---

### Mem0 (Scalable Memory Layer)

**Source:** [Mem0: Building Production-Ready AI Agents](https://arxiv.org/abs/2504.19413)

Mem0 achieves **26% accuracy improvement**, **91% lower latency**, and **90% token savings**.

**Key Features:**
- **Memory scopes:** User memory (cross-session), Session memory, Agent memory
- **Dynamic extraction:** Automatically identifies salient information
- **Graph-based variant:** Captures relational structures between concepts
- **Priority scoring:** Decides what gets stored to prevent bloat

**Memory Types Supported:**
- Episodic (events)
- Semantic (facts)
- Procedural (how-to)
- Associative (relationships)

**Applicable to Mnemosyne:** Priority scoring for what to store. Memory scopes could separate user facts from conversation events.

---

### SimpleMem (Efficient Lifelong Memory)

**Source:** [SimpleMem: Efficient Lifelong Memory for LLM Agents](https://arxiv.org/html/2601.02553v1)

Achieves **30x token reduction** while maintaining accuracy.

**Three-Stage Pipeline:**

1. **Semantic Structured Compression**
   - Entropy-aware filtering (threshold τ = 0.35)
   - Combines entity-level novelty + semantic divergence
   - Coreference resolution (pronouns → explicit names)
   - Temporal anchoring (relative time → ISO timestamps)

2. **Recursive Memory Consolidation**
   - Affinity scoring: semantic similarity + temporal proximity
   - Clustering related memories (threshold τ = 0.85)
   - Synthesizes into higher-level abstractions

3. **Adaptive Query-Aware Retrieval**
   - Hybrid: dense embeddings + BM25 + metadata constraints
   - Dynamic retrieval depth (k=3 for simple, k=20 for complex)

**Applicable to Mnemosyne:** Entropy-aware filtering is exactly what we need. The consolidation approach could replace current promotion-based system.

---

### MemGPT / Letta (Two-Tier Architecture)

**Source:** [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)

Inspired by OS virtual memory management.

**Architecture:**
- **Tier 1 (Main Context):** Core memories always in context
- **Tier 2 (External):**
  - Recall storage (recent, searchable)
  - Archival storage (long-term, indexed)

**Key Concept:** LLM manages its own memory through function calls, deciding when to store/retrieve.

**Applicable to Mnemosyne:** Already partially implemented via Psyche's tool-based memory access.

---

## Retrieval Quality Techniques

### Hybrid Search (Industry Standard)

**Sources:**
- [Optimizing RAG with Hybrid Search & Reranking](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
- [Qdrant Hybrid Search Documentation](https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/)

**The Problem:**
- Vector search captures semantic meaning but misses exact keywords
- BM25 captures keywords but misses semantic similarity
- Neither alone is sufficient for high-quality retrieval

**The Solution:**
```
Dense Top-K  ─┐
              ├─→ RRF Fusion ─→ Cross-Encoder Rerank ─→ Final Results
BM25 Top-K   ─┘
```

**Techniques:**
1. **Reciprocal Rank Fusion (RRF):** Combines ranked lists from different retrievers
2. **Cross-Encoder Reranking:** Scores query-document pairs directly (slower but more accurate)
3. **ColBERT (Late Interaction):** Token-level similarity, good accuracy/speed tradeoff

**Benchmarks:**
- Anthropic found hybrid + reranking reduced retrieval failure by 67%
- Cross-encoders like `ms-marco-MiniLM-L-12-v2` significantly improve NDCG/MRR

**Applicable to Mnemosyne:** Add BM25 alongside vector search. Consider cross-encoder reranking for top results.

---

### Importance Scoring (Generative Agents)

**Source:** [Park et al. - Generative Agents](https://arxiv.org/abs/2304.03442)

**Weighted Memory Retrieval (WMR):**
```
score = α * recency + β * importance + γ * relevance
```

Where:
- **Recency:** Exponential decay (0.995 per hour)
- **Importance:** LLM-generated significance score (1-10)
- **Relevance:** Semantic similarity to current query

**Applicable to Mnemosyne:** Already have importance_score field. Need to incorporate into retrieval ranking.

---

### Memory Decay (Human-like Recall)

**Source:** [Integrating Dynamic Human-like Memory Recall](https://arxiv.org/html/2404.00573v1)

**Mathematical Model:**
```
p(t) = 1 - exp(-r * e^(-t/g_n))
```

Where:
- `r` = relevance (cosine similarity)
- `t` = elapsed time since memory creation
- `g_n` = decay constant (increases with each recall)

**Spacing Effect:** Memories recalled over longer intervals strengthen more than frequent short-interval recalls.

**Threshold-based Retrieval:** Only retrieve when probability > 0.86

**Applicable to Mnemosyne:** Implement decay-based scoring. Track access_count and last_accessed (already in Memory model).

---

### MemR3 (Reflective Reasoning)

**Source:** [MemR3: Memory Retrieval via Reflective Reasoning](https://arxiv.org/html/2512.20237v1)

**Key Innovation: Evidence-Gap Tracking**
- Maintains explicit state: what is known (evidence) vs unknown (gaps)
- Triggers targeted retrieval for gaps rather than generic search

**Closed-Loop Router:**
- Dynamically switches between Retrieve, Reflect, Answer
- Iterative query reformulation based on identified gaps
- Early termination when sufficient evidence found

**Results:** +7.29% over standard RAG

**Applicable to Mnemosyne:** Consider iterative retrieval with gap analysis for complex queries.

---

## Memory Quality Filtering

### Entropy-Aware Filtering (SimpleMem)

**Criteria for Storing:**
```
info_score = entity_novelty + semantic_divergence
store if info_score > τ_redundant (0.35)
```

This prevents:
- Duplicate information
- Low-information content
- Questions (high similarity to stored questions, low novelty)

### What NOT to Store

Based on research consensus:
1. **Questions** - Echo the query, don't provide answers
2. **Short utterances** - "Yes", "OK", "Thanks" add noise
3. **Repeated information** - Detected via semantic similarity
4. **Temporary context** - "Right now I'm..." vs persistent facts

### Quality Signals

| Signal | Description | Weight |
|--------|-------------|--------|
| Content length | Longer = more informative | Medium |
| Entity density | More entities = more facts | High |
| Declarative form | "X is Y" vs "Is X Y?" | High |
| Role | Assistant > User for knowledge | High |
| Memory type | Semantic > Episodic for recall | Medium |

---

## Recommendations for Mnemosyne

### Priority 1: Hybrid Search
- Add BM25 full-text search alongside vector embeddings
- Use RRF to combine results
- **Impact:** ~30-67% improvement in retrieval quality

### Priority 2: Storage Filtering
- Minimum content length (50+ chars)
- Skip user questions (detect by `?` or question words)
- Compute entity novelty before storing
- **Impact:** Prevents future pollution

### Priority 3: Quality-Weighted Ranking
```python
def quality_score(memory, distance):
    base = 1.0 - min(distance, 2.0) / 2.0

    # Length bonus (up to 500 chars)
    length_bonus = min(len(memory.content), 500) / 500 * 0.2

    # Type bonus
    type_bonus = 0.3 if memory.type == "semantic" else 0.0

    # Recency factor
    age_hours = (now - memory.created_at).total_seconds() / 3600
    recency = 0.995 ** age_hours

    # Importance contribution
    importance = memory.importance_score * 0.2

    return base + length_bonus + type_bonus + recency * 0.1 + importance
```

### Priority 4: Memory Consolidation
- Cluster similar memories periodically
- Synthesize into higher-level summaries
- Delete superseded individual memories
- **Impact:** Reduces database size, improves search

### Future: Graph-Based Memory
- Extract entities and relationships from text
- Enable graph traversal for connected concepts
- Track temporal validity of facts
- **Impact:** State-of-the-art performance (Zep achieves 94.8%)

---

## Implementation Complexity

| Technique | Complexity | Dependencies | Sessions |
|-----------|------------|--------------|----------|
| BM25 hybrid search | Medium | rank_bm25 or similar | 1-2 |
| Storage filtering | Low | None | 0.5 |
| Quality ranking | Low | None | 0.5 |
| Cross-encoder reranking | Medium | sentence-transformers | 1 |
| Memory consolidation | High | Clustering library | 2-3 |
| Graph-based memory | Very High | Neo4j or similar | 5+ |

---

## Conclusion

The field has moved beyond simple vector search. Key insights:

1. **Hybrid search is now standard** - Combining semantic + keyword + metadata
2. **What you store matters** - Entropy-aware filtering prevents pollution
3. **Ranking needs multiple signals** - Relevance, recency, importance, type
4. **Consolidation is essential** - Prevents fragmentation and redundancy

For Mnemosyne, the highest-impact improvements are:
1. Add BM25 to search (complements embeddings)
2. Filter storage quality (prevent questions/short messages)
3. Quality-weighted ranking (not just distance)

These can be implemented in ~3 sessions and should significantly improve Psyche's memory recall.

---

## Sources

- [Memory in the Age of AI Agents Survey](https://arxiv.org/abs/2512.13564)
- [Zep: Temporal Knowledge Graph Architecture](https://arxiv.org/abs/2501.13956)
- [Mem0: Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413)
- [SimpleMem: Efficient Lifelong Memory](https://arxiv.org/html/2601.02553v1)
- [MemGPT: LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- [MemR3: Reflective Reasoning Retrieval](https://arxiv.org/html/2512.20237v1)
- [Human-like Memory Recall in LLM Agents](https://arxiv.org/html/2404.00573v1)
- [LangMem Conceptual Guide](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)
- [Optimizing RAG with Hybrid Search](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
- [Qdrant Hybrid Search Tutorial](https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/)
