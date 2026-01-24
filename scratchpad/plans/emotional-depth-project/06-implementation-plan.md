# Emotional Depth Project - Implementation Plan

## Overview

This document outlines the implementation approach for the Emotional Depth Project, starting with retrieval quality improvements and building toward full emotional integration.

**Total Scope:** 17-22 sessions across 5 parts
**Starting Point:** Part 5 (Phase 7 Quality Improvements)
**Rationale:** Foundational retrieval quality enables all subsequent emotional features

---

## Implementation Phases

```
Phase A: Foundation (This Plan)
├── Part 5: Retrieval Quality     [~4 sessions]  ← START HERE
└── Part 1: Mood-Congruent        [4-5 sessions]

Phase B: Temporal Awareness
└── Part 3: Trajectory Tracking   [2-3 sessions]

Phase C: Responsive Dreams
└── Part 2: Emotion-Shaped Dreams [2 sessions]

Phase D: Extended Model (Optional)
└── Part 4: PAD Model             [5-8 sessions]
```

**Why this order:**
1. Part 5 fixes retrieval quality issues that currently degrade all memory operations
2. Part 1 layers emotional awareness onto quality retrieval
3. Part 3 can be done independently but informs Parts 1 and 2
4. Part 2 benefits from both mood-congruent retrieval and trajectory
5. Part 4 is a larger refactor, deferred until 2D model feels limiting

---

## Part 5: Retrieval Quality Improvements

### Problem Summary

Memory retrieval returns poor results due to:
1. **Questions outrank answers** - semantic similarity favors "Who is your mother?" over actual answers about mothers
2. **No keyword matching** - pure vector search misses exact term matches
3. **No quality filtering** - short snippets and user questions pollute results
4. **No quality weighting** - all memories treated equally regardless of type, length, role

### Solution Components

| Component | Impact | Effort |
|-----------|--------|--------|
| Hybrid search (BM25 + vector + RRF) | High | 1.5 sessions |
| Storage-side filtering | Medium | 0.5 sessions |
| Quality-weighted ranking | Medium | 0.5 sessions |
| Testing and iteration | - | 1 session |

---

### Session 5.1: Hybrid Search Infrastructure

**Goal:** Add BM25 keyword search alongside vector search, combine via Reciprocal Rank Fusion

#### Tasks

1. **Add rank-bm25 dependency**
   - File: `pyproject.toml`
   - Add `rank-bm25` to dependencies

2. **Add BM25 index to ChromaMemoryStore**
   - File: `src/mnemosyne/storage/chroma_store.py`
   - Add `_bm25_index` and `_bm25_corpus` fields
   - Add `_tokenize()` method for text preprocessing
   - Add `_rebuild_bm25_index()` method
   - Call rebuild on init and after add/delete operations

3. **Implement BM25 search method**
   - File: `src/mnemosyne/storage/chroma_store.py`
   - Add `_bm25_search(query, n_results)` method
   - Return list of (memory_id, bm25_score) tuples

4. **Implement Reciprocal Rank Fusion**
   - File: `src/mnemosyne/storage/chroma_store.py`
   - Add `_reciprocal_rank_fusion(vector_results, bm25_results, k=60)` method
   - Combine rankings: `score = sum(1/(k + rank) for each list)`

5. **Create hybrid search method**
   - File: `src/mnemosyne/storage/chroma_store.py`
   - Add `search_memories_hybrid()` method
   - Parameters: `query`, `n_results`, `vector_weight`, `bm25_weight`
   - Retrieve 2x candidates from each method, combine via RRF

#### Verification
```python
# Test that "Who is your mother" returns answer before question
store = ChromaMemoryStore("./data/memory")
results = store.search_memories_hybrid("Who is your mother", n_results=5)
# Answer containing "Nyx" should rank higher than the question itself
```

---

### Session 5.2: Storage Filtering and Quality Scoring

**Goal:** Prevent low-quality memories from being stored, add quality-weighted ranking

#### Tasks

1. **Add content quality assessment**
   - File: `src/psyche/core/memory_handler.py`
   - Add `_should_store_message(msg)` method
   - Skip messages under 50 characters
   - Skip user questions (detect by "?" or question words)
   - Return `(should_store: bool, memory_type: str)`

2. **Update store_messages() to filter**
   - File: `src/psyche/core/memory_handler.py`
   - Call `_should_store_message()` before storing
   - Log skipped messages at debug level
   - Use returned memory_type for smarter categorization

3. **Add quality scoring to ChromaStore**
   - File: `src/mnemosyne/storage/chroma_store.py`
   - Add `_compute_quality_score(memory, distance)` method
   - Factors: semantic relevance, recency decay, importance, content length, memory type, role

4. **Integrate quality scoring into hybrid search**
   - File: `src/mnemosyne/storage/chroma_store.py`
   - After RRF combination, apply quality score multiplier
   - Final ranking: RRF score * quality score

5. **Add configuration for quality weights**
   - File: `src/mnemosyne/config/settings.py`
   - Add `RetrievalSettings` with configurable weights
   - Defaults: relevance=0.5, recency=0.3, importance=0.2

#### Verification
```bash
# Test storage filtering
# Have conversation with short messages and questions
# Verify only substantive assistant responses are stored

# Test quality ranking
# Query for information, verify longer/semantic memories rank higher
```

---

### Session 5.3: Integration and Wire-Up

**Goal:** Make hybrid search the default, update server and clients

#### Tasks

1. **Update Mnemosyne server tool**
   - File: `src/mnemosyne/server.py`
   - Update `search_memories` tool to use hybrid search by default
   - Add optional parameters: `use_hybrid`, `vector_weight`, `bm25_weight`
   - Backward compatible: existing calls work unchanged

2. **Update MnemosyneClient**
   - File: `src/psyche/mcp/client.py`
   - Update `search_memories()` to pass new parameters
   - Default to hybrid search

3. **Update memory_handler retrieval**
   - File: `src/psyche/core/memory_handler.py`
   - Update `retrieve_relevant()` to use hybrid search
   - Pass quality filtering preferences

4. **Add cleanup tool (optional)**
   - File: `src/mnemosyne/server.py`
   - Add `cleanup_memories` tool for removing low-quality existing memories
   - Parameters: `min_length`, `remove_questions`, `dry_run`

#### Verification
```bash
# End-to-end test via Psyche
psyche
> Who created you?
# Should retrieve factual answer, not the question itself

# Check server logs for hybrid search usage
```

---

### Session 5.4: Testing and Refinement

**Goal:** Comprehensive testing, tune weights, handle edge cases

#### Tasks

1. **Unit tests for BM25 and RRF**
   - File: `tests/mnemosyne/unit/test_hybrid_search.py`
   - Test tokenization
   - Test BM25 scoring
   - Test RRF combination
   - Test quality scoring components

2. **Integration tests for retrieval quality**
   - File: `tests/mnemosyne/integration/test_retrieval_quality.py`
   - Test questions vs answers ranking
   - Test short vs long content ranking
   - Test semantic vs episodic type ranking
   - Test recency decay

3. **Tune weights based on results**
   - Adjust vector_weight vs bm25_weight
   - Adjust quality score factors
   - Document final tuned values

4. **Edge case handling**
   - Empty BM25 index (no memories yet)
   - Very short queries
   - Non-ASCII content
   - Large result sets

---

## Part 1: Mood-Congruent Retrieval

*Builds on Part 5's quality improvements*

### Problem Summary

Emotional state influences memory storage but not retrieval. Memory recall should be mood-congruent - current emotional state should act as a retrieval cue.

### Solution: Post-Retrieval Emotional Reranking

After hybrid search returns quality-ranked results, rerank by emotional similarity to current state.

---

### Session 1.1: Emotional Similarity Infrastructure

**Goal:** Add emotional similarity computation to ChromaStore

#### Tasks

1. **Add emotional similarity function**
   - File: `src/mnemosyne/storage/chroma_store.py`
   - Add `_emotional_similarity(query_emotion, memory_emotion)` method
   - Euclidean distance in valence-arousal space, normalized to [0,1]
   - Handle missing emotional context (return 0.5 neutral)

2. **Add combined scoring function**
   - File: `src/mnemosyne/storage/chroma_store.py`
   - Add `_combined_score(quality_score, emotional_similarity, weights)` method
   - Combine quality (from Part 5) with emotional similarity

3. **Add emotional reranking method**
   - File: `src/mnemosyne/storage/chroma_store.py`
   - Add `_rerank_by_emotion(results, query_emotion, emotion_weight)` method
   - Recompute scores incorporating emotional similarity
   - Sort by combined score

---

### Session 1.2: Search Method Updates

**Goal:** Integrate emotional context into search API

#### Tasks

1. **Update search_memories signature**
   - File: `src/mnemosyne/storage/chroma_store.py`
   - Add `emotional_context: Optional[EmotionalContext]` parameter
   - Add `emotion_weight: float = 0.3` parameter
   - When emotional_context provided, retrieve 2x candidates then rerank

2. **Update Mnemosyne server tool schema**
   - File: `src/mnemosyne/server.py`
   - Add `emotional_context` object to inputSchema
   - Add `emotion_weight` parameter
   - Update handler to parse and pass emotional context

3. **Update MnemosyneClient**
   - File: `src/psyche/mcp/client.py`
   - Add `emotional_context` and `emotion_weight` parameters
   - Pass through to tool call

---

### Session 1.3: Automatic Emotional Context

**Goal:** Psyche automatically includes current emotion in memory queries

#### Tasks

1. **Update memory_handler to fetch emotion**
   - File: `src/psyche/core/memory_handler.py`
   - In `retrieve_relevant()`, fetch current emotional state from Elpis
   - Pass emotional context to search call
   - Graceful degradation if Elpis unavailable

2. **Add configuration**
   - File: `src/mnemosyne/config/settings.py`
   - Add `emotion_weight` to RetrievalSettings
   - Add `candidate_multiplier` for reranking pool size

3. **Update configs**
   - File: `configs/mnemosyne.toml`
   - Add `[retrieval]` section with emotion settings

---

### Session 1.4-1.5: Testing

**Goal:** Verify mood-congruent retrieval works correctly

#### Tasks

1. **Unit tests for emotional similarity**
   - File: `tests/mnemosyne/unit/test_emotional_similarity.py`
   - Test identical emotions → similarity 1.0
   - Test opposite corners → similarity ~0
   - Test missing context → returns 0.5

2. **Integration tests for mood-congruent retrieval**
   - File: `tests/mnemosyne/integration/test_mood_congruent_retrieval.py`
   - Store memories with varied emotional contexts
   - Query with specific emotional state
   - Verify emotionally similar memories rank higher

3. **End-to-end verification**
   - Set Psyche to specific emotional state
   - Make memory queries
   - Verify results reflect emotional congruence

---

## Part 3: Trajectory Tracking

*Independent of Parts 1-2, but enhances both*

### Problem Summary

Current system knows emotional position but not direction. Trajectory enables:
- Proactive intervention before reaching depleted states
- Spiral detection to avoid reinforcing negative loops
- More nuanced mood-congruent retrieval

### Sessions

| Session | Focus |
|---------|-------|
| 3.1 | History tracking, velocity computation |
| 3.2 | Trend/spiral detection, server exposure |
| 3.3 | Integration with retrieval and dreams |

---

## Part 2: Emotion-Shaped Dreaming

*Benefits from Parts 1 and 3*

### Problem Summary

Dreams currently use generic prompts and random memory retrieval. They should respond to emotional needs - a dream when frustrated should seek different memories than one when depleted.

### Sessions

| Session | Focus |
|---------|-------|
| 2.1 | DreamIntention dataclass, quadrant mappings |
| 2.2 | Update dream methods, testing |

---

## Part 4: Extended Emotional Model (PAD)

*Deferred until 2D model feels limiting*

### Problem Summary

Some emotional states don't map cleanly to 2D - feeling overwhelmed vs in-control are both high arousal but qualitatively different.

### Sessions

| Session | Focus |
|---------|-------|
| 4.1 | Add dominance to EmotionalState |
| 4.2 | Update event mappings to 3D |
| 4.3 | Update emotional similarity to 3D |
| 4.4 | Update dreaming for 8 octants |
| 4.5-4.7 | Train octant steering vectors (optional) |

---

## File Change Summary

### Part 5 Changes

| File | Changes |
|------|---------|
| `pyproject.toml` | Add rank-bm25 dependency |
| `src/mnemosyne/storage/chroma_store.py` | BM25 index, hybrid search, RRF, quality scoring |
| `src/mnemosyne/server.py` | Update tool, add cleanup tool |
| `src/mnemosyne/config/settings.py` | Add RetrievalSettings |
| `src/psyche/core/memory_handler.py` | Storage filtering, retrieval updates |
| `src/psyche/mcp/client.py` | Update search_memories signature |
| `configs/mnemosyne.toml` | Add retrieval config section |

### Part 1 Changes (builds on Part 5)

| File | Changes |
|------|---------|
| `src/mnemosyne/storage/chroma_store.py` | Emotional similarity, reranking |
| `src/mnemosyne/server.py` | Emotional context in tool schema |
| `src/psyche/core/memory_handler.py` | Auto-include emotion in queries |
| `src/psyche/mcp/client.py` | Emotional context parameters |

---

## Open Design Questions

1. **Score combination**: Multiplicative or additive for quality + emotion?
   - Current plan: `final = quality_score * (1 + emotion_weight * emotion_similarity)`
   - Alternative: `final = (1 - emotion_weight) * quality_score + emotion_weight * emotion_similarity`

2. **Mood-congruence vs contrast**: Always seek similar emotions, or seek contrast during spirals?
   - Initial: Always congruent
   - Future: Use trajectory to detect spirals, reduce congruence or seek contrast

3. **BM25 index rebuild strategy**: Full rebuild vs incremental?
   - Initial: Full rebuild on add/delete (simpler)
   - Future: Incremental updates if performance becomes an issue

4. **Quality score weights**: What's the right balance?
   - Initial: relevance=0.5, recency=0.3, importance=0.2
   - Tune based on testing

---

## Success Criteria

### Part 5 Complete When:
- [ ] Hybrid search implemented and tested
- [ ] Questions no longer outrank answers in retrieval
- [ ] Storage filtering prevents low-quality memories
- [ ] Quality scoring improves result relevance
- [ ] All tests passing

### Part 1 Complete When:
- [ ] Emotional similarity computed correctly
- [ ] Retrieval incorporates current emotional state
- [ ] Emotionally similar memories rank higher
- [ ] Backward compatible (works without emotional context)
- [ ] All tests passing

---

## Getting Started

Begin with **Session 5.1: Hybrid Search Infrastructure**

1. Add rank-bm25 dependency
2. Implement BM25 index in ChromaStore
3. Implement RRF combination
4. Create hybrid search method
5. Verify questions no longer outrank answers

This provides the foundation for all subsequent improvements.
