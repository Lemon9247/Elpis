# Session Report: Phase 7 Hybrid Search Implementation

**Date:** 2026-01-24
**Type:** Implementation
**Topic:** Retrieval quality improvements (Part 5 of emotional depth project)

---

## Summary

Implemented the foundational retrieval quality improvements from the emotional depth project plan. This addresses the documented issue where questions outranked answers in pure vector search.

## What Was Built

### 1. Hybrid Search (BM25 + Vector + RRF)

**File:** `src/mnemosyne/storage/chroma_store.py`

- Added BM25 index that rebuilds on memory add/delete
- Implemented tokenization for keyword matching
- Created `_bm25_search()` method for keyword-based retrieval
- Implemented Reciprocal Rank Fusion (RRF) to combine rankings
- New `search_memories_hybrid()` method combines everything

The hybrid approach addresses the core problem: pure vector similarity finds "Who is your mother?" semantically similar to the query "Who is your mother" - but BM25 keyword matching will boost results containing actual answer content like "Nyx" or "mother".

### 2. Quality-Weighted Scoring

**File:** `src/mnemosyne/storage/chroma_store.py`

Added `_compute_quality_score()` that factors in:
- **Recency decay**: Exponential decay (~0.89 after 1 day, ~0.70 after 1 week)
- **Importance score**: LLM-computed or derived
- **Content length**: Longer content = more information
- **Memory type**: Semantic memories weighted 1.2x
- **Role factor**: Assistant responses weighted 1.1x, user messages 0.9x

### 3. Emotional Similarity for Mood-Congruent Retrieval

**File:** `src/mnemosyne/storage/chroma_store.py`

Added `_emotional_similarity()` using Euclidean distance in valence-arousal space:
- Identical emotions = similarity 1.0
- Opposite corners (excited vs depleted) = similarity ~0
- Missing context = neutral 0.5

Integrated into hybrid search as optional post-retrieval reranking.

### 4. Storage-Side Filtering

**File:** `src/psyche/core/memory_handler.py`

Added `_should_store_message()` to prevent low-quality memories:
- Skip messages under 50 characters
- Skip user questions (detected by ? or question-word starters)
- Automatically classify assistant declarative statements as semantic type

### 5. Server and Client Updates

**Files:** `src/mnemosyne/server.py`, `src/psyche/mcp/client.py`, `src/psyche/core/memory_handler.py`

- Updated `search_memories` tool schema with hybrid search options
- Added emotional_context and emotion_weight parameters
- MnemosyneClient now supports hybrid search parameters
- MemoryHandler auto-fetches emotional context for mood-congruent retrieval

### 6. Configuration

**Files:** `src/mnemosyne/config/settings.py`, `configs/mnemosyne.toml`

Added `RetrievalSettings` with configurable:
- emotion_weight (default 0.3)
- candidate_multiplier (default 2)
- vector_weight / bm25_weight (default 0.5 each)
- Quality scoring weights (recency, importance, relevance)

### 7. Tests

**File:** `tests/mnemosyne/unit/test_hybrid_search.py`

Unit tests for:
- Tokenization
- RRF combination
- Emotional similarity
- Quality scoring
- Storage filtering

---

## Commits Made

1. `Add unified implementation plan for emotional depth project`
2. `Add hybrid search with BM25 and quality scoring to Mnemosyne`
3. `Add storage-side filtering to prevent low-quality memories`
4. `Update Mnemosyne server to use hybrid search by default`
5. `Update clients to support hybrid search and mood-congruent retrieval`
6. `Add retrieval configuration settings for hybrid search`
7. `Add unit tests for hybrid search functionality`

---

## Design Decisions

### Why Post-Retrieval Reranking

Rather than modifying ChromaDB queries (which would filter rather than weight), we use post-retrieval reranking:
1. Retrieve more candidates (2x default)
2. Compute combined scores
3. Rerank and return top N

This is minimally invasive, flexible, and backward compatible.

### Why RRF Over Other Fusion Methods

Reciprocal Rank Fusion is:
- Simple to implement
- Doesn't require score normalization (works on ranks)
- Well-validated in IR literature
- Standard constant k=60 works well

### Why Hybrid Search as Default

Making hybrid search the default means:
- Existing code gets the improvement automatically
- The `use_hybrid=False` option preserves pure vector search if needed
- Quality scoring always applies (can be tuned via weights)

---

## What's Next

### Immediate (Part 1: Mood-Congruent Retrieval)

The emotional similarity infrastructure is already built into this implementation. What remains for Part 1:
- Tune emotion_weight based on testing
- Consider mood-congruence vs contrast during emotional spirals

### Future Parts

- **Part 2 (Dreaming)**: Build on mood-congruent retrieval for emotion-shaped dreams
- **Part 3 (Trajectory)**: Add emotional history tracking
- **Part 4 (PAD Model)**: Extend to 3D emotional space

---

## Open Questions

1. **BM25 Index Rebuild Performance**: Currently rebuilds on every add/delete. May need incremental updates if this becomes a bottleneck with large databases.

2. **Quality Score Weights**: Current defaults are reasonable guesses. Should tune based on actual retrieval results.

3. **Emotion Weight Tuning**: 0.3 default means 70% semantic, 30% emotional. May want to adjust based on use case.

---

## Files Changed

| File | Lines | Summary |
|------|-------|---------|
| `pyproject.toml` | +1 | rank-bm25 dependency |
| `src/mnemosyne/storage/chroma_store.py` | +372 | Hybrid search, BM25, quality scoring |
| `src/mnemosyne/server.py` | +46 | Tool schema updates, hybrid search handler |
| `src/mnemosyne/config/settings.py` | +41 | RetrievalSettings |
| `src/psyche/core/memory_handler.py` | +64 | Storage filtering, auto-emotion |
| `src/psyche/mcp/client.py` | +19 | Hybrid search parameters |
| `configs/mnemosyne.toml` | +13 | Retrieval config section |
| `tests/mnemosyne/unit/test_hybrid_search.py` | +334 | Unit tests |

---

## For Future Sessions

The retrieval quality foundation is complete. The next session could:

1. **Test in practice**: Run Psyche with the new hybrid search and verify questions no longer outrank answers
2. **Begin Part 2**: Implement emotion-shaped dreaming
3. **Begin Part 3**: Add trajectory tracking

The infrastructure for mood-congruent retrieval (Part 1) is already integrated into this implementation, so Parts 2-3 can proceed without additional retrieval work.
