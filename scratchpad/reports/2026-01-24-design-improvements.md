# Session Report: Design Improvements for Emotional Depth

**Date**: 2026-01-24
**Branch**: `claude/review-emotional-depth-plan-bnnJp`
**Commits**: 7 commits (bb3538d through 2b65262)

## Overview

This session implemented design improvements identified in a code review of the emotional depth features. The focus was on performance, configurability, and nuanced behavior.

## Improvements Implemented

### 1. BM25 Lazy Rebuilding (Performance)

**Problem**: `_rebuild_bm25_index()` was called on every `add_memory()` and `delete_memory()`, causing O(n) operations for each change.

**Solution**:
- Added `_bm25_dirty` flag
- Index only rebuilds on search when dirty
- Batch additions/deletions only trigger single rebuild

**Files**: `src/mnemosyne/storage/chroma_store.py`

### 2. Eliminate Double-Fetch in Hybrid Search (Performance)

**Problem**: After vector search returned IDs, we queried ChromaDB again for each memory.

**Solution**:
- Include documents/metadatas in initial vector query
- Build memory cache from query results
- Only fetch via `get_memory()` for BM25-only results

**Files**: `src/mnemosyne/storage/chroma_store.py`

### 3. Configurable Quality Scoring (Maintainability)

**Problem**: Quality scoring weights and factors were hardcoded in methods.

**Solution**: Added to `RetrievalSettings`:
- `recency_decay_rate`, `max_recency_hours`
- `max_content_length`
- `semantic_type_factor`, `assistant_role_factor`, `user_role_factor`
- `use_angular_similarity`

**Files**: `src/mnemosyne/config/settings.py`, `configs/mnemosyne.toml`

### 4. Angular Emotional Similarity (Accuracy)

**Problem**: Euclidean distance treats (0.8, 0.8) and (0.4, 0.4) as different, but they're the same emotional direction.

**Solution**:
- Added `use_angular_similarity` config option
- Cosine similarity focuses on direction, not magnitude
- Configurable per deployment

**Files**: `src/mnemosyne/storage/chroma_store.py`

### 5. Configurable Trajectory Thresholds (Maintainability)

**Problem**: Trajectory detection used magic numbers (0.01, 0.02, etc).

**Solution**: Added to `EmotionSettings`:
- `trajectory_history_size`
- `momentum_positive_threshold`, `momentum_negative_threshold`
- `trend_improving_threshold`, `trend_declining_threshold`
- `spiral_history_count`, `spiral_increasing_threshold`

**Files**: `src/elpis/config/settings.py`, `src/elpis/emotion/state.py`, `configs/elpis.toml`

### 6. Direction-Aware Spiral Detection (Accuracy)

**Problem**: Spiral detection only detected presence, not direction (positive vs negative).

**Solution**:
- `_detect_spiral()` now returns `(bool, str)` tuple
- Directions: "positive", "negative", "escalating", "withdrawing", "none"
- Enables different responses based on spiral type

**Files**: `src/elpis/emotion/state.py`

### 7. Improved Storage Filtering (Accuracy)

**Problem**: Simplistic filtering missed nuanced cases.

**Solution**:
- Added `HIGH_VALUE_PATTERNS` (definitions, reasoning, preferences)
- Added `LOW_VALUE_PATTERNS` (greetings, acknowledgments)
- Long questions with context now stored (valuable for recall)
- Semantic memory assignment for high-value content

**Files**: `src/psyche/core/memory_handler.py`

### 8. Dynamic Dream Query Generation (Personalization)

**Problem**: Dreams always used static queries per quadrant.

**Solution**:
- `TRAJECTORY_QUERY_MODIFIERS` map spiral/trend to queries
- `_generate_dynamic_queries()` adapts to emotional trajectory
- `_extract_recent_topics()` extracts topics from recent memories
- Configurable via `DreamConfig.use_dynamic_queries`

**Files**: `src/psyche/handlers/dream_handler.py`

### 9. Integration Tests (Reliability)

Added comprehensive tests:
- `tests/mnemosyne/integration/test_emotional_depth.py`
  - Hybrid search integration
  - Quality scoring with config
  - Angular vs Euclidean similarity
  - Storage filtering patterns

- `tests/elpis/integration/test_emotional_trajectory.py`
  - TrajectoryConfig from settings
  - Momentum/trend detection
  - Direction-aware spiral detection
  - Quadrant time tracking

## Architecture Notes

### TrajectoryConfig Pattern

Created `TrajectoryConfig` dataclass to hold trajectory detection thresholds. This allows:
1. Default values in the dataclass
2. Factory method `from_settings()` to load from `EmotionSettings`
3. Easy testing with custom thresholds

### Memory Cache in Hybrid Search

The memory cache pattern in `search_memories_hybrid`:
```python
memory_cache: Dict[str, Memory] = {}
# Populate from vector search results
for memory_id, distance in vector_results:
    memory = self._query_result_to_memory(results, 0, i)
    memory_cache[memory_id] = memory
# Use cache or fallback to get_memory for BM25-only results
```

This eliminates N additional queries while maintaining correctness for BM25-only matches.

## Open Questions

1. **Angular vs Euclidean**: Should angular similarity be the default? It better captures emotional direction but ignores intensity.

2. **Spiral Response**: Now that we detect spiral direction, what should happen?
   - Positive spirals: Log and let continue
   - Negative spirals: Trigger homeostatic intervention?
   - Escalating: Suggest calming?

3. **Topic Extraction**: Current keyword extraction is basic. Could use TF-IDF or even LLM-based extraction for better topics.

## Summary

7 commits implementing performance optimizations, configurability improvements, and accuracy enhancements. All changes are backward-compatible with existing configurations.
