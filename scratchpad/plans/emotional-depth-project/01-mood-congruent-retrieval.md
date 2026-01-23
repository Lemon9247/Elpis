# Part 1: Mood-Congruent Memory Retrieval

## Problem Statement

Emotional state flows *into* memory storage (emotionally intense moments get remembered, emotional context travels with the memory) but doesn't flow *out* during retrieval. Current mood doesn't shape what surfaces.

This breaks the loop that makes memory feel experiential. Human memory is mood-congruent - emotional state acts as a retrieval cue.

## Approach: Post-Retrieval Reranking

Rather than modifying ChromaDB queries (which would filter rather than weight), use post-retrieval reranking:

1. Retrieve more candidates than requested (2x by default)
2. Compute combined score: semantic similarity + emotional similarity
3. Rerank by combined score
4. Return top N

**Why this approach:**
- Minimally invasive to existing code
- Flexible (tunable weights)
- Doesn't require schema changes
- Can be toggled on/off
- Backward compatible (works unchanged when no emotional context provided)

---

## Technical Design

### Emotional Similarity Metric

Euclidean distance in valence-arousal space, normalized to [0,1]:

```python
def emotional_similarity(
    query_emotion: EmotionalContext,
    memory_emotion: EmotionalContext
) -> float:
    """
    Compute similarity between two emotional states.

    Returns:
        Float 0-1, higher = more similar
    """
    if not query_emotion or not memory_emotion:
        return 0.5  # Neutral if either missing

    valence_diff = query_emotion.valence - memory_emotion.valence
    arousal_diff = query_emotion.arousal - memory_emotion.arousal
    distance = (valence_diff**2 + arousal_diff**2) ** 0.5

    # Max possible distance in 2D space [-1,1] x [-1,1] is sqrt(8) ≈ 2.83
    max_distance = 2.83
    similarity = 1.0 - (distance / max_distance)
    return similarity
```

### Combined Scoring

```python
def combined_score(
    semantic_distance: float,
    emotional_similarity: float,
    semantic_weight: float = 0.7,
    emotion_weight: float = 0.3,
) -> float:
    """
    Combine semantic and emotional scores.

    Args:
        semantic_distance: ChromaDB L2 distance (lower = more similar)
        emotional_similarity: 0-1 similarity (higher = more similar)
        semantic_weight: Weight for semantic component
        emotion_weight: Weight for emotional component

    Returns:
        Combined score (higher = better match)
    """
    # Convert distance to similarity
    # Normalize assuming max useful distance ~2.0
    semantic_similarity = max(0.0, 1.0 - semantic_distance / 2.0)

    return (semantic_weight * semantic_similarity) + (emotion_weight * emotional_similarity)
```

---

## Implementation Changes

### File: `src/mnemosyne/storage/chroma_store.py`

**Modify `search_memories()` signature:**

```python
def search_memories(
    self,
    query: str,
    n_results: int = 10,
    status_filter: Optional[MemoryStatus] = None,
    emotional_context: Optional[EmotionalContext] = None,  # NEW
    emotion_weight: float = 0.3,  # NEW
) -> List[Memory]:
    """
    Semantic search for memories with optional emotional weighting.

    Args:
        query: Search query
        n_results: Number of results to return
        status_filter: Filter by memory status
        emotional_context: Current emotional state for mood-congruent retrieval
        emotion_weight: Weight for emotional similarity (0-1, default 0.3)

    Returns:
        List of memories, ranked by combined semantic-emotional score
    """
```

**Add retrieval multiplier when reranking:**

```python
# If emotional context provided, retrieve more candidates for reranking
retrieve_n = n_results * 2 if emotional_context else n_results
```

**Add reranking logic:**

```python
# After collecting results_with_distance...

if emotional_context and emotion_weight > 0:
    results_with_distance = self._rerank_by_emotion(
        results_with_distance,
        emotional_context,
        emotion_weight,
    )

# Return top N
return [m for m, _ in results_with_distance[:n_results]]
```

**Add helper method:**

```python
def _rerank_by_emotion(
    self,
    results: List[tuple[Memory, float]],
    query_emotion: EmotionalContext,
    emotion_weight: float,
) -> List[tuple[Memory, float]]:
    """Rerank results by combined semantic-emotional score."""
    semantic_weight = 1.0 - emotion_weight

    scored_results = []
    for memory, semantic_distance in results:
        memory_emotion = memory.emotional_context
        emotion_sim = self._emotional_similarity(query_emotion, memory_emotion)
        score = self._combined_score(semantic_distance, emotion_sim, semantic_weight, emotion_weight)
        scored_results.append((memory, score))

    # Sort by combined score (higher = better)
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return scored_results

def _emotional_similarity(
    self,
    query_emotion: EmotionalContext,
    memory_emotion: Optional[EmotionalContext],
) -> float:
    """Compute emotional similarity between query and memory."""
    if not query_emotion or not memory_emotion:
        return 0.5

    valence_diff = query_emotion.valence - memory_emotion.valence
    arousal_diff = query_emotion.arousal - memory_emotion.arousal
    distance = (valence_diff**2 + arousal_diff**2) ** 0.5

    max_distance = 2.83  # sqrt(8)
    return 1.0 - (distance / max_distance)

def _combined_score(
    self,
    semantic_distance: float,
    emotional_similarity: float,
    semantic_weight: float,
    emotion_weight: float,
) -> float:
    """Combine semantic distance and emotional similarity."""
    semantic_similarity = max(0.0, 1.0 - semantic_distance / 2.0)
    return (semantic_weight * semantic_similarity) + (emotion_weight * emotional_similarity)
```

### File: `src/mnemosyne/server.py`

**Update tool schema:**

```python
Tool(
    name="search_memories",
    description="Search memories semantically with optional emotional context for mood-congruent retrieval",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "n_results": {
                "type": "integer",
                "default": 10,
                "description": "Number of results",
            },
            "emotional_context": {
                "type": "object",
                "properties": {
                    "valence": {"type": "number", "minimum": -1, "maximum": 1},
                    "arousal": {"type": "number", "minimum": -1, "maximum": 1},
                },
                "description": "Current emotional state for mood-congruent retrieval",
            },
            "emotion_weight": {
                "type": "number",
                "default": 0.3,
                "minimum": 0,
                "maximum": 1,
                "description": "Weight for emotional similarity (0 = pure semantic, 1 = pure emotional)",
            },
        },
        "required": ["query"],
    },
)
```

**Update handler:**

```python
async def _handle_search_memories(args: Dict[str, Any]) -> Dict[str, Any]:
    query = args["query"]
    n_results = args.get("n_results", 10)

    # Parse emotional context if provided
    emotional_ctx = None
    if args.get("emotional_context"):
        ec = args["emotional_context"]
        emotional_ctx = EmotionalContext(
            valence=ec.get("valence", 0.0),
            arousal=ec.get("arousal", 0.0),
            quadrant=ec.get("quadrant", "neutral"),
        )

    emotion_weight = args.get("emotion_weight", 0.3)

    memories = await asyncio.to_thread(
        memory_store.search_memories,
        query,
        n_results,
        emotional_context=emotional_ctx,
        emotion_weight=emotion_weight,
    )

    # ... rest of serialization unchanged
```

### File: `src/psyche/mcp/client.py`

**Update `MnemosyneClient.search_memories()`:**

```python
async def search_memories(
    self,
    query: str,
    n_results: int = 10,
    emotional_context: Optional[Dict[str, float]] = None,
    emotion_weight: float = 0.3,
) -> Dict:
    """
    Search memories with optional emotional context.

    Args:
        query: Search query
        n_results: Number of results
        emotional_context: Dict with 'valence' and 'arousal' keys
        emotion_weight: Weight for emotional similarity (0-1)
    """
    args = {"query": query, "n_results": n_results}
    if emotional_context:
        args["emotional_context"] = emotional_context
        args["emotion_weight"] = emotion_weight
    return await self._call_tool("search_memories", args)
```

### File: `src/psyche/core/memory_handler.py`

**Automatically include emotional context:**

```python
async def retrieve_relevant(
    self,
    query: str,
    n: int = None,
    include_emotion: bool = True,
) -> List[Dict]:
    """
    Retrieve relevant memories, optionally considering current emotional state.

    Args:
        query: Search query
        n: Number of results
        include_emotion: Whether to use mood-congruent retrieval
    """
    emotion_ctx = None
    if include_emotion and self._elpis_client:
        try:
            emotion = await self._elpis_client.get_emotion()
            emotion_ctx = {"valence": emotion.valence, "arousal": emotion.arousal}
        except Exception:
            pass  # Graceful degradation - search works without emotion

    return await self._mnemosyne_client.search_memories(
        query,
        n_results=n or self._default_n,
        emotional_context=emotion_ctx,
    )
```

---

## Configuration

### File: `src/mnemosyne/config/settings.py`

```python
class RetrievalSettings(BaseSettings):
    """Settings for memory retrieval."""

    model_config = SettingsConfigDict(env_prefix="MNEMOSYNE_RETRIEVAL__")

    emotion_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Default weight for emotional similarity in retrieval (0-1)",
    )
    candidate_multiplier: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Retrieve N*multiplier candidates when reranking",
    )
```

### File: `configs/mnemosyne.toml`

```toml
[retrieval]
emotion_weight = 0.3      # 30% emotion, 70% semantic
candidate_multiplier = 2  # Retrieve 2x candidates for reranking
```

---

## Testing

### Unit Tests: `tests/mnemosyne/unit/test_emotional_similarity.py`

```python
def test_emotional_similarity_identical():
    """Identical emotions should have similarity 1.0."""
    e1 = EmotionalContext(valence=0.5, arousal=0.3, quadrant="excited")
    e2 = EmotionalContext(valence=0.5, arousal=0.3, quadrant="excited")
    assert emotional_similarity(e1, e2) == pytest.approx(1.0)

def test_emotional_similarity_opposite():
    """Opposite corners should have low similarity."""
    e1 = EmotionalContext(valence=1.0, arousal=1.0, quadrant="excited")
    e2 = EmotionalContext(valence=-1.0, arousal=-1.0, quadrant="depleted")
    assert emotional_similarity(e1, e2) == pytest.approx(0.0, abs=0.01)

def test_emotional_similarity_missing():
    """Missing context should return neutral 0.5."""
    e1 = EmotionalContext(valence=0.5, arousal=0.3, quadrant="excited")
    assert emotional_similarity(e1, None) == 0.5
    assert emotional_similarity(None, e1) == 0.5

def test_combined_score_weights():
    """Verify weight blending."""
    # Pure semantic (emotion_weight=0)
    score = combined_score(0.5, 0.9, semantic_weight=1.0, emotion_weight=0.0)
    assert score == pytest.approx(0.75)  # 1 - 0.5/2 = 0.75

    # Pure emotional (emotion_weight=1)
    score = combined_score(0.5, 0.9, semantic_weight=0.0, emotion_weight=1.0)
    assert score == pytest.approx(0.9)
```

### Integration Tests: `tests/mnemosyne/integration/test_mood_congruent_retrieval.py`

```python
@pytest.fixture
async def store_with_emotional_memories(memory_store):
    """Store memories with varied emotional contexts."""
    # Happy memory
    happy = Memory(
        content="Had a wonderful conversation today",
        emotional_context=EmotionalContext(valence=0.8, arousal=0.3, quadrant="calm"),
    )
    # Frustrated memory
    frustrated = Memory(
        content="Struggled with a difficult problem",
        emotional_context=EmotionalContext(valence=-0.5, arousal=0.7, quadrant="frustrated"),
    )
    # Neutral memory
    neutral = Memory(
        content="Discussed technical details",
        emotional_context=EmotionalContext(valence=0.0, arousal=0.0, quadrant="neutral"),
    )

    for m in [happy, frustrated, neutral]:
        await asyncio.to_thread(memory_store.add_memory, m)

    return memory_store

async def test_mood_congruent_ranking(store_with_emotional_memories):
    """When searching with happy emotion, happy memories should rank higher."""
    store = store_with_emotional_memories

    # Search with calm/happy emotional context
    happy_ctx = EmotionalContext(valence=0.7, arousal=0.2, quadrant="calm")
    results = await asyncio.to_thread(
        store.search_memories,
        "conversation",
        n_results=3,
        emotional_context=happy_ctx,
        emotion_weight=0.5,
    )

    # Happy memory should rank first
    assert "wonderful" in results[0].content

async def test_pure_semantic_unchanged(store_with_emotional_memories):
    """With emotion_weight=0, ranking should match pure semantic."""
    store = store_with_emotional_memories

    semantic_results = await asyncio.to_thread(
        store.search_memories, "conversation", n_results=3
    )

    emotional_results = await asyncio.to_thread(
        store.search_memories,
        "conversation",
        n_results=3,
        emotional_context=EmotionalContext(valence=0.8, arousal=0.0, quadrant="calm"),
        emotion_weight=0.0,
    )

    # Same order
    assert [m.id for m in semantic_results] == [m.id for m in emotional_results]
```

---

## Session Estimate

| Task | Sessions |
|------|----------|
| Emotional similarity + reranking in ChromaStore | 1 |
| Update Mnemosyne server tool schema + handler | 0.5 |
| Update MnemosyneClient | 0.5 |
| Update memory_handler to pass emotion | 0.5 |
| Configuration settings | 0.5 |
| Unit tests | 1 |
| Integration tests | 1 |
| **Total** | **4-5** |

---

## Future Considerations

- **Decay over depth**: First results more mood-congruent, later results more semantically diverse
- **Adaptive weights**: Learn effective emotion_weight from outcomes
- **Quadrant-specific weights**: Some emotional states might benefit from more/less mood-congruence
