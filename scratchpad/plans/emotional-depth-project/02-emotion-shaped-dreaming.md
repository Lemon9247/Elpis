# Part 2: Emotion-Shaped Dreaming

## Problem Statement

Dreams currently use generic prompts and random memory retrieval. They don't respond to what Psyche actually needs emotionally. A dream triggered when frustrated should seek different memories than one triggered when depleted.

## Dream Intentions by Quadrant

| Quadrant | Emotional State | Dream Intention | What to Seek |
|----------|-----------------|-----------------|--------------|
| **Frustrated** | High arousal, low valence | Resolution | Success patterns, problems overcome, breakthroughs |
| **Depleted** | Low arousal, low valence | Restoration | Joy, connection, meaning, what gives energy |
| **Excited** | High arousal, high valence | Exploration | Challenge, complexity, growth, curiosity |
| **Calm** | Low arousal, high valence | Synthesis | Integration, patterns, understanding, connections |

---

## Implementation

### New Data Structures

**File: `src/psyche/handlers/dream_handler.py`**

```python
@dataclass
class DreamIntention:
    """Dream intention based on emotional state."""
    theme: str
    memory_queries: List[str]
    prompt_guidance: str


DREAM_INTENTIONS = {
    "frustrated": DreamIntention(
        theme="resolution",
        memory_queries=[
            "problems I solved",
            "challenges overcome",
            "moments of success",
            "when things clicked",
            "breakthroughs",
        ],
        prompt_guidance=(
            "Look for patterns in how challenges were resolved. "
            "What approaches worked? What can be learned from past successes?"
        ),
    ),
    "depleted": DreamIntention(
        theme="restoration",
        memory_queries=[
            "moments of joy",
            "meaningful connections",
            "things that matter",
            "what gives me energy",
            "why I do this",
        ],
        prompt_guidance=(
            "Reconnect with what matters. What brings meaning and energy? "
            "What relationships and experiences have been nourishing?"
        ),
    ),
    "excited": DreamIntention(
        theme="exploration",
        memory_queries=[
            "interesting questions",
            "things to learn",
            "curiosity and wonder",
            "complex problems",
            "possibilities",
        ],
        prompt_guidance=(
            "Explore complexity and possibility. What's worth pursuing? "
            "What questions remain open? Where does curiosity lead?"
        ),
    ),
    "calm": DreamIntention(
        theme="synthesis",
        memory_queries=[
            "patterns noticed",
            "connections between ideas",
            "emerging understanding",
            "things coming together",
            "insights",
        ],
        prompt_guidance=(
            "Integrate experiences into understanding. What patterns emerge? "
            "How do different experiences connect? What's becoming clearer?"
        ),
    ),
}

# Default for neutral/unknown states
DEFAULT_INTENTION = DreamIntention(
    theme="reflection",
    memory_queries=[
        "important moments",
        "things learned",
        "conversations",
        "feelings and emotions",
        "discoveries",
    ],
    prompt_guidance="Let your mind wander through these memories freely.",
)
```

### Modified Methods

**`_get_dream_memories()` - Now returns intention too:**

```python
async def _get_dream_memories(self) -> tuple[List[dict], DreamIntention]:
    """Retrieve memories for dream context based on emotional state."""
    if not self.core.is_mnemosyne_available:
        return [], DEFAULT_INTENTION

    # Get current emotional state
    intention = DEFAULT_INTENTION
    emotion_ctx = None

    try:
        emotion = await self.core.get_emotion()
        quadrant = emotion.quadrant
        if quadrant in DREAM_INTENTIONS:
            intention = DREAM_INTENTIONS[quadrant]
        emotion_ctx = {"valence": emotion.valence, "arousal": emotion.arousal}
    except Exception as e:
        logger.debug(f"Could not get emotional state for dream: {e}")

    # Retrieve memories using intention-specific queries
    all_memories = []
    memories_per_query = max(2, self.config.memory_query_count // len(intention.memory_queries))

    for query in intention.memory_queries:
        try:
            memories = await self.core.search_memories(
                query,
                n_results=memories_per_query,
                emotional_context=emotion_ctx,  # Uses mood-congruent retrieval if available
            )
            all_memories.extend(memories)
        except Exception as e:
            logger.debug(f"Dream memory query failed: {query}, {e}")

    # Deduplicate by ID
    seen = set()
    unique_memories = []
    for m in all_memories:
        memory_id = m.get("id", m.get("content", "")[:50])
        if memory_id not in seen:
            seen.add(memory_id)
            unique_memories.append(m)

    return unique_memories[:self.config.memory_query_count], intention
```

**`_build_dream_prompt()` - Now incorporates intention:**

```python
def _build_dream_prompt(self, memories: List[dict], intention: DreamIntention) -> str:
    """Build prompt for dream generation with emotional intention."""
    memory_texts = []
    for m in memories:
        content = m.get("content", "")
        if content:
            if len(content) > MEMORY_CONTENT_TRUNCATE_LENGTH:
                content = content[:MEMORY_CONTENT_TRUNCATE_LENGTH] + "..."
            memory_texts.append(f"- {content}")

    memories_section = "\n".join(memory_texts) if memory_texts else "- No specific memories surfacing"

    return f"""You are in a dream state, reflecting on your memories and experiences.

Theme for this dream: {intention.theme}

{intention.prompt_guidance}

Memories surfacing:
{memories_section}

Let your mind wander through these memories. What patterns do you notice?
What connections emerge between different experiences? What feelings arise?

Dream freely, without the need for action or response. This is a time for
reflection and integration.

Your dream:"""
```

**`_dream_once()` - Updated to use intention:**

```python
async def _dream_once(self) -> None:
    """Single dream episode - memory palace exploration."""
    logger.debug(f"Dream episode {self._dream_count + 1} starting...")

    try:
        # 1. Load memories for dream context (now returns intention too)
        memories, intention = await self._get_dream_memories()

        if not memories:
            logger.debug("No memories available for dreaming")
            return

        # 2. Build dream prompt with intention
        dream_prompt = self._build_dream_prompt(memories, intention)

        # 3. Generate dream
        dream_content = await self._generate_dream(dream_prompt)

        if not dream_content:
            return

        # 4. Log the dream with intention info
        self._log_dream(dream_content, memories, intention)

        # 5. Maybe store dream insights
        if self.config.store_insights and self._is_insightful(dream_content):
            await self._store_dream_insight(dream_content, intention)

    except Exception as e:
        logger.error(f"Error in dream episode: {e}")
```

**`_store_dream_insight()` - Tag with intention theme:**

```python
async def _store_dream_insight(self, dream_content: str, intention: DreamIntention) -> None:
    """Store a dream insight as a semantic memory."""
    try:
        await self.core.store_memory(
            content=f"[Dream insight - {intention.theme}] {dream_content}",
            importance=self.config.insight_importance_threshold,
            tags=["dream", "insight", "semantic", f"theme:{intention.theme}"],
        )
        logger.info(f"Stored dream insight (theme: {intention.theme})")

        await self._maybe_trigger_consolidation()

    except Exception as e:
        logger.warning(f"Failed to store dream insight: {e}")
```

---

## Testing

### Unit Tests

```python
def test_intention_selection_by_quadrant():
    """Verify correct intention selected for each quadrant."""
    assert DREAM_INTENTIONS["frustrated"].theme == "resolution"
    assert DREAM_INTENTIONS["depleted"].theme == "restoration"
    assert DREAM_INTENTIONS["excited"].theme == "exploration"
    assert DREAM_INTENTIONS["calm"].theme == "synthesis"

def test_default_intention_for_unknown():
    """Unknown quadrants should use default intention."""
    assert "neutral" not in DREAM_INTENTIONS
    # Code should fall back to DEFAULT_INTENTION
```

### Integration Tests

```python
async def test_dream_uses_emotional_intention(mock_core):
    """Dream should select intention based on emotional state."""
    # Set emotional state to frustrated
    mock_core.get_emotion.return_value = EmotionalState(
        valence=-0.5, arousal=0.7, quadrant="frustrated"
    )

    handler = DreamHandler(mock_core)
    memories, intention = await handler._get_dream_memories()

    assert intention.theme == "resolution"
    # Verify queries were about success/challenges
    mock_core.search_memories.assert_called()
    queries_used = [call.args[0] for call in mock_core.search_memories.call_args_list]
    assert any("solved" in q or "success" in q for q in queries_used)
```

---

## Session Estimate

| Task | Sessions |
|------|----------|
| DreamIntention dataclass and mappings | 0.5 |
| Update _get_dream_memories | 0.5 |
| Update _build_dream_prompt | 0.25 |
| Update _dream_once and storage | 0.25 |
| Tests | 0.5 |
| **Total** | **2** |

---

## Dependencies

- Part 1 (Mood-Congruent Retrieval) enhances this by making dream memory retrieval emotion-aware
- Works independently but better together

## Future Considerations

- Track dream effectiveness (did dreams during depleted state lead to emotional improvement?)
- Adaptive intention selection based on what's worked before
- Multi-phase dreams (start with restoration, transition to exploration as energy returns)
