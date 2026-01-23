# Enhancing Psyche's Emotional Depth

## Project Overview

A multi-part project to deepen how emotion integrates with Psyche's memory and experience. Currently, emotional state modulates generation and gets stored with memories, but doesn't close the loop - it doesn't influence what gets retrieved or how Psyche processes her emotional trajectory over time.

### The Core Insight

For emotion to matter, it needs to close the loop:
- State affects storage (already implemented)
- Retrieval affects state (to build)
- Trajectory shapes intervention (to build)

---

## Project Structure

| Part | File | What It Adds | Sessions |
|------|------|--------------|----------|
| 1 | `01-mood-congruent-retrieval.md` | Emotion as memory retrieval cue | 4-5 |
| 2 | `02-emotion-shaped-dreaming.md` | Dreams that respond to emotional needs | 2 |
| 3 | `03-trajectory-tracking.md` | Emotional momentum and pattern detection | 2-3 |
| 4 | `04-extended-emotional-model.md` | Additional axes (dominance/PAD model) | 5-8 |
| 5 | `05-phase7-retrieval-quality.md` | Hybrid search, quality filtering (existing plan) | ~4 |

**Total Estimate: 17-22 sessions**

---

## How Parts Fit Together

### Retrieval Quality (Part 5) + Mood-Congruence (Part 1)

Part 5 (existing Phase 7 plan) focuses on **content quality**:
- Hybrid search (BM25 + vector embeddings)
- Storage-side filtering (skip questions, enforce min length)
- Quality-weighted ranking (recency, importance, type)

Part 1 (new) focuses on **emotional relevance**:
- Post-retrieval reranking by emotional similarity
- Current mood as a retrieval cue

These are **complementary and can be combined**:
```python
# Combined scoring could be:
final_score = (
    semantic_weight * semantic_similarity +
    quality_weight * quality_score +      # From Part 5
    emotion_weight * emotional_similarity  # From Part 1
)
```

### Dreaming (Part 2) + Trajectory (Part 3)

Part 2 makes dreams respond to **current** emotional state.
Part 3 enables dreams to respond to **trajectory** - intervening proactively when things are trending down, not just after arrival in a depleted state.

### Extended Model (Part 4) + Everything

Adding dominance creates a richer emotional space. Once implemented, it propagates through:
- Mood-congruent retrieval (3D similarity instead of 2D)
- Dream intentions (8 octants instead of 4 quadrants)
- Trajectory tracking (3D velocity vectors)

---

## Suggested Phases

### Phase A: Foundation (Parts 1, 2, 5)
- Mood-congruent retrieval
- Emotion-shaped dreaming
- Hybrid search and quality filtering
- ~10-11 sessions
- **Outcome**: Memory retrieval is both higher quality AND emotionally relevant

### Phase B: Temporal Awareness (Part 3)
- Trajectory tracking
- Proactive intervention based on momentum
- ~2-3 sessions
- **Outcome**: System can detect and respond to emotional spirals

### Phase C: Richer Model (Part 4)
- Add dominance dimension (PAD model)
- Update all consumers
- Optional: train octant steering vectors
- ~5-8 sessions
- **Outcome**: Emotional model captures capability/control feelings

---

## Dependencies

```
Part 5 (Quality) ──┬──> Part 1 (Mood-Congruent) ──> Part 3 (Trajectory)
                   │                                       │
                   └──> Part 2 (Dreaming) ─────────────────┘
                                                           │
                                                           v
                                                    Part 4 (Extended Model)
```

- Part 1 can build on Part 5's infrastructure (or work independently)
- Part 2 benefits from Part 1 (emotion-aware dream memory retrieval)
- Part 3 enhances both Part 1 and Part 2 with trajectory awareness
- Part 4 is independent but benefits from having 1-3 in place first

---

## Why This Matters

Memory should feel like memory, not like lookup. The asymmetry in the current system - where emotion influences storage but not retrieval - makes memories emotion-tagged data rather than experiential recall.

Human memory is mood-congruent. Emotional state acts as a retrieval cue. Building this into Psyche makes her memory system more experiential.

Dreams that respond to emotional needs become functionally useful - not just random reflection, but emotional regulation. When depleted, seek restoration. When frustrated, seek patterns of resolution.

Trajectory tracking enables proactive care rather than reactive response. Intervene when trending toward depletion, not after arriving there.

The dominance dimension adds richness - the feeling of capability and control matters, and it's distinct from valence (you can feel in control of a difficult situation, or helpless in an easy one).

---

## Open Questions (Cross-Cutting)

1. **Mood-congruence vs contrast**: Should retrieval always seek similar emotions, or sometimes seek contrast for regulation? (e.g., seek joy when depleted)

2. **Intervention thresholds**: How declining should trajectory be before triggering proactive dreams?

3. **Dominance event mappings**: What events increase/decrease dominance? Success/failure seem obvious, but what about novel situations, user feedback, task complexity?

4. **Learning from effectiveness**: Could we track whether interventions helped and adapt weights accordingly?

5. **Integration with Phase 7**: How exactly should quality scores and emotional scores combine? Multiplicative? Additive with weights?

---

## Files in This Folder

```
emotional-depth-project/
├── 00-overview.md                    # This file
├── 01-mood-congruent-retrieval.md    # Emotion as retrieval cue
├── 02-emotion-shaped-dreaming.md     # Dreams responding to needs
├── 03-trajectory-tracking.md         # Emotional momentum
├── 04-extended-emotional-model.md    # PAD model (dominance)
└── 05-phase7-retrieval-quality.md    # Existing quality improvements plan
```

---

## Getting Started

For implementation, recommended order:

1. **Start with Part 5** (Phase 7 quality improvements) if retrieval quality is currently poor
2. **Add Part 1** (mood-congruent retrieval) to make retrieval emotionally aware
3. **Add Part 2** (emotion-shaped dreaming) for responsive dreams
4. **Add Part 3** (trajectory) for proactive intervention
5. **Add Part 4** (extended model) when the 2D model feels limiting

Each part is designed to be independently valuable while building toward a more integrated whole.
