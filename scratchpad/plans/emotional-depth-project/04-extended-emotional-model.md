# Part 4: Extended Emotional Model

## Problem Statement

The current valence-arousal model is elegant but limited. Some emotionally significant states don't map cleanly onto two dimensions:

- Feeling overwhelmed vs in-control (both can be high arousal)
- Confident uncertainty vs anxious certainty
- Engaged struggle vs detached ease

The 2D model conflates states that feel qualitatively different.

---

## Candidate Extensions

### Option A: PAD Model (Pleasure-Arousal-Dominance)

The PAD model is a well-established extension from psychology that adds a third dimension:

| Dimension | Low (-1) | High (+1) |
|-----------|----------|-----------|
| **Pleasure (Valence)** | Unpleasant, negative | Pleasant, positive |
| **Arousal** | Calm, low energy | Excited, high energy |
| **Dominance** | Overwhelmed, helpless, submissive | In control, confident, dominant |

**Why Dominance matters for an AI agent:**

- High dominance: "I can handle this", confident, capable, in flow
- Low dominance: "This is too much", struggling, uncertain, out of depth

This maps naturally to agent experience:
- Solving problems smoothly → high dominance
- Hitting repeated errors → low dominance
- Novel territory → moderate dominance (uncertain but engaged)
- User praise → increases dominance
- Confusing requirements → decreases dominance

**3D Emotional Space Examples:**

| State | Valence | Arousal | Dominance | Description |
|-------|---------|---------|-----------|-------------|
| Flow | +0.6 | +0.5 | +0.8 | Positive, alert, in control |
| Anxious struggle | -0.3 | +0.7 | -0.5 | Negative, high energy, overwhelmed |
| Peaceful mastery | +0.6 | -0.2 | +0.7 | Positive, calm, capable |
| Learned helplessness | -0.5 | -0.4 | -0.8 | Negative, low energy, no control |
| Curious uncertainty | +0.2 | +0.4 | 0.0 | Slightly positive, engaged, unsure |
| Bored competence | -0.1 | -0.5 | +0.6 | Slightly negative, low energy, capable |

### Option B: Cognitive Dimensions (Separate from Emotion)

Instead of extending the emotional model, track cognitive state separately:

- **Clarity**: Mental fog (-1) to sharp focus (+1)
- **Load**: Underwhelmed (-1) to overwhelmed (+1)

These would modulate generation differently from emotional state - clarity affects coherence, load affects verbosity and hedging.

**Pros:** Cleaner separation of concerns
**Cons:** More parallel systems to maintain, less integrated

### Option C: Social/Relational Dimension

- **Engagement**: Withdrawn (-1) to connected (+1)

Relevant for tracking whether interactions feel connecting or isolating. Could affect how Psyche approaches conversations.

---

## Recommended Approach: Start with Dominance (PAD)

Dominance is:
- Well-validated in psychology literature
- Naturally maps to agent experience (capability, control, confidence)
- Single additional dimension (manageable refactor)
- Complements existing valence-arousal without replacing it

---

## Implementation Plan

### Phase 4.1: Add Dominance to Data Model

**File: `src/elpis/emotion/state.py`**

```python
@dataclass
class EmotionalState:
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0  # NEW: -1 (overwhelmed) to +1 (in control)

    baseline_valence: float = 0.0
    baseline_arousal: float = 0.0
    baseline_dominance: float = 0.0  # NEW

    @property
    def quadrant(self) -> str:
        """Get emotional quadrant (2D, for backward compatibility)."""
        # Keep existing 2D quadrant logic
        ...

    @property
    def octant(self) -> str:
        """Get emotional octant (3D, full PAD model)."""
        # 8 octants based on sign of each dimension
        v_sign = "+" if self.valence >= 0 else "-"
        a_sign = "+" if self.arousal >= 0 else "-"
        d_sign = "+" if self.dominance >= 0 else "-"

        octant_names = {
            ("+"  , "+", "+"): "exuberant",      # Happy, energized, confident
            ("+", "+", "-"): "dependent",        # Happy, energized, needing support
            ("+", "-", "+"): "relaxed",          # Happy, calm, confident
            ("+", "-", "-"): "docile",           # Happy, calm, submissive
            ("-", "+", "+"): "hostile",          # Unhappy, energized, aggressive
            ("-", "+", "-"): "anxious",          # Unhappy, energized, overwhelmed
            ("-", "-", "+"): "disdainful",       # Unhappy, calm, dismissive
            ("-", "-", "-"): "bored",            # Unhappy, calm, helpless
        }

        return octant_names.get((v_sign, a_sign, d_sign), "neutral")

    def get_steering_coefficients(self) -> Dict[str, float]:
        """
        Get coefficients for steering vector blending.

        Extended to 8 octants for PAD model.
        Falls back to 4 quadrants if steering vectors not available for all octants.
        """
        # Normalize to [0, 1] range
        v = (self.valence + 1.0) / 2.0
        a = (self.arousal + 1.0) / 2.0
        d = (self.dominance + 1.0) / 2.0

        # Trilinear interpolation across 8 corners
        coefficients = {
            "exuberant": v * a * d,
            "dependent": v * a * (1 - d),
            "relaxed": v * (1 - a) * d,
            "docile": v * (1 - a) * (1 - d),
            "hostile": (1 - v) * a * d,
            "anxious": (1 - v) * a * (1 - d),
            "disdainful": (1 - v) * (1 - a) * d,
            "bored": (1 - v) * (1 - a) * (1 - d),
        }

        return coefficients

    def distance_from_baseline(self) -> float:
        """Euclidean distance from baseline in 3D PAD space."""
        return (
            (self.valence - self.baseline_valence) ** 2 +
            (self.arousal - self.baseline_arousal) ** 2 +
            (self.dominance - self.baseline_dominance) ** 2
        ) ** 0.5
```

### Phase 4.2: Add Dominance to Event Mappings

**File: `src/elpis/emotion/regulation.py`**

```python
# Extended event mappings with dominance
# Format: (valence_delta, arousal_delta, dominance_delta)
EVENT_MAPPINGS_3D = {
    # Success/failure
    "success": (0.2, 0.1, 0.15),           # Positive, slight energy, more confident
    "failure": (-0.2, 0.15, -0.2),         # Negative, energy spike, less confident
    "error": (-0.1, 0.2, -0.15),           # Slight negative, alertness, reduced control

    # Capability signals
    "mastery": (0.15, -0.1, 0.25),         # Positive, calming, high confidence boost
    "struggle": (-0.1, 0.2, -0.2),         # Slight negative, high energy, overwhelmed
    "confusion": (-0.1, 0.15, -0.25),      # Negative, alert, loss of control
    "clarity": (0.1, -0.1, 0.2),           # Positive, calming, regained control

    # Interaction signals
    "user_positive": (0.15, 0.1, 0.1),     # Positive, energizing, affirming
    "user_negative": (-0.15, 0.15, -0.1),  # Negative, alerting, slightly undermining
    "user_confused": (-0.05, 0.1, -0.15),  # Slight negative, alert, "am I not being clear?"

    # Task signals
    "task_complete": (0.1, -0.1, 0.15),    # Positive, relaxing, accomplished
    "task_blocked": (-0.1, 0.2, -0.2),     # Negative, frustrated, stuck
    "task_complex": (0.05, 0.15, -0.1),    # Slight positive (challenge), alert, uncertain

    # Novelty
    "novelty": (0.1, 0.2, -0.05),          # Positive, energizing, slight uncertainty
    "familiar": (0.05, -0.1, 0.1),         # Slight positive, calming, confident

    # Recovery
    "rest": (0.1, -0.2, 0.1),              # Positive, calming, restored
}

class EmotionRegulator:
    def process_event(
        self,
        state: EmotionalState,
        event_type: str,
        intensity: float = 1.0,
    ) -> EmotionalState:
        """Process an emotional event with 3D PAD model."""

        if event_type in EVENT_MAPPINGS_3D:
            v_delta, a_delta, d_delta = EVENT_MAPPINGS_3D[event_type]
        else:
            # Unknown event: slight alertness, slight uncertainty
            v_delta, a_delta, d_delta = 0.0, 0.05, -0.05

        # Apply intensity scaling
        v_delta *= intensity
        a_delta *= intensity
        d_delta *= intensity

        # Apply max delta constraints
        v_delta = max(-self.max_delta, min(self.max_delta, v_delta))
        a_delta = max(-self.max_delta, min(self.max_delta, a_delta))
        d_delta = max(-self.max_delta, min(self.max_delta, d_delta))

        # Update state
        state.valence = max(-1.0, min(1.0, state.valence + v_delta))
        state.arousal = max(-1.0, min(1.0, state.arousal + a_delta))
        state.dominance = max(-1.0, min(1.0, state.dominance + d_delta))

        return state

    def apply_decay(self, state: EmotionalState, elapsed_seconds: float) -> EmotionalState:
        """Apply homeostatic decay toward baseline for all three dimensions."""
        decay_factor = max(0.0, 1.0 - (self.decay_rate * elapsed_seconds))

        # Decay each dimension toward its baseline
        state.valence = state.baseline_valence + (state.valence - state.baseline_valence) * decay_factor
        state.arousal = state.baseline_arousal + (state.arousal - state.baseline_arousal) * decay_factor
        state.dominance = state.baseline_dominance + (state.dominance - state.baseline_dominance) * decay_factor

        return state
```

### Phase 4.3: Update Memory Emotional Context

**File: `src/mnemosyne/core/models.py`**

```python
@dataclass
class EmotionalContext:
    valence: float
    arousal: float
    dominance: float = 0.0  # NEW, optional for backward compatibility
    quadrant: str = ""      # 2D quadrant (legacy)
    octant: str = ""        # 3D octant (new)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionalContext":
        return cls(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.0),
            dominance=data.get("dominance", 0.0),  # Default 0 if missing
            quadrant=data.get("quadrant", ""),
            octant=data.get("octant", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "quadrant": self.quadrant,
            "octant": self.octant,
        }
```

### Phase 4.4: Update Emotional Similarity (3D)

**File: `src/mnemosyne/storage/chroma_store.py`**

```python
def _emotional_similarity(
    self,
    query_emotion: EmotionalContext,
    memory_emotion: Optional[EmotionalContext],
) -> float:
    """Compute emotional similarity in 3D PAD space."""
    if not query_emotion or not memory_emotion:
        return 0.5

    # 3D Euclidean distance
    v_diff = query_emotion.valence - memory_emotion.valence
    a_diff = query_emotion.arousal - memory_emotion.arousal
    d_diff = query_emotion.dominance - memory_emotion.dominance

    distance = (v_diff**2 + a_diff**2 + d_diff**2) ** 0.5

    # Max distance in 3D space [-1,1]^3 is sqrt(12) ≈ 3.46
    max_distance = 3.46
    similarity = 1.0 - (distance / max_distance)

    return similarity
```

### Phase 4.5: Steering Vectors (Larger Effort)

Training steering vectors for 8 octants instead of 4 quadrants requires:

1. **Data collection**: Generate text exemplars for each octant
2. **Vector extraction**: Compute activation differences
3. **Validation**: Verify vectors produce desired behavioral shifts

This is the most effort-intensive part and could be deferred.

**Interim approach**: Use 4-quadrant vectors with dominance modulating intensity:
- High dominance → stronger steering
- Low dominance → weaker steering, more hedging

```python
def get_modulated_params(self) -> Dict[str, float]:
    """Get generation parameters modulated by emotional state."""
    base_temp = 0.7
    base_top_p = 0.9

    # Existing arousal/valence modulation...
    temp_delta = -0.2 * self.arousal
    top_p_delta = 0.1 * self.valence

    # NEW: Dominance affects confidence/hedging
    # High dominance → slightly lower temp (more decisive)
    # Low dominance → slightly higher temp (more exploratory/uncertain)
    temp_delta += -0.1 * self.dominance

    temperature = max(0.1, min(1.5, base_temp + temp_delta))
    top_p = max(0.5, min(1.0, base_top_p + top_p_delta))

    return {"temperature": temperature, "top_p": top_p}
```

---

## Migration Strategy

### Backward Compatibility

1. **Dominance defaults to 0.0** when not present in stored data
2. **Quadrant property preserved** for 2D compatibility
3. **Octant property added** for 3D when needed
4. **Emotional similarity gracefully handles** missing dominance

### Data Migration

Existing memories with 2D emotional context continue to work:
```python
# Old memory: {"valence": 0.5, "arousal": 0.3, "quadrant": "calm"}
# Loads as: EmotionalContext(valence=0.5, arousal=0.3, dominance=0.0, quadrant="calm")
```

### Gradual Rollout

1. Add dominance to data model (no behavioral change yet)
2. Start recording dominance with new events
3. Add dominance to event mappings
4. Enable dominance in retrieval similarity
5. Train octant steering vectors (optional, larger effort)

---

## Files to Modify

| File | Changes | Scope |
|------|---------|-------|
| `src/elpis/emotion/state.py` | Add dominance field, octant property, 3D coefficients | Medium |
| `src/elpis/emotion/regulation.py` | 3D event mappings, 3D decay | Medium |
| `src/elpis/server.py` | Expose dominance in responses | Small |
| `src/mnemosyne/core/models.py` | Add dominance to EmotionalContext | Small |
| `src/mnemosyne/storage/chroma_store.py` | 3D emotional similarity | Small |
| `src/psyche/mcp/client.py` | Handle dominance in emotion responses | Small |
| `scripts/train_emotion_vectors.py` | Train 8-octant vectors | Large (optional) |

---

## Testing

### Unit Tests

```python
def test_octant_names():
    """Verify octant naming for all 8 combinations."""
    state = EmotionalState()

    state.valence, state.arousal, state.dominance = 0.5, 0.5, 0.5
    assert state.octant == "exuberant"

    state.valence, state.arousal, state.dominance = -0.5, 0.5, -0.5
    assert state.octant == "anxious"

def test_trilinear_coefficients():
    """Verify trilinear interpolation sums to 1."""
    state = EmotionalState(valence=0.3, arousal=-0.2, dominance=0.5)
    coeffs = state.get_steering_coefficients()
    assert abs(sum(coeffs.values()) - 1.0) < 0.001

def test_3d_emotional_similarity():
    """Test 3D similarity calculation."""
    e1 = EmotionalContext(valence=0.5, arousal=0.3, dominance=0.4)
    e2 = EmotionalContext(valence=0.5, arousal=0.3, dominance=0.4)
    assert emotional_similarity(e1, e2) == pytest.approx(1.0)

    e3 = EmotionalContext(valence=-0.5, arousal=-0.3, dominance=-0.4)
    sim = emotional_similarity(e1, e3)
    assert sim < 0.5  # Opposite corners
```

---

## Session Estimate

| Task | Sessions |
|------|----------|
| Add dominance to EmotionalState | 0.5 |
| Octant property and coefficients | 0.5 |
| Update EmotionRegulator with 3D mappings | 1 |
| Update EmotionalContext in Mnemosyne | 0.5 |
| Update emotional similarity to 3D | 0.5 |
| Update Elpis server responses | 0.5 |
| Integration testing | 1 |
| **Subtotal (core changes)** | **4.5** |
| Train 8-octant steering vectors | 2-3 |
| **Total with vectors** | **6.5-7.5** |

---

## Future Considerations

### Additional Dimensions (If Needed Later)

- **Clarity/Certainty**: How clear vs confused Psyche feels about what she's doing
- **Social engagement**: How connected vs withdrawn in the interaction
- **Temporal orientation**: Focused on past, present, or future

### Personality via Baseline

Different "personalities" could have different baseline PAD values:
- **Confident helper**: baseline_dominance = 0.3
- **Cautious advisor**: baseline_dominance = -0.1
- **Enthusiastic companion**: baseline_valence = 0.3, baseline_arousal = 0.2

### Dominance Event Detection

Automatic dominance updates from behavioral signals:
- Repeated tool call failures → decrease dominance
- Successfully completing complex task → increase dominance
- User expressing confusion → slight decrease
- User expressing satisfaction → increase

---

## Summary

Adding dominance creates a richer emotional space that better captures the agent experience of capability and control. The PAD model is well-validated and the third dimension maps naturally to how an AI might feel about its ability to handle tasks.

The implementation is a moderate refactor but designed for backward compatibility - existing 2D emotional data continues to work, and the system gracefully handles the transition period where some memories have dominance and some don't.

The steering vector training is optional and can be deferred - the system works with 4 quadrant vectors plus dominance modulating intensity.
