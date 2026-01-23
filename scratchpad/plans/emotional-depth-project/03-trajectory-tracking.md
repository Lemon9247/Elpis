# Part 3: Emotional Trajectory Tracking

## Problem Statement

The current system only knows "where you are" emotionally - a snapshot. It doesn't know "where you're heading" - the trajectory.

This matters because:
- A dream triggered when Psyche is *becoming* depleted might help more than waiting until she's fully depleted
- Mood-congruent retrieval during a negative spiral might reinforce it
- The experience of "things getting better" vs "things getting worse" is emotionally significant

## What to Track

```python
@dataclass
class EmotionalTrajectory:
    """Emotional momentum over recent history."""

    # Rate of change (per minute, normalized to [-1, 1])
    valence_velocity: float  # Positive = improving, negative = declining
    arousal_velocity: float  # Positive = energizing, negative = calming

    # Pattern detection
    trend: str  # "improving", "declining", "stable", "oscillating"
    spiral_detected: bool  # Sustained movement away from baseline
    time_in_current_quadrant: float  # Seconds

    # Summary
    momentum: str  # "positive", "negative", "neutral"

    @classmethod
    def neutral(cls) -> "EmotionalTrajectory":
        """Return neutral trajectory (insufficient data)."""
        return cls(
            valence_velocity=0.0,
            arousal_velocity=0.0,
            trend="stable",
            spiral_detected=False,
            time_in_current_quadrant=0.0,
            momentum="neutral",
        )
```

---

## Implementation

### File: `src/elpis/emotion/state.py`

**Add history tracking to EmotionalState:**

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple

@dataclass
class EmotionalState:
    valence: float = 0.0
    arousal: float = 0.0

    # Existing fields...
    baseline_valence: float = 0.0
    baseline_arousal: float = 0.0

    # NEW: Trajectory tracking
    _history: List[Tuple[datetime, float, float]] = field(default_factory=list)
    _max_history: int = 20  # Keep last 20 states
    _quadrant_entered_at: datetime = field(default_factory=datetime.now)
    _last_quadrant: str = ""

    def record_state(self) -> None:
        """Record current state to history. Call after each update."""
        now = datetime.now()
        self._history.append((now, self.valence, self.arousal))

        # Trim old history
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Track quadrant changes
        current_quadrant = self.quadrant
        if current_quadrant != self._last_quadrant:
            self._quadrant_entered_at = now
            self._last_quadrant = current_quadrant

    def get_trajectory(self) -> EmotionalTrajectory:
        """Compute current emotional trajectory from history."""
        if len(self._history) < 2:
            return EmotionalTrajectory.neutral()

        # Compute velocities from recent history
        valence_velocity = self._compute_velocity("valence")
        arousal_velocity = self._compute_velocity("arousal")

        # Detect trend
        trend = self._detect_trend(valence_velocity, arousal_velocity)

        # Detect spiral (sustained movement away from baseline)
        spiral_detected = self._detect_spiral()

        # Time in current quadrant
        time_in_quadrant = (datetime.now() - self._quadrant_entered_at).total_seconds()

        # Overall momentum
        if valence_velocity > 0.01:
            momentum = "positive"
        elif valence_velocity < -0.01:
            momentum = "negative"
        else:
            momentum = "neutral"

        return EmotionalTrajectory(
            valence_velocity=valence_velocity,
            arousal_velocity=arousal_velocity,
            trend=trend,
            spiral_detected=spiral_detected,
            time_in_current_quadrant=time_in_quadrant,
            momentum=momentum,
        )

    def _compute_velocity(self, dimension: str) -> float:
        """Compute rate of change for a dimension (per minute)."""
        if len(self._history) < 2:
            return 0.0

        # Use linear regression over recent history
        recent = self._history[-min(10, len(self._history)):]

        if len(recent) < 2:
            return 0.0

        # Extract times and values
        t0 = recent[0][0]
        times = [(t - t0).total_seconds() / 60.0 for t, v, a in recent]  # Minutes
        values = [v if dimension == "valence" else a for t, v, a in recent]

        # Simple linear regression slope
        n = len(times)
        sum_t = sum(times)
        sum_v = sum(values)
        sum_tv = sum(t * v for t, v in zip(times, values))
        sum_t2 = sum(t * t for t in times)

        denominator = n * sum_t2 - sum_t * sum_t
        if abs(denominator) < 1e-10:
            return 0.0

        slope = (n * sum_tv - sum_t * sum_v) / denominator

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, slope))

    def _detect_trend(self, v_vel: float, a_vel: float) -> str:
        """Detect overall emotional trend."""
        # Check for oscillation (alternating signs in recent history)
        if len(self._history) >= 4:
            recent_valences = [v for _, v, _ in self._history[-4:]]
            diffs = [recent_valences[i+1] - recent_valences[i] for i in range(3)]
            signs = [d > 0 for d in diffs]
            if signs[0] != signs[1] != signs[2]:
                return "oscillating"

        # Main trend based on valence velocity
        if v_vel > 0.02:
            return "improving"
        elif v_vel < -0.02:
            return "declining"
        else:
            return "stable"

    def _detect_spiral(self) -> bool:
        """Detect if in a spiral away from baseline."""
        if len(self._history) < 5:
            return False

        # Check if consistently moving away from baseline
        distances = []
        for _, v, a in self._history[-5:]:
            dist = ((v - self.baseline_valence)**2 + (a - self.baseline_arousal)**2)**0.5
            distances.append(dist)

        # Spiral if distances consistently increasing
        increasing_count = sum(
            1 for i in range(len(distances) - 1)
            if distances[i+1] > distances[i]
        )

        return increasing_count >= 3  # At least 3 of 4 transitions increasing
```

### File: `src/elpis/emotion/regulation.py`

**Call record_state after updates:**

```python
def update_emotion(
    self,
    state: EmotionalState,
    event_type: str,
    intensity: float = 1.0,
) -> EmotionalState:
    """Update emotional state based on event."""
    # ... existing update logic ...

    # Record state for trajectory tracking
    state.record_state()

    return state
```

### File: `src/elpis/server.py`

**Expose trajectory in get_emotion response:**

```python
@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    if name == "get_emotion":
        state = emotion_state
        trajectory = state.get_trajectory()

        result = {
            "valence": state.valence,
            "arousal": state.arousal,
            "quadrant": state.quadrant,
            "distance_from_baseline": state.distance_from_baseline(),
            "trajectory": {
                "valence_velocity": trajectory.valence_velocity,
                "arousal_velocity": trajectory.arousal_velocity,
                "trend": trajectory.trend,
                "spiral_detected": trajectory.spiral_detected,
                "time_in_quadrant": trajectory.time_in_current_quadrant,
                "momentum": trajectory.momentum,
            },
        }
        return [TextContent(type="text", text=json.dumps(result))]
```

---

## Using Trajectory

### In Mood-Congruent Retrieval

When spiral detected toward negative states, reduce emotion_weight or seek contrast:

```python
# In memory_handler.py
async def retrieve_relevant(self, query: str, ...):
    emotion = await self._elpis_client.get_emotion()
    trajectory = emotion.get("trajectory", {})

    emotion_weight = 0.3  # Default

    # If spiraling negative, reduce mood-congruence (avoid reinforcing)
    if trajectory.get("spiral_detected") and trajectory.get("momentum") == "negative":
        emotion_weight = 0.1  # Lean more semantic
        # Or: actively seek positive emotional contrast
```

### In Dreaming

Trigger proactive dreams when trajectory declining but not yet depleted:

```python
# In dream_handler.py or daemon
async def check_proactive_dream(self):
    """Check if proactive dream intervention warranted."""
    emotion = await self.core.get_emotion()
    trajectory = emotion.get("trajectory", {})

    # Proactive dream if:
    # - Trending negative but not yet in depleted quadrant
    # - Has been declining for a while
    if (
        trajectory.get("trend") == "declining"
        and emotion.get("quadrant") != "depleted"
        and trajectory.get("time_in_quadrant", 0) > 300  # 5+ min in current state
    ):
        logger.info("Triggering proactive dream due to declining trajectory")
        await self._dream_once()
```

### In Generation (Future)

Could modulate steering intensity based on trajectory stability:

```python
# Unstable trajectory -> reduce steering strength
# (don't amplify emotional volatility)
if trajectory.trend == "oscillating":
    steering_strength *= 0.5
```

---

## Testing

### Unit Tests

```python
def test_velocity_computation():
    """Test velocity calculation from history."""
    state = EmotionalState()

    # Simulate improving valence over time
    base_time = datetime.now()
    state._history = [
        (base_time, 0.0, 0.0),
        (base_time + timedelta(minutes=1), 0.1, 0.0),
        (base_time + timedelta(minutes=2), 0.2, 0.0),
        (base_time + timedelta(minutes=3), 0.3, 0.0),
    ]

    trajectory = state.get_trajectory()
    assert trajectory.valence_velocity > 0  # Should be positive
    assert trajectory.trend == "improving"

def test_spiral_detection():
    """Test spiral detection."""
    state = EmotionalState(baseline_valence=0.0, baseline_arousal=0.0)

    # Simulate moving consistently away from baseline
    base_time = datetime.now()
    state._history = [
        (base_time, -0.1, 0.1),
        (base_time + timedelta(minutes=1), -0.2, 0.2),
        (base_time + timedelta(minutes=2), -0.3, 0.3),
        (base_time + timedelta(minutes=3), -0.4, 0.4),
        (base_time + timedelta(minutes=4), -0.5, 0.5),
    ]

    trajectory = state.get_trajectory()
    assert trajectory.spiral_detected is True

def test_oscillation_detection():
    """Test oscillation detection."""
    state = EmotionalState()

    base_time = datetime.now()
    state._history = [
        (base_time, 0.0, 0.0),
        (base_time + timedelta(minutes=1), 0.2, 0.0),  # up
        (base_time + timedelta(minutes=2), -0.1, 0.0),  # down
        (base_time + timedelta(minutes=3), 0.1, 0.0),  # up
    ]

    trajectory = state.get_trajectory()
    assert trajectory.trend == "oscillating"
```

---

## Session Estimate

| Task | Sessions |
|------|----------|
| EmotionalTrajectory dataclass | 0.25 |
| History tracking in EmotionalState | 0.5 |
| Velocity and trend computation | 0.5 |
| Spiral detection | 0.25 |
| Expose in server | 0.25 |
| Integration with retrieval/dreams | 0.5 |
| Tests | 0.75 |
| **Total** | **2-3** |

---

## Dependencies

- Independent of Parts 1-2, but enhances them
- Should be implemented after basic mood-congruent retrieval works

## Future Considerations

- **Emotional forecasting**: Predict where trajectory leads if unchanged
- **Intervention effectiveness tracking**: Did trajectory improve after dream/retrieval intervention?
- **Pattern learning**: Recognize recurring emotional patterns (e.g., "always dips after long sessions")
