"""Valence-Arousal emotional state model with trajectory tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Tuple
import time


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "valence_velocity": round(self.valence_velocity, 4),
            "arousal_velocity": round(self.arousal_velocity, 4),
            "trend": self.trend,
            "spiral_detected": self.spiral_detected,
            "time_in_quadrant": round(self.time_in_current_quadrant, 1),
            "momentum": self.momentum,
        }


@dataclass
class EmotionalState:
    """
    2D Emotional state using Valence-Arousal model.

    Valence: Pleasant (+1) to Unpleasant (-1)
    Arousal: High energy (+1) to Low energy (-1)

    Quadrants:
    - High Arousal, High Valence: Excited, Happy, Curious
    - High Arousal, Low Valence: Angry, Frustrated, Anxious
    - Low Arousal, High Valence: Calm, Content, Satisfied
    - Low Arousal, Low Valence: Sad, Bored, Tired
    """

    valence: float = 0.0  # -1.0 to +1.0
    arousal: float = 0.0  # -1.0 to +1.0
    last_update: float = field(default_factory=time.time)
    update_count: int = 0

    # Homeostasis target (can be customized per "personality")
    baseline_valence: float = 0.0
    baseline_arousal: float = 0.0

    # Global steering strength multiplier for emotional expression
    # 0.0 = no emotional modulation
    # 1.0 = normal expression
    # >1.0 = exaggerated expression (use carefully!)
    steering_strength: float = 1.0

    # Trajectory tracking
    _history: List[Tuple[datetime, float, float]] = field(default_factory=list)
    _max_history: int = 20  # Keep last 20 states
    _quadrant_entered_at: datetime = field(default_factory=datetime.now)
    _last_quadrant: str = ""

    def __post_init__(self) -> None:
        """Validate initial state bounds."""
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(-1.0, min(1.0, self.arousal))

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation."""
        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "quadrant": self.get_quadrant(),
            "last_update": self.last_update,
            "update_count": self.update_count,
            "baseline": {
                "valence": self.baseline_valence,
                "arousal": self.baseline_arousal,
            },
            "steering_coefficients": self.get_steering_coefficients(),
            "trajectory": self.get_trajectory().to_dict(),
        }

    def get_quadrant(self) -> str:
        """
        Return named emotional quadrant.

        Returns:
            String name of current emotional quadrant
        """
        if self.arousal >= 0:
            return "excited" if self.valence >= 0 else "frustrated"
        else:
            return "calm" if self.valence >= 0 else "depleted"

    def get_modulated_params(self) -> Dict[str, float]:
        """
        Convert emotional state to LLM inference parameters.

        Mapping:
        - High arousal -> Lower temperature (more focused)
        - Low arousal -> Higher temperature (more exploratory)
        - High valence -> Higher top_p (broader sampling)
        - Low valence -> Lower top_p (more conservative)

        Returns:
            Dictionary with 'temperature' and 'top_p' values
        """
        # Base parameters
        base_temp = 0.7
        base_top_p = 0.9

        # Arousal modulates temperature inversely
        # High arousal = focused = lower temp
        temp_delta = -0.2 * self.arousal  # Range: -0.2 to +0.2
        temperature = max(0.1, min(1.5, base_temp + temp_delta))

        # Valence modulates top_p
        # High valence = confident = broader sampling
        top_p_delta = 0.1 * self.valence  # Range: -0.1 to +0.1
        top_p = max(0.5, min(1.0, base_top_p + top_p_delta))

        return {
            "temperature": round(temperature, 2),
            "top_p": round(top_p, 2),
        }

    def reset(self) -> None:
        """Reset to baseline state."""
        self.valence = self.baseline_valence
        self.arousal = self.baseline_arousal
        self.last_update = time.time()

    def shift(self, valence_delta: float, arousal_delta: float) -> None:
        """
        Shift emotional state by given deltas.

        Args:
            valence_delta: Change in valence (-1 to +1)
            arousal_delta: Change in arousal (-1 to +1)
        """
        self.valence = max(-1.0, min(1.0, self.valence + valence_delta))
        self.arousal = max(-1.0, min(1.0, self.arousal + arousal_delta))
        self.last_update = time.time()
        self.update_count += 1

        # Record state for trajectory tracking
        self.record_state()

    def record_state(self) -> None:
        """Record current state to history for trajectory tracking."""
        now = datetime.now()
        self._history.append((now, self.valence, self.arousal))

        # Trim old history
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Track quadrant changes
        current_quadrant = self.get_quadrant()
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
        trend = self._detect_trend(valence_velocity)

        # Detect spiral (sustained movement away from baseline)
        spiral_detected = self._detect_spiral()

        # Time in current quadrant
        time_in_quadrant = (datetime.now() - self._quadrant_entered_at).total_seconds()

        # Overall momentum based on valence velocity
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
        """
        Compute rate of change for a dimension (per minute).

        Uses linear regression over recent history.
        """
        if len(self._history) < 2:
            return 0.0

        # Use up to last 10 states
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

    def _detect_trend(self, valence_velocity: float) -> str:
        """Detect overall emotional trend."""
        # Check for oscillation (alternating signs in recent history)
        if len(self._history) >= 4:
            recent_valences = [v for _, v, _ in self._history[-4:]]
            diffs = [recent_valences[i+1] - recent_valences[i] for i in range(3)]
            signs = [d > 0 for d in diffs]
            if len(set(signs)) > 1 and signs[0] != signs[1] and signs[1] != signs[2]:
                return "oscillating"

        # Main trend based on valence velocity
        if valence_velocity > 0.02:
            return "improving"
        elif valence_velocity < -0.02:
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

    def distance_from_baseline(self) -> float:
        """
        Calculate Euclidean distance from baseline state.

        Returns:
            Distance value (0 = at baseline, sqrt(8) = maximum)
        """
        valence_diff = self.valence - self.baseline_valence
        arousal_diff = self.arousal - self.baseline_arousal
        return (valence_diff**2 + arousal_diff**2) ** 0.5

    def get_steering_coefficients(self) -> Dict[str, float]:
        """
        Convert emotional state to steering vector blend coefficients.

        Maps the 2D valence-arousal space to weights for each quadrant's
        steering vector using bilinear interpolation. The coefficients
        sum to 1.0 and represent how much of each "pure" emotional state
        to blend.

        Returns:
            Dictionary mapping emotion names to blend weights (0.0 to 1.0):

            - "excited": high valence, high arousal
            - "frustrated": low valence, high arousal
            - "calm": high valence, low arousal
            - "depleted": low valence, low arousal
        """
        # Normalize valence/arousal from [-1, 1] to [0, 1]
        v = (self.valence + 1.0) / 2.0  # 0 = negative, 1 = positive
        a = (self.arousal + 1.0) / 2.0  # 0 = low, 1 = high

        # Compute quadrant weights using bilinear interpolation
        # This gives smooth blending between adjacent emotional states
        coefficients = {
            "excited": v * a,                    # high valence, high arousal
            "frustrated": (1.0 - v) * a,         # low valence, high arousal
            "calm": v * (1.0 - a),               # high valence, low arousal
            "depleted": (1.0 - v) * (1.0 - a),   # low valence, low arousal
        }

        # Apply global steering strength
        # (allows personality-based scaling of emotional expression)
        if self.steering_strength != 1.0:
            coefficients = {
                k: v * self.steering_strength
                for k, v in coefficients.items()
            }

        return coefficients

    def get_dominant_emotion(self) -> tuple[str, float]:
        """
        Get the strongest emotional component.

        Useful for logging, debugging, or UI display.

        Returns:
            Tuple of (emotion_name, coefficient)
        """
        coefficients = self.get_steering_coefficients()
        return max(coefficients.items(), key=lambda x: x[1])
