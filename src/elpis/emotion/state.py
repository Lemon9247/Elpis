"""Valence-Arousal emotional state model."""

from dataclasses import dataclass, field
from typing import Dict, Any
import time


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

    def distance_from_baseline(self) -> float:
        """
        Calculate Euclidean distance from baseline state.

        Returns:
            Distance value (0 = at baseline, sqrt(8) = maximum)
        """
        valence_diff = self.valence - self.baseline_valence
        arousal_diff = self.arousal - self.baseline_arousal
        return (valence_diff**2 + arousal_diff**2) ** 0.5
