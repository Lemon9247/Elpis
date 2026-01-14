"""Homeostatic regulation for emotional state."""

import time
from typing import Dict, Optional, Tuple
from loguru import logger

from elpis.emotion.state import EmotionalState


# Event type -> (valence_delta, arousal_delta)
EVENT_MAPPINGS: Dict[str, Tuple[float, float]] = {
    # Success events (positive valence, variable arousal)
    "success": (0.2, 0.1),  # Task completed
    "test_passed": (0.15, 0.05),  # Tests pass
    "insight": (0.25, 0.2),  # Novel solution found

    # Failure events (negative valence, high arousal)
    "failure": (-0.2, 0.15),  # Task failed
    "test_failed": (-0.15, 0.1),  # Tests fail
    "error": (-0.1, 0.2),  # Unexpected error

    # Frustration events (negative valence, high arousal)
    "frustration": (-0.15, 0.25),  # Repeated failures
    "blocked": (-0.2, 0.3),  # Can't proceed

    # Calm events (neutral/positive valence, low arousal)
    "idle": (0.0, -0.1),  # Waiting for input
    "routine": (0.05, -0.05),  # Familiar task

    # Novelty events (positive valence, high arousal)
    "novelty": (0.1, 0.2),  # New domain/task
    "exploration": (0.15, 0.15),  # Learning new things

    # User interaction events
    "user_positive": (0.15, 0.1),  # User gives positive feedback
    "user_negative": (-0.1, 0.15),  # User gives negative feedback
    "user_question": (0.05, 0.1),  # User asks question
}


class HomeostasisRegulator:
    """
    Manages emotional state with homeostatic return dynamics.

    Key behaviors:
    1. Events shift valence/arousal based on type and intensity
    2. State decays toward baseline over time
    3. Extreme states decay faster (non-linear homeostasis)
    """

    def __init__(
        self,
        state: EmotionalState,
        decay_rate: float = 0.1,  # Per-second decay toward baseline
        max_delta: float = 0.5,  # Maximum single-event change
    ):
        """
        Initialize the homeostasis regulator.

        Args:
            state: The EmotionalState to regulate
            decay_rate: Rate of decay toward baseline (per second)
            max_delta: Maximum allowed change from single event
        """
        self.state = state
        self.decay_rate = decay_rate
        self.max_delta = max_delta

    def process_event(
        self,
        event_type: str,
        intensity: float = 1.0,
        context: Optional[str] = None,
    ) -> None:
        """
        Process an emotional event and update state.

        Args:
            event_type: Type of event (see EVENT_MAPPINGS)
            intensity: Event intensity multiplier (0.0 to 2.0)
            context: Optional description for logging
        """
        # Apply time-based decay first
        self._apply_decay()

        # Look up event mapping
        if event_type not in EVENT_MAPPINGS:
            # Unknown event - mild arousal increase (alertness)
            logger.debug(f"Unknown event type: {event_type}, using default")
            valence_delta, arousal_delta = 0.0, 0.05
        else:
            valence_delta, arousal_delta = EVENT_MAPPINGS[event_type]

        # Apply intensity modifier (clamped to 0-2)
        intensity = max(0.0, min(2.0, intensity))
        valence_delta *= intensity
        arousal_delta *= intensity

        # Clamp to max delta
        valence_delta = max(-self.max_delta, min(self.max_delta, valence_delta))
        arousal_delta = max(-self.max_delta, min(self.max_delta, arousal_delta))

        # Update state
        self.state.shift(valence_delta, arousal_delta)

        logger.debug(
            f"Emotional event: {event_type} (intensity={intensity:.2f})"
            f" -> valence={self.state.valence:.3f}, arousal={self.state.arousal:.3f}"
            + (f" [{context}]" if context else "")
        )

    def process_response(self, content: str) -> None:
        """
        Infer emotional events from generated content.

        Simple heuristics - can be enhanced with sentiment analysis later.

        Args:
            content: Generated text content to analyze
        """
        content_lower = content.lower()

        # Success indicators
        if any(
            w in content_lower
            for w in ["successfully", "completed", "done", "fixed", "working"]
        ):
            self.process_event("success", intensity=0.5)
            return

        # Error indicators
        if any(
            w in content_lower
            for w in ["error", "failed", "cannot", "unable", "exception"]
        ):
            self.process_event("error", intensity=0.5)
            return

        # Exploration indicators
        if any(
            w in content_lower
            for w in ["interesting", "discover", "learn", "new", "found"]
        ):
            self.process_event("exploration", intensity=0.3)
            return

        # Uncertainty indicators
        if any(
            w in content_lower
            for w in ["unsure", "uncertain", "might", "perhaps", "unclear"]
        ):
            self.process_event("novelty", intensity=0.2)
            return

    def _apply_decay(self) -> None:
        """Apply time-based decay toward baseline."""
        now = time.time()
        elapsed = now - self.state.last_update

        if elapsed <= 0:
            return

        # Calculate decay factor (exponential decay)
        # decay_factor approaches 0 as time increases
        decay_factor = max(0.0, 1.0 - (self.decay_rate * elapsed))

        # Decay toward baseline
        valence_diff = self.state.valence - self.state.baseline_valence
        arousal_diff = self.state.arousal - self.state.baseline_arousal

        self.state.valence = self.state.baseline_valence + (valence_diff * decay_factor)
        self.state.arousal = self.state.baseline_arousal + (arousal_diff * decay_factor)
        self.state.last_update = now

    def get_available_events(self) -> Dict[str, Tuple[float, float]]:
        """
        Return available event types and their effects.

        Returns:
            Dictionary mapping event type to (valence_delta, arousal_delta)
        """
        return EVENT_MAPPINGS.copy()
