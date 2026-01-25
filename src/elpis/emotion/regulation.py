"""Homeostatic regulation for emotional state."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger

from elpis.emotion.state import EmotionalState


# Word lists for multi-factor response analysis
SUCCESS_WORDS = frozenset([
    "successfully", "completed", "done", "fixed", "working", "solved",
    "accomplished", "achieved", "passed", "correct", "works", "resolved",
])

ERROR_WORDS = frozenset([
    "error", "failed", "cannot", "unable", "exception", "failure",
    "broken", "crash", "bug", "issue", "problem", "wrong",
])

FRUSTRATION_WORDS = frozenset([
    "still", "again", "yet", "another", "keeps", "repeatedly",
    "always", "never", "same", "stuck", "frustrat",
])

EXPLORATION_WORDS = frozenset([
    "interesting", "discover", "learn", "new", "found", "explore",
    "curious", "novel", "fascinating", "insight",
])

UNCERTAINTY_WORDS = frozenset([
    "unsure", "uncertain", "might", "perhaps", "unclear", "maybe",
    "possibly", "wonder", "confused", "difficult",
])

# Frustration pattern indicators (used with error words)
FRUSTRATION_AMPLIFIERS = frozenset([
    "still", "again", "yet another", "keeps", "repeatedly", "same",
])


@dataclass
class EventRecord:
    """Record of a single emotional event."""

    event_type: str
    timestamp: float
    intensity: float


@dataclass
class EventHistory:
    """Tracks recent events for context-aware intensity modulation."""

    events: List[EventRecord] = field(default_factory=list)
    max_age_seconds: float = 600.0  # 10 minutes
    compounding_factor: float = 0.2  # Intensity increase per repeated event
    dampening_factor: float = 0.2  # Intensity decrease per repeated success
    max_compounding: float = 2.0  # Maximum intensity multiplier
    min_dampening: float = 0.5  # Minimum intensity multiplier

    def record_event(self, event_type: str, intensity: float) -> None:
        """Record an event with timestamp."""
        self.events.append(EventRecord(
            event_type=event_type,
            timestamp=time.time(),
            intensity=intensity,
        ))
        self._trim_old_events()

    def _trim_old_events(self) -> None:
        """Remove events older than max_age_seconds."""
        cutoff = time.time() - self.max_age_seconds
        self.events = [e for e in self.events if e.timestamp > cutoff]

    def get_intensity_modifier(self, event_type: str) -> float:
        """
        Get intensity modifier based on recent event history.

        - Repeated failures/errors compound (increase intensity)
        - Repeated successes dampen (decrease intensity)
        """
        self._trim_old_events()

        # Count recent similar events
        negative_events = {"failure", "error", "test_failed", "frustration", "blocked"}
        positive_events = {"success", "test_passed", "insight"}

        recent_count = sum(1 for e in self.events if e.event_type == event_type)

        if event_type in negative_events:
            # Compound negative events: 1.0 -> 1.2 -> 1.4 -> ... (cap at max_compounding)
            modifier = 1.0 + (recent_count * self.compounding_factor)
            return min(modifier, self.max_compounding)
        elif event_type in positive_events:
            # Dampen repeated successes: 1.0 -> 0.8 -> 0.6 -> ... (floor at min_dampening)
            modifier = 1.0 - (recent_count * self.dampening_factor)
            return max(modifier, self.min_dampening)

        return 1.0  # No modification for neutral events

    def get_streak_type(self) -> Optional[str]:
        """
        Detect if there's a current streak of similar events.

        Returns:
            "failure_streak", "success_streak", or None
        """
        if len(self.events) < 2:
            return None

        negative_events = {"failure", "error", "test_failed", "frustration", "blocked"}
        positive_events = {"success", "test_passed", "insight"}

        # Check last 3 events
        recent = self.events[-3:] if len(self.events) >= 3 else self.events
        recent_types = [e.event_type for e in recent]

        if all(t in negative_events for t in recent_types):
            return "failure_streak"
        elif all(t in positive_events for t in recent_types):
            return "success_streak"

        return None


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
"""Maps event types to (valence_delta, arousal_delta) tuples."""


class HomeostasisRegulator:
    """
    Manages emotional state with homeostatic return dynamics.

    Key behaviors:
    1. Events shift valence/arousal based on type and intensity
    2. State decays toward baseline over time
    3. Extreme states decay faster (non-linear homeostasis)
    4. Context-aware intensity through event history tracking
    5. Mood inertia resists rapid emotional swings
    6. Emotion-specific decay rates per quadrant
    """

    def __init__(
        self,
        state: EmotionalState,
        decay_rate: float = 0.1,  # Per-second decay toward baseline
        max_delta: float = 0.5,  # Maximum single-event change
        # Context-aware intensity settings
        streak_compounding_enabled: bool = True,
        streak_compounding_factor: float = 0.2,
        # Mood inertia settings
        mood_inertia_enabled: bool = True,
        mood_inertia_resistance: float = 0.4,
        # Quadrant decay multipliers
        decay_multiplier_excited: float = 1.0,
        decay_multiplier_frustrated: float = 0.7,
        decay_multiplier_calm: float = 1.2,
        decay_multiplier_depleted: float = 0.8,
        # Response analysis settings
        response_analysis_threshold: float = 0.3,
    ):
        """
        Initialize the homeostasis regulator.

        Args:
            state: The EmotionalState to regulate
            decay_rate: Rate of decay toward baseline (per second)
            max_delta: Maximum allowed change from single event
            streak_compounding_enabled: Enable event compounding/dampening
            streak_compounding_factor: Intensity change per repeated event
            mood_inertia_enabled: Enable mood inertia resistance
            mood_inertia_resistance: Max resistance factor (0-1)
            decay_multiplier_*: Per-quadrant decay rate multipliers
            response_analysis_threshold: Min score to trigger emotion from response
        """
        self.state = state
        self.decay_rate = decay_rate
        self.max_delta = max_delta

        # Context-aware intensity
        self.streak_compounding_enabled = streak_compounding_enabled
        self.event_history = EventHistory(
            compounding_factor=streak_compounding_factor,
            dampening_factor=streak_compounding_factor,
        )

        # Mood inertia
        self.mood_inertia_enabled = mood_inertia_enabled
        self.mood_inertia_resistance = mood_inertia_resistance

        # Quadrant-specific decay multipliers
        self.decay_multipliers = {
            "excited": decay_multiplier_excited,
            "frustrated": decay_multiplier_frustrated,
            "calm": decay_multiplier_calm,
            "depleted": decay_multiplier_depleted,
        }

        # Response analysis
        self.response_analysis_threshold = response_analysis_threshold

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

        # Apply context-aware intensity from event history
        if self.streak_compounding_enabled:
            history_modifier = self.event_history.get_intensity_modifier(event_type)
            intensity *= history_modifier
            # Re-clamp after history modification
            intensity = max(0.0, min(2.0, intensity))

        # Apply mood inertia resistance
        if self.mood_inertia_enabled:
            inertia_modifier = self._get_inertia_modifier(valence_delta)
            intensity *= inertia_modifier

        valence_delta *= intensity
        arousal_delta *= intensity

        # Clamp to max delta
        valence_delta = max(-self.max_delta, min(self.max_delta, valence_delta))
        arousal_delta = max(-self.max_delta, min(self.max_delta, arousal_delta))

        # Record event in history (before updating state)
        if self.streak_compounding_enabled:
            self.event_history.record_event(event_type, intensity)

        # Update state
        self.state.shift(valence_delta, arousal_delta)

        logger.debug(
            f"Emotional event: {event_type} (intensity={intensity:.2f})"
            f" -> valence={self.state.valence:.3f}, arousal={self.state.arousal:.3f}"
            + (f" [{context}]" if context else "")
        )

    def _get_inertia_modifier(self, valence_delta: float) -> float:
        """
        Calculate mood inertia modifier based on current trajectory.

        Events aligned with current momentum get a slight boost (1.1x).
        Events counter to strong momentum get resistance (0.6x-0.8x).

        Args:
            valence_delta: The valence change from the event

        Returns:
            Intensity modifier (0.6 to 1.1)
        """
        trajectory = self.state.get_trajectory()

        # Neutral momentum = no modification
        if trajectory.momentum == "neutral":
            return 1.0

        # Determine if event aligns with or counters momentum
        event_is_positive = valence_delta > 0
        momentum_is_positive = trajectory.momentum == "positive"
        aligned = event_is_positive == momentum_is_positive

        if aligned:
            # Slight boost for aligned events
            return 1.1
        else:
            # Resistance based on velocity strength
            velocity_magnitude = abs(trajectory.valence_velocity)
            # Scale resistance: stronger momentum = more resistance
            # velocity_magnitude typically 0-0.5, map to resistance 0.8-0.6
            resistance = 1.0 - (self.mood_inertia_resistance * min(velocity_magnitude * 2, 1.0))
            return max(0.6, resistance)

    def process_response(self, content: str) -> None:
        """
        Infer emotional events from generated content using multi-factor scoring.

        Scores multiple emotion indicators and triggers the dominant emotion
        if above threshold. Detects frustration patterns for compound effects.

        Args:
            content: Generated text content to analyze
        """
        content_lower = content.lower()
        words = set(content_lower.split())

        # Score each emotion category
        scores = self._score_response_content(content_lower, words)

        # Check for frustration amplification (error words + frustration patterns)
        frustration_boost = self._detect_frustration_pattern(content_lower)

        # Find dominant emotion
        if not scores:
            return

        dominant_emotion, dominant_score = max(scores.items(), key=lambda x: x[1])

        # Only trigger if above threshold
        if dominant_score < self.response_analysis_threshold:
            return

        # Map score categories to events
        event_mapping = {
            "success": "success",
            "error": "error",
            "exploration": "exploration",
            "uncertainty": "novelty",
            "frustration": "frustration",
        }

        # Handle frustration detection
        if frustration_boost > 0 and scores.get("error", 0) > 0:
            # Frustration pattern detected with errors
            self.process_event(
                "frustration",
                intensity=0.5 + (frustration_boost * 0.3),
                context="frustration pattern detected",
            )
            return

        event_type = event_mapping.get(dominant_emotion)
        if event_type:
            # Scale intensity by score (0.3 threshold -> 0.3 base, 1.0 score -> 0.8 intensity)
            intensity = 0.3 + (dominant_score * 0.5)
            self.process_event(event_type, intensity=intensity)

    def _score_response_content(
        self, content_lower: str, words: set
    ) -> Dict[str, float]:
        """
        Score content for multiple emotion indicators.

        Args:
            content_lower: Lowercase content string
            words: Set of words in content

        Returns:
            Dictionary mapping emotion category to score (0-1)
        """
        scores: Dict[str, float] = {}

        # Count matches for each category
        success_count = sum(1 for w in SUCCESS_WORDS if w in content_lower)
        error_count = sum(1 for w in ERROR_WORDS if w in content_lower)
        exploration_count = sum(1 for w in EXPLORATION_WORDS if w in content_lower)
        uncertainty_count = sum(1 for w in UNCERTAINTY_WORDS if w in content_lower)
        frustration_count = sum(1 for w in FRUSTRATION_WORDS if w in content_lower)

        # Normalize scores (diminishing returns for many matches)
        # Use sqrt to reduce impact of many matches
        if success_count > 0:
            scores["success"] = min(1.0, (success_count ** 0.5) / 2)
        if error_count > 0:
            scores["error"] = min(1.0, (error_count ** 0.5) / 2)
        if exploration_count > 0:
            scores["exploration"] = min(1.0, (exploration_count ** 0.5) / 2.5)
        if uncertainty_count > 0:
            scores["uncertainty"] = min(1.0, (uncertainty_count ** 0.5) / 3)
        if frustration_count > 0:
            scores["frustration"] = min(1.0, (frustration_count ** 0.5) / 2)

        return scores

    def _detect_frustration_pattern(self, content_lower: str) -> float:
        """
        Detect frustration patterns (repeated failure indicators).

        Looks for combinations like "still getting error", "yet another failure", etc.

        Args:
            content_lower: Lowercase content string

        Returns:
            Frustration boost factor (0-1)
        """
        boost = 0.0

        # Check for frustration amplifiers near error words
        for amplifier in FRUSTRATION_AMPLIFIERS:
            if amplifier in content_lower:
                # Check if any error word is nearby (within ~50 chars)
                amp_pos = content_lower.find(amplifier)
                nearby_text = content_lower[max(0, amp_pos - 50):amp_pos + 50 + len(amplifier)]
                if any(error in nearby_text for error in ERROR_WORDS):
                    boost += 0.3

        return min(1.0, boost)

    def _apply_decay(self) -> None:
        """Apply time-based decay toward baseline with quadrant-specific rates."""
        now = time.time()
        elapsed = now - self.state.last_update

        if elapsed <= 0:
            return

        # Get quadrant-specific decay multiplier
        quadrant = self.state.get_quadrant()
        quadrant_multiplier = self.decay_multipliers.get(quadrant, 1.0)

        # Calculate decay factor (exponential decay) with quadrant modifier
        # Higher multiplier = faster decay, lower = slower decay (emotion persists)
        effective_decay_rate = self.decay_rate * quadrant_multiplier
        decay_factor = max(0.0, 1.0 - (effective_decay_rate * elapsed))

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
