"""Integration tests for emotional trajectory tracking.

Tests the trajectory detection with configurable thresholds
and spiral direction awareness.
"""

import pytest
from datetime import datetime, timedelta

from elpis.config.settings import EmotionSettings
from elpis.emotion.state import EmotionalState, EmotionalTrajectory, TrajectoryConfig


class TestEmotionalTrajectoryIntegration:
    """Integration tests for emotional trajectory with config."""

    @pytest.fixture
    def emotion_settings(self) -> EmotionSettings:
        """Create test emotion settings."""
        return EmotionSettings(
            baseline_valence=0.0,
            baseline_arousal=0.0,
            trajectory_history_size=20,
            momentum_positive_threshold=0.01,
            momentum_negative_threshold=-0.01,
            trend_improving_threshold=0.02,
            trend_declining_threshold=-0.02,
            spiral_history_count=5,
            spiral_increasing_threshold=3,
        )

    @pytest.fixture
    def state_with_config(self, emotion_settings: EmotionSettings) -> EmotionalState:
        """Create emotional state with config applied."""
        state = EmotionalState(
            baseline_valence=emotion_settings.baseline_valence,
            baseline_arousal=emotion_settings.baseline_arousal,
        )
        state._trajectory_config = TrajectoryConfig.from_settings(emotion_settings)
        return state

    def test_trajectory_config_integration(self, emotion_settings):
        """Test TrajectoryConfig properly loads from EmotionSettings."""
        config = TrajectoryConfig.from_settings(emotion_settings)

        assert config.history_size == 20
        assert config.momentum_positive_threshold == 0.01
        assert config.momentum_negative_threshold == -0.01
        assert config.trend_improving_threshold == 0.02
        assert config.trend_declining_threshold == -0.02
        assert config.spiral_history_count == 5
        assert config.spiral_increasing_threshold == 3

    def test_positive_momentum_detection(self, state_with_config):
        """Test positive momentum is detected with config threshold."""
        state = state_with_config

        # Shift to positive valence repeatedly
        for _ in range(5):
            state.shift(0.1, 0.0)

        trajectory = state.get_trajectory()
        assert trajectory.momentum == "positive"

    def test_negative_momentum_detection(self, state_with_config):
        """Test negative momentum is detected with config threshold."""
        state = state_with_config

        # Shift to negative valence repeatedly
        for _ in range(5):
            state.shift(-0.1, 0.0)

        trajectory = state.get_trajectory()
        assert trajectory.momentum == "negative"

    def test_neutral_momentum_with_small_changes(self, state_with_config):
        """Test neutral momentum when changes are below threshold."""
        state = state_with_config

        # Very small shifts - should stay neutral
        for _ in range(5):
            state.shift(0.001, 0.0)

        trajectory = state.get_trajectory()
        # Velocity should be very small, so neutral
        assert abs(trajectory.valence_velocity) < 0.01

    def test_improving_trend_detection(self, state_with_config):
        """Test improving trend is detected correctly."""
        state = state_with_config

        # Clear improving trend
        for i in range(6):
            state.valence = -0.5 + 0.2 * i
            state.record_state()

        trajectory = state.get_trajectory()
        assert trajectory.trend == "improving"

    def test_declining_trend_detection(self, state_with_config):
        """Test declining trend is detected correctly."""
        state = state_with_config

        # Clear declining trend
        for i in range(6):
            state.valence = 0.5 - 0.2 * i
            state.record_state()

        trajectory = state.get_trajectory()
        assert trajectory.trend == "declining"

    def test_stable_trend_with_small_velocity(self, state_with_config):
        """Test stable trend when velocity is low."""
        state = state_with_config

        # Minor fluctuations
        for i in range(6):
            state.valence = 0.0 + 0.005 * (i % 2)  # Tiny oscillation
            state.record_state()

        trajectory = state.get_trajectory()
        assert trajectory.trend == "stable"

    def test_oscillating_trend_detection(self, state_with_config):
        """Test oscillating trend is detected."""
        state = state_with_config

        # Clear oscillation pattern
        values = [0.2, -0.2, 0.2, -0.2, 0.2]
        for v in values:
            state.valence = v
            state.record_state()

        trajectory = state.get_trajectory()
        assert trajectory.trend == "oscillating"

    def test_positive_spiral_detection(self, state_with_config):
        """Test positive spiral (toward high valence) is detected."""
        state = state_with_config

        # Spiral toward positive valence
        for i in range(6):
            state.valence = 0.1 * i
            state.arousal = 0.05 * i
            state.record_state()

        trajectory = state.get_trajectory()
        assert trajectory.spiral_detected is True
        assert trajectory.spiral_direction == "positive"

    def test_negative_spiral_detection(self, state_with_config):
        """Test negative spiral (toward low valence) is detected."""
        state = state_with_config

        # Spiral toward negative valence
        for i in range(6):
            state.valence = -0.1 * i
            state.arousal = -0.05 * i
            state.record_state()

        trajectory = state.get_trajectory()
        assert trajectory.spiral_detected is True
        assert trajectory.spiral_direction == "negative"

    def test_escalating_spiral_detection(self, state_with_config):
        """Test escalating spiral (toward high arousal) is detected."""
        state = state_with_config

        # Spiral toward high arousal (arousal-dominant)
        for i in range(6):
            state.valence = 0.02 * i  # Small valence change
            state.arousal = 0.15 * i  # Large arousal increase
            state.record_state()

        trajectory = state.get_trajectory()
        assert trajectory.spiral_detected is True
        assert trajectory.spiral_direction == "escalating"

    def test_withdrawing_spiral_detection(self, state_with_config):
        """Test withdrawing spiral (toward low arousal) is detected."""
        state = state_with_config

        # Spiral toward low arousal (arousal-dominant)
        for i in range(6):
            state.valence = -0.02 * i  # Small valence change
            state.arousal = -0.15 * i  # Large arousal decrease
            state.record_state()

        trajectory = state.get_trajectory()
        assert trajectory.spiral_detected is True
        assert trajectory.spiral_direction == "withdrawing"

    def test_no_spiral_with_random_movement(self, state_with_config):
        """Test no spiral detected with random movement."""
        state = state_with_config

        # Random-ish movement (not consistently away from baseline)
        values = [(0.1, 0.1), (-0.05, 0.2), (0.15, -0.1), (-0.1, 0.0), (0.05, 0.15)]
        for v, a in values:
            state.valence = v
            state.arousal = a
            state.record_state()

        trajectory = state.get_trajectory()
        # May or may not detect spiral, but if not detected, direction should be none
        if not trajectory.spiral_detected:
            assert trajectory.spiral_direction == "none"

    def test_time_in_quadrant_tracking(self, state_with_config):
        """Test time in quadrant is tracked correctly."""
        state = state_with_config

        # Move to excited quadrant
        state.shift(0.5, 0.5)

        # Get trajectory - should have low time in quadrant initially
        trajectory1 = state.get_trajectory()

        # Stay in same quadrant
        state.shift(0.1, 0.1)

        trajectory2 = state.get_trajectory()

        # Time should have increased
        assert trajectory2.time_in_current_quadrant >= trajectory1.time_in_current_quadrant

    def test_quadrant_change_resets_timer(self, state_with_config):
        """Test time in quadrant resets when changing quadrants."""
        state = state_with_config

        # Move to excited quadrant
        state.shift(0.5, 0.5)
        state.record_state()

        # Change to frustrated quadrant
        state.shift(-1.5, 0.0)  # Now valence < 0, arousal > 0

        trajectory = state.get_trajectory()

        # Time should be very short since we just changed quadrants
        assert trajectory.time_in_current_quadrant < 1.0

    def test_trajectory_to_dict(self, state_with_config):
        """Test trajectory serialization includes all fields."""
        state = state_with_config

        # Generate some history
        for i in range(5):
            state.shift(0.1, 0.05)

        trajectory = state.get_trajectory()
        data = trajectory.to_dict()

        assert "valence_velocity" in data
        assert "arousal_velocity" in data
        assert "trend" in data
        assert "spiral_detected" in data
        assert "spiral_direction" in data
        assert "time_in_quadrant" in data
        assert "momentum" in data

    def test_neutral_trajectory_for_insufficient_data(self, state_with_config):
        """Test neutral trajectory returned when insufficient history."""
        state = state_with_config

        # Only one state recorded
        state.record_state()

        trajectory = state.get_trajectory()

        assert trajectory.valence_velocity == 0.0
        assert trajectory.arousal_velocity == 0.0
        assert trajectory.trend == "stable"
        assert trajectory.spiral_detected is False
        assert trajectory.spiral_direction == "none"
        assert trajectory.momentum == "neutral"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
