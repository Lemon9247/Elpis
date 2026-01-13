"""Unit tests for the emotional regulation system."""

import time
import pytest

from elpis.emotion.state import EmotionalState
from elpis.emotion.regulation import HomeostasisRegulator, EVENT_MAPPINGS


class TestEmotionalState:
    """Tests for EmotionalState class."""

    def test_initial_state_is_neutral(self):
        """New emotional state should be neutral."""
        state = EmotionalState()
        assert state.valence == 0.0
        assert state.arousal == 0.0
        # At exact zero, arousal >= 0 is true, so quadrant is "excited"
        assert state.get_quadrant() == "excited"

    def test_state_bounds_enforced(self):
        """State values should be clamped to [-1, 1]."""
        state = EmotionalState(valence=2.0, arousal=-3.0)
        assert state.valence == 1.0
        assert state.arousal == -1.0

    def test_quadrant_classification(self):
        """Quadrants should be correctly classified."""
        # High arousal, high valence = excited
        state = EmotionalState(valence=0.5, arousal=0.5)
        assert state.get_quadrant() == "excited"

        # High arousal, low valence = frustrated
        state = EmotionalState(valence=-0.5, arousal=0.5)
        assert state.get_quadrant() == "frustrated"

        # Low arousal, high valence = calm
        state = EmotionalState(valence=0.5, arousal=-0.5)
        assert state.get_quadrant() == "calm"

        # Low arousal, low valence = depleted
        state = EmotionalState(valence=-0.5, arousal=-0.5)
        assert state.get_quadrant() == "depleted"

    def test_modulated_params_high_arousal(self):
        """High arousal should lower temperature (focused)."""
        state = EmotionalState(arousal=1.0)
        params = state.get_modulated_params()
        assert params["temperature"] < 0.7  # Base is 0.7

    def test_modulated_params_low_arousal(self):
        """Low arousal should raise temperature (exploratory)."""
        state = EmotionalState(arousal=-1.0)
        params = state.get_modulated_params()
        assert params["temperature"] > 0.7  # Base is 0.7

    def test_modulated_params_high_valence(self):
        """High valence should raise top_p (broader sampling)."""
        state = EmotionalState(valence=1.0)
        params = state.get_modulated_params()
        assert params["top_p"] > 0.9  # Base is 0.9

    def test_modulated_params_low_valence(self):
        """Low valence should lower top_p (conservative)."""
        state = EmotionalState(valence=-1.0)
        params = state.get_modulated_params()
        assert params["top_p"] < 0.9  # Base is 0.9

    def test_shift_updates_state(self):
        """Shift should update valence and arousal."""
        state = EmotionalState()
        state.shift(0.3, 0.2)
        assert state.valence == 0.3
        assert state.arousal == 0.2
        assert state.update_count == 1

    def test_shift_clamps_values(self):
        """Shift should clamp values to bounds."""
        state = EmotionalState(valence=0.9, arousal=0.9)
        state.shift(0.5, 0.5)
        assert state.valence == 1.0
        assert state.arousal == 1.0

    def test_reset_returns_to_baseline(self):
        """Reset should return state to baseline."""
        state = EmotionalState(baseline_valence=0.1, baseline_arousal=-0.1)
        state.shift(0.5, 0.5)
        state.reset()
        assert state.valence == 0.1
        assert state.arousal == -0.1

    def test_distance_from_baseline(self):
        """Distance calculation should be correct."""
        state = EmotionalState()
        assert state.distance_from_baseline() == 0.0

        state.shift(1.0, 0.0)
        assert state.distance_from_baseline() == 1.0

        state = EmotionalState()
        state.shift(0.6, 0.8)
        assert abs(state.distance_from_baseline() - 1.0) < 0.001

    def test_to_dict_contains_all_fields(self):
        """to_dict should contain all required fields."""
        state = EmotionalState(valence=0.5, arousal=-0.3)
        d = state.to_dict()

        assert "valence" in d
        assert "arousal" in d
        assert "quadrant" in d
        assert "last_update" in d
        assert "update_count" in d
        assert "baseline" in d


class TestHomeostasisRegulator:
    """Tests for HomeostasisRegulator class."""

    def test_process_known_event(self):
        """Known events should shift state correctly."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(state)

        regulator.process_event("success")

        # Success should increase valence and arousal
        assert state.valence > 0
        assert state.arousal > 0

    def test_process_unknown_event(self):
        """Unknown events should cause mild arousal increase."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(state)

        regulator.process_event("unknown_event_xyz")

        assert state.valence == 0.0
        assert state.arousal > 0  # Mild alertness

    def test_intensity_scaling(self):
        """Event intensity should scale the effect."""
        state1 = EmotionalState()
        regulator1 = HomeostasisRegulator(state1)
        regulator1.process_event("success", intensity=1.0)
        val1 = state1.valence

        state2 = EmotionalState()
        regulator2 = HomeostasisRegulator(state2)
        regulator2.process_event("success", intensity=0.5)
        val2 = state2.valence

        assert val1 > val2  # Higher intensity = bigger effect

    def test_intensity_clamped(self):
        """Intensity should be clamped to [0, 2]."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(state)

        # Extreme intensity should be clamped
        regulator.process_event("success", intensity=10.0)
        val_high = state.valence

        state.reset()
        regulator.process_event("success", intensity=2.0)
        val_normal = state.valence

        # Both should produce same result (clamped at 2.0)
        assert val_high == val_normal

    def test_max_delta_enforced(self):
        """Single event change should be limited by max_delta."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(state, max_delta=0.1)

        regulator.process_event("success", intensity=2.0)

        assert state.valence <= 0.1
        assert state.arousal <= 0.1

    def test_process_response_success(self):
        """Response with success indicators should trigger positive emotion."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(state)

        regulator.process_response("Task completed successfully!")

        assert state.valence > 0

    def test_process_response_error(self):
        """Response with error indicators should trigger negative emotion."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(state)

        regulator.process_response("Error: unable to complete the task")

        assert state.valence < 0
        assert state.arousal > 0  # Errors increase arousal

    def test_decay_toward_baseline(self):
        """State should decay toward baseline over time."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(state, decay_rate=0.5)

        # Set to excited state
        state.shift(1.0, 1.0)
        original_distance = state.distance_from_baseline()

        # Simulate time passing
        state.last_update = time.time() - 1.0  # 1 second ago

        # Process any event to trigger decay
        regulator.process_event("idle")

        # State should be closer to baseline
        assert state.distance_from_baseline() < original_distance

    def test_get_available_events(self):
        """Should return all available event types."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(state)

        events = regulator.get_available_events()

        assert "success" in events
        assert "failure" in events
        assert "frustration" in events
        assert "novelty" in events


class TestEventMappings:
    """Tests for EVENT_MAPPINGS configuration."""

    def test_success_events_increase_valence(self):
        """Success events should increase valence."""
        success_events = ["success", "test_passed", "insight"]
        for event in success_events:
            valence_delta, _ = EVENT_MAPPINGS[event]
            assert valence_delta > 0, f"{event} should increase valence"

    def test_failure_events_decrease_valence(self):
        """Failure events should decrease valence."""
        failure_events = ["failure", "test_failed", "error", "frustration", "blocked"]
        for event in failure_events:
            valence_delta, _ = EVENT_MAPPINGS[event]
            assert valence_delta < 0, f"{event} should decrease valence"

    def test_all_events_have_tuples(self):
        """All events should have (valence, arousal) tuples."""
        for event, mapping in EVENT_MAPPINGS.items():
            assert isinstance(mapping, tuple), f"{event} should be a tuple"
            assert len(mapping) == 2, f"{event} should have 2 elements"
            valence, arousal = mapping
            assert isinstance(valence, (int, float)), f"{event} valence should be numeric"
            assert isinstance(arousal, (int, float)), f"{event} arousal should be numeric"
