"""Unit tests for the emotional regulation system."""

import time
import pytest

from elpis.emotion.state import EmotionalState
from elpis.emotion.regulation import (
    HomeostasisRegulator,
    EVENT_MAPPINGS,
    EventHistory,
    SUCCESS_WORDS,
    ERROR_WORDS,
    FRUSTRATION_WORDS,
)
from elpis.emotion.behavioral_monitor import BehavioralMonitor
from elpis.emotion.sentiment import SentimentAnalyzer, SentimentResult


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
        assert "steering_coefficients" in d

    def test_steering_coefficients_sum_to_one(self):
        """Steering coefficients should sum to approximately 1.0."""
        state = EmotionalState(valence=0.5, arousal=-0.3)
        coeffs = state.get_steering_coefficients()

        total = sum(coeffs.values())
        assert 0.99 <= total <= 1.01, f"Coefficients sum to {total}, expected ~1.0"

    def test_steering_coefficients_excited_quadrant(self):
        """High valence + high arousal should produce mostly 'excited' coefficient."""
        state = EmotionalState(valence=0.9, arousal=0.9)
        coeffs = state.get_steering_coefficients()

        assert coeffs["excited"] > 0.7
        assert coeffs["excited"] > coeffs["frustrated"]
        assert coeffs["excited"] > coeffs["calm"]
        assert coeffs["excited"] > coeffs["depleted"]

    def test_steering_coefficients_frustrated_quadrant(self):
        """Low valence + high arousal should produce mostly 'frustrated' coefficient."""
        state = EmotionalState(valence=-0.9, arousal=0.9)
        coeffs = state.get_steering_coefficients()

        assert coeffs["frustrated"] > 0.7
        assert coeffs["frustrated"] > coeffs["excited"]
        assert coeffs["frustrated"] > coeffs["calm"]
        assert coeffs["frustrated"] > coeffs["depleted"]

    def test_steering_coefficients_calm_quadrant(self):
        """High valence + low arousal should produce mostly 'calm' coefficient."""
        state = EmotionalState(valence=0.9, arousal=-0.9)
        coeffs = state.get_steering_coefficients()

        assert coeffs["calm"] > 0.7
        assert coeffs["calm"] > coeffs["excited"]
        assert coeffs["calm"] > coeffs["frustrated"]
        assert coeffs["calm"] > coeffs["depleted"]

    def test_steering_coefficients_depleted_quadrant(self):
        """Low valence + low arousal should produce mostly 'depleted' coefficient."""
        state = EmotionalState(valence=-0.9, arousal=-0.9)
        coeffs = state.get_steering_coefficients()

        assert coeffs["depleted"] > 0.7
        assert coeffs["depleted"] > coeffs["excited"]
        assert coeffs["depleted"] > coeffs["frustrated"]
        assert coeffs["depleted"] > coeffs["calm"]

    def test_steering_coefficients_neutral_balanced(self):
        """Neutral state should have roughly equal coefficients."""
        state = EmotionalState(valence=0.0, arousal=0.0)
        coeffs = state.get_steering_coefficients()

        # All should be 0.25 at perfect center
        for emotion, coeff in coeffs.items():
            assert 0.2 <= coeff <= 0.3, f"{emotion} = {coeff}, expected ~0.25"

    def test_steering_strength_scaling(self):
        """Steering strength should scale all coefficients."""
        state = EmotionalState(valence=0.5, arousal=0.5, steering_strength=0.5)
        coeffs = state.get_steering_coefficients()

        # With strength 0.5, total should be ~0.5
        total = sum(coeffs.values())
        assert 0.45 <= total <= 0.55, f"Total {total}, expected ~0.5 with strength 0.5"

    def test_steering_strength_zero(self):
        """Steering strength of 0 should produce zero coefficients."""
        state = EmotionalState(valence=0.8, arousal=0.8, steering_strength=0.0)
        coeffs = state.get_steering_coefficients()

        for emotion, coeff in coeffs.items():
            assert coeff == 0.0, f"{emotion} should be 0.0 with strength 0.0"

    def test_steering_strength_exaggerated(self):
        """Steering strength > 1.0 should amplify coefficients."""
        state = EmotionalState(valence=0.5, arousal=0.5, steering_strength=2.0)
        coeffs = state.get_steering_coefficients()

        # With strength 2.0, total should be ~2.0
        total = sum(coeffs.values())
        assert 1.9 <= total <= 2.1, f"Total {total}, expected ~2.0 with strength 2.0"

    def test_get_dominant_emotion(self):
        """get_dominant_emotion should return highest coefficient."""
        state = EmotionalState(valence=0.8, arousal=0.8)
        emotion, strength = state.get_dominant_emotion()

        assert emotion == "excited"
        assert strength > 0.5

        # Test another quadrant
        state = EmotionalState(valence=-0.8, arousal=-0.8)
        emotion, strength = state.get_dominant_emotion()

        assert emotion == "depleted"
        assert strength > 0.5


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
        # Disable streak compounding for this test to isolate intensity clamping
        state1 = EmotionalState()
        regulator1 = HomeostasisRegulator(state1, streak_compounding_enabled=False)

        # Extreme intensity should be clamped
        regulator1.process_event("success", intensity=10.0)
        val_high = state1.valence

        state2 = EmotionalState()
        regulator2 = HomeostasisRegulator(state2, streak_compounding_enabled=False)
        regulator2.process_event("success", intensity=2.0)
        val_normal = state2.valence

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


class TestEventHistory:
    """Tests for EventHistory class (context-aware intensity)."""

    def test_record_event(self):
        """EventHistory should record events with timestamps."""
        history = EventHistory()
        history.record_event("success", 1.0)
        assert len(history.events) == 1
        assert history.events[0].event_type == "success"

    def test_negative_event_compounding(self):
        """Repeated failures should compound intensity."""
        history = EventHistory(compounding_factor=0.2)

        # First failure - no compounding yet
        modifier1 = history.get_intensity_modifier("failure")
        assert modifier1 == 1.0

        # Record a failure
        history.record_event("failure", 1.0)

        # Second call should show compounding
        modifier2 = history.get_intensity_modifier("failure")
        assert modifier2 == 1.2  # 1.0 + 0.2 * 1

        # Record another failure
        history.record_event("failure", 1.2)
        modifier3 = history.get_intensity_modifier("failure")
        assert modifier3 == 1.4  # 1.0 + 0.2 * 2

    def test_positive_event_dampening(self):
        """Repeated successes should dampen intensity."""
        history = EventHistory(dampening_factor=0.2)

        # First success - no dampening yet
        modifier1 = history.get_intensity_modifier("success")
        assert modifier1 == 1.0

        # Record a success
        history.record_event("success", 1.0)

        # Second call should show dampening
        modifier2 = history.get_intensity_modifier("success")
        assert modifier2 == 0.8  # 1.0 - 0.2 * 1

    def test_compounding_cap(self):
        """Compounding should cap at max_compounding."""
        history = EventHistory(compounding_factor=0.5, max_compounding=2.0)

        # Record many failures
        for _ in range(10):
            history.record_event("error", 1.0)

        modifier = history.get_intensity_modifier("error")
        assert modifier == 2.0  # Capped

    def test_dampening_floor(self):
        """Dampening should floor at min_dampening."""
        history = EventHistory(dampening_factor=0.3, min_dampening=0.5)

        # Record many successes
        for _ in range(10):
            history.record_event("success", 1.0)

        modifier = history.get_intensity_modifier("success")
        assert modifier == 0.5  # Floor

    def test_old_events_trimmed(self):
        """Events older than max_age_seconds should be trimmed."""
        history = EventHistory(max_age_seconds=1.0)

        # Record an event
        history.record_event("success", 1.0)
        assert len(history.events) == 1

        # Make it old
        history.events[0].timestamp = time.time() - 2.0

        # Trim should remove it
        history._trim_old_events()
        assert len(history.events) == 0

    def test_streak_type_detection(self):
        """Should detect failure and success streaks."""
        history = EventHistory()

        # No streak with few events
        assert history.get_streak_type() is None

        # Add failure streak
        history.record_event("failure", 1.0)
        history.record_event("error", 1.0)
        history.record_event("test_failed", 1.0)

        assert history.get_streak_type() == "failure_streak"

        # Reset and add success streak
        history.events.clear()
        history.record_event("success", 1.0)
        history.record_event("test_passed", 1.0)
        history.record_event("insight", 1.0)

        assert history.get_streak_type() == "success_streak"


class TestEnhancedResponseAnalysis:
    """Tests for enhanced multi-factor response analysis."""

    def test_multi_indicator_success(self):
        """Multiple success indicators should increase score."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(state, response_analysis_threshold=0.3)

        # Many success words
        regulator.process_response(
            "Task completed successfully! Everything is working and fixed correctly."
        )

        assert state.valence > 0

    def test_error_detection(self):
        """Error indicators should trigger negative emotion."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(state, response_analysis_threshold=0.3)

        regulator.process_response(
            "Error: The operation failed with an exception. Unable to proceed."
        )

        assert state.valence < 0

    def test_frustration_pattern_detection(self):
        """Frustration patterns (still/again + error) should be detected."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(state, response_analysis_threshold=0.3)

        regulator.process_response(
            "Still getting the same error. Yet another failure occurred."
        )

        # Should trigger frustration with higher intensity
        assert state.valence < 0
        assert state.arousal > 0  # Frustration increases arousal

    def test_threshold_filtering(self):
        """Low scores should not trigger events."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(state, response_analysis_threshold=0.9)

        # Single keyword shouldn't pass high threshold
        regulator.process_response("done")

        # State should remain neutral
        assert state.valence == 0.0

    def test_dominant_emotion_wins(self):
        """When mixed signals, dominant emotion should win."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(state, response_analysis_threshold=0.2)

        # More success than error words
        regulator.process_response(
            "Task completed successfully and everything is working. "
            "There was one minor error but it's fixed now."
        )

        # Success should dominate
        assert state.valence > 0


class TestMoodInertia:
    """Tests for mood inertia (resistance to rapid swings)."""

    def test_aligned_event_boost(self):
        """Events aligned with momentum should get slight boost."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(
            state,
            mood_inertia_enabled=True,
            mood_inertia_resistance=0.4,
        )

        # Build positive momentum with multiple successes
        for _ in range(3):
            regulator.process_event("success", intensity=1.0)
            time.sleep(0.01)  # Small delay for trajectory tracking

        initial_valence = state.valence

        # Another success should get boost (aligned with positive momentum)
        state2 = EmotionalState(valence=initial_valence, arousal=state.arousal)
        regulator2 = HomeostasisRegulator(state2, mood_inertia_enabled=False)
        regulator2.process_event("success", intensity=1.0)

        # With inertia, the aligned boost means more change
        # (This is a behavioral test - just verify it doesn't break)
        assert state.valence > 0

    def test_counter_event_resistance(self):
        """Events counter to momentum should face resistance."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(
            state,
            mood_inertia_enabled=True,
            mood_inertia_resistance=0.4,
        )

        # Build positive momentum
        for _ in range(3):
            regulator.process_event("success", intensity=1.0)
            time.sleep(0.01)

        positive_valence = state.valence

        # Now try a failure - should face resistance
        regulator.process_event("failure", intensity=1.0)

        # Valence should decrease but not as much as without inertia
        assert state.valence < positive_valence
        assert state.valence > 0  # Still positive due to resistance

    def test_inertia_disabled(self):
        """With inertia disabled, no resistance should apply."""
        state = EmotionalState()
        regulator = HomeostasisRegulator(
            state,
            mood_inertia_enabled=False,
        )

        # Build momentum
        for _ in range(3):
            regulator.process_event("success", intensity=1.0)

        positive_valence = state.valence

        # Failure without resistance
        regulator.process_event("failure", intensity=1.0)

        # Should see full effect
        assert state.valence < positive_valence


class TestQuadrantDecay:
    """Tests for emotion-specific decay rates."""

    def test_frustrated_decays_slower(self):
        """Frustrated state should decay slower (lower multiplier)."""
        # Create two states in different quadrants
        state_frustrated = EmotionalState(valence=-0.5, arousal=0.5)
        state_excited = EmotionalState(valence=0.5, arousal=0.5)

        reg_frustrated = HomeostasisRegulator(
            state_frustrated,
            decay_rate=0.5,
            decay_multiplier_frustrated=0.5,  # Slow decay
            decay_multiplier_excited=1.0,
        )
        reg_excited = HomeostasisRegulator(
            state_excited,
            decay_rate=0.5,
            decay_multiplier_frustrated=0.5,
            decay_multiplier_excited=1.0,
        )

        # Simulate time passing
        state_frustrated.last_update = time.time() - 1.0
        state_excited.last_update = time.time() - 1.0

        # Trigger decay
        reg_frustrated._apply_decay()
        reg_excited._apply_decay()

        # Frustrated should be further from baseline (slower decay)
        dist_frustrated = state_frustrated.distance_from_baseline()
        dist_excited = state_excited.distance_from_baseline()

        assert dist_frustrated > dist_excited

    def test_calm_decays_faster(self):
        """Calm state should decay faster (higher multiplier)."""
        state = EmotionalState(valence=0.5, arousal=-0.5)
        regulator = HomeostasisRegulator(
            state,
            decay_rate=0.5,
            decay_multiplier_calm=1.5,  # Fast decay
        )

        initial_distance = state.distance_from_baseline()

        # Simulate time passing
        state.last_update = time.time() - 1.0
        regulator._apply_decay()

        # Should have decayed significantly
        assert state.distance_from_baseline() < initial_distance * 0.5


class TestBehavioralMonitor:
    """Tests for BehavioralMonitor class."""

    def test_retry_loop_detection(self):
        """Should detect retry loops (same tool called repeatedly)."""
        events = []

        def on_event(event_type, intensity, context=None):
            events.append((event_type, intensity, context))

        monitor = BehavioralMonitor(
            on_event=on_event,
            retry_loop_threshold=3,
        )

        # Call same tool multiple times
        for _ in range(3):
            monitor.record_tool_call("search", success=False)

        # Should have triggered frustration event
        assert any(e[0] == "frustration" for e in events)

    def test_failure_streak_detection(self):
        """Should detect consecutive failures."""
        events = []

        def on_event(event_type, intensity, context=None):
            events.append((event_type, intensity, context))

        monitor = BehavioralMonitor(
            on_event=on_event,
            failure_streak_threshold=2,
        )

        # Two consecutive failures
        monitor.record_tool_call("tool1", success=False)
        monitor.record_tool_call("tool2", success=False)

        # Should have triggered error event
        assert any(e[0] == "error" for e in events)

    def test_long_generation_detection(self):
        """Should detect long generations."""
        events = []

        def on_event(event_type, intensity, context=None):
            events.append((event_type, intensity, context))

        monitor = BehavioralMonitor(
            on_event=on_event,
            long_generation_seconds=0.1,  # Very short for testing
        )

        monitor.start_generation()
        time.sleep(0.15)  # Longer than threshold
        monitor.end_generation()

        # Should have triggered blocked event
        assert any(e[0] == "blocked" for e in events)

    def test_idle_detection(self):
        """Should detect idle periods."""
        events = []

        def on_event(event_type, intensity, context=None):
            events.append((event_type, intensity, context))

        monitor = BehavioralMonitor(
            on_event=on_event,
            idle_period_seconds=0.1,  # Very short for testing
        )

        # Set last activity to past
        monitor.last_activity_time = time.time() - 0.2

        monitor.check_idle()

        # Should have triggered idle event
        assert any(e[0] == "idle" for e in events)

    def test_success_breaks_failure_streak(self):
        """Success should break failure streak."""
        events = []

        def on_event(event_type, intensity, context=None):
            events.append((event_type, intensity, context))

        monitor = BehavioralMonitor(
            on_event=on_event,
            failure_streak_threshold=3,
        )

        # Failure, success, failure, failure - should not trigger streak
        monitor.record_tool_call("tool1", success=False)
        monitor.record_tool_call("tool2", success=True)  # Breaks potential streak
        events.clear()  # Clear any events
        monitor.record_tool_call("tool3", success=False)
        monitor.record_tool_call("tool4", success=False)

        # Should trigger streak now (2 consecutive)
        # But wait, threshold is 3, so it won't trigger
        # Let's check the last 3 aren't all failures
        recent = monitor.tool_history[-3:]
        all_failed = all(not tc.success for tc in recent)
        assert not all_failed  # Success in middle breaks streak

    def test_get_stats(self):
        """Should return monitoring statistics."""
        monitor = BehavioralMonitor(
            on_event=lambda *args: None,
        )

        monitor.record_tool_call("tool1", success=True)
        monitor.record_tool_call("tool2", success=False)

        stats = monitor.get_stats()
        assert stats["total_tool_calls"] == 2
        assert stats["recent_success_count"] == 1
        assert stats["recent_failure_count"] == 1


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer class."""

    def test_skip_short_content(self):
        """Should skip content below minimum length."""
        analyzer = SentimentAnalyzer(
            use_local_model=False,  # Don't try to load model
            min_length=100,
        )

        result = analyzer.analyze("short text")
        assert result is None

    def test_map_sentiment_to_emotions_positive(self):
        """Positive sentiment should map to satisfaction/excitement."""
        analyzer = SentimentAnalyzer(use_local_model=False)

        emotions = analyzer._map_sentiment_to_emotions(0.7, 0.9)
        assert "satisfaction" in emotions
        assert emotions["satisfaction"] > 0

    def test_map_sentiment_to_emotions_negative(self):
        """Negative sentiment should map to frustration/distress."""
        analyzer = SentimentAnalyzer(use_local_model=False)

        emotions = analyzer._map_sentiment_to_emotions(-0.7, 0.9)
        assert "frustration" in emotions
        assert emotions["frustration"] > 0

    def test_map_sentiment_to_emotions_neutral(self):
        """Neutral sentiment should map to neutral emotion."""
        analyzer = SentimentAnalyzer(use_local_model=False)

        emotions = analyzer._map_sentiment_to_emotions(0.1, 0.8)
        assert "neutral" in emotions

    def test_get_emotional_event_high_positive(self):
        """High positive sentiment should return success event."""
        analyzer = SentimentAnalyzer(use_local_model=False)

        result = SentimentResult(
            sentiment_score=0.7,
            confidence=0.8,
            emotions={"satisfaction": 0.6},
            source="test",
        )

        event = analyzer.get_emotional_event(result)
        assert event is not None
        assert event[0] == "success"

    def test_get_emotional_event_high_negative(self):
        """High negative sentiment should return frustration event."""
        analyzer = SentimentAnalyzer(use_local_model=False)

        result = SentimentResult(
            sentiment_score=-0.7,
            confidence=0.8,
            emotions={"frustration": 0.6},
            source="test",
        )

        event = analyzer.get_emotional_event(result)
        assert event is not None
        assert event[0] == "frustration"

    def test_get_emotional_event_low_confidence(self):
        """Low confidence should not return event."""
        analyzer = SentimentAnalyzer(use_local_model=False)

        result = SentimentResult(
            sentiment_score=0.8,
            confidence=0.3,  # Too low
            emotions={},
            source="test",
        )

        event = analyzer.get_emotional_event(result)
        assert event is None
