"""Behavioral monitoring for emotional state inference.

Detects patterns in tool usage and generation behavior to trigger
appropriate emotional events:
- Retry loops (same tool called repeatedly)
- Failure streaks (consecutive failures)
- Long generations (potentially blocked/struggling)
- Idle periods (calming effect)
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from loguru import logger


@dataclass
class ToolCall:
    """Record of a single tool call."""

    tool_name: str
    timestamp: float
    success: bool
    duration: Optional[float] = None


@dataclass
class GenerationRecord:
    """Record of a generation event."""

    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None


class BehavioralMonitor:
    """
    Monitors behavioral patterns to trigger emotional events.

    Detects:
    - Retry loops: Same tool called 3+ times in succession
    - Failure streaks: 2+ consecutive failures
    - Long generations: Generation taking >30s
    - Idle periods: No activity for >2 minutes
    """

    def __init__(
        self,
        on_event: Callable[[str, float, Optional[str]], None],
        retry_loop_threshold: int = 3,
        failure_streak_threshold: int = 2,
        long_generation_seconds: float = 30.0,
        idle_period_seconds: float = 120.0,
        history_max_size: int = 50,
    ):
        """
        Initialize the behavioral monitor.

        Args:
            on_event: Callback to trigger emotional events (event_type, intensity, context)
            retry_loop_threshold: Number of same-tool calls to detect retry loop
            failure_streak_threshold: Number of consecutive failures for compounding
            long_generation_seconds: Duration to consider generation "long"
            idle_period_seconds: Duration without activity for idle event
            history_max_size: Maximum tool calls to keep in history
        """
        self.on_event = on_event
        self.retry_loop_threshold = retry_loop_threshold
        self.failure_streak_threshold = failure_streak_threshold
        self.long_generation_seconds = long_generation_seconds
        self.idle_period_seconds = idle_period_seconds
        self.history_max_size = history_max_size

        self.tool_history: List[ToolCall] = []
        self.current_generation: Optional[GenerationRecord] = None
        self.last_activity_time: float = time.time()

        # Track detected patterns to avoid repeated triggers
        self._last_retry_loop_trigger: float = 0.0
        self._last_idle_trigger: float = 0.0
        self._retry_loop_cooldown: float = 30.0  # Don't trigger again for 30s
        self._idle_cooldown: float = 60.0  # Don't trigger idle again for 60s

    def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        duration: Optional[float] = None,
    ) -> None:
        """
        Record a tool call and check for behavioral patterns.

        Args:
            tool_name: Name of the tool that was called
            success: Whether the tool call succeeded
            duration: How long the tool call took (if known)
        """
        now = time.time()
        self.last_activity_time = now

        self.tool_history.append(ToolCall(
            tool_name=tool_name,
            timestamp=now,
            success=success,
            duration=duration,
        ))

        # Trim history
        if len(self.tool_history) > self.history_max_size:
            self.tool_history = self.tool_history[-self.history_max_size:]

        # Check for patterns
        self._check_retry_loop()
        self._check_failure_streak()

    def start_generation(self) -> None:
        """Mark the start of a generation."""
        self.current_generation = GenerationRecord(start_time=time.time())
        self.last_activity_time = time.time()

    def end_generation(self) -> None:
        """
        Mark the end of a generation and check for long duration.

        Should be called when generation completes.
        """
        if self.current_generation is None:
            return

        now = time.time()
        self.current_generation.end_time = now
        self.current_generation.duration = now - self.current_generation.start_time
        self.last_activity_time = now

        # Check for long generation
        if self.current_generation.duration >= self.long_generation_seconds:
            logger.debug(
                f"Long generation detected: {self.current_generation.duration:.1f}s"
            )
            self.on_event(
                "blocked",
                0.3,  # Mild blocked event
                f"generation took {self.current_generation.duration:.1f}s",
            )

        self.current_generation = None

    def check_idle(self) -> None:
        """
        Check if we've been idle long enough to trigger a calming event.

        Should be called periodically (e.g., in a background task).
        """
        now = time.time()
        idle_duration = now - self.last_activity_time

        if idle_duration >= self.idle_period_seconds:
            # Check cooldown
            if now - self._last_idle_trigger < self._idle_cooldown:
                return

            logger.debug(f"Idle period detected: {idle_duration:.1f}s")
            self._last_idle_trigger = now
            self.on_event(
                "idle",
                0.5,  # Moderate calming effect
                f"idle for {idle_duration:.1f}s",
            )

    def _check_retry_loop(self) -> None:
        """Check if recent tool calls form a retry loop pattern."""
        if len(self.tool_history) < self.retry_loop_threshold:
            return

        now = time.time()

        # Check cooldown
        if now - self._last_retry_loop_trigger < self._retry_loop_cooldown:
            return

        # Get recent tool calls
        recent = self.tool_history[-self.retry_loop_threshold:]

        # Check if all are the same tool
        tool_names = [tc.tool_name for tc in recent]
        if len(set(tool_names)) == 1:
            # Check if they're close together in time (within 60s span)
            time_span = recent[-1].timestamp - recent[0].timestamp
            if time_span < 60.0:
                logger.debug(f"Retry loop detected: {tool_names[0]} called {len(recent)} times")
                self._last_retry_loop_trigger = now
                self.on_event(
                    "frustration",
                    0.6,
                    f"retry loop: {tool_names[0]} x{len(recent)}",
                )

    def _check_failure_streak(self) -> None:
        """Check for consecutive failures."""
        if len(self.tool_history) < self.failure_streak_threshold:
            return

        # Get recent tool calls
        recent = self.tool_history[-self.failure_streak_threshold:]

        # Check if all failed
        if all(not tc.success for tc in recent):
            # Calculate streak length (may be longer than threshold)
            streak_length = 0
            for tc in reversed(self.tool_history):
                if not tc.success:
                    streak_length += 1
                else:
                    break

            # Only trigger if this is a new failure extending the streak
            if streak_length == self.failure_streak_threshold or (
                streak_length > self.failure_streak_threshold
                and not self.tool_history[-2].success  # Previous was also failure
            ):
                # Intensity increases with streak length
                intensity = 0.5 + (0.1 * (streak_length - self.failure_streak_threshold))
                intensity = min(1.0, intensity)

                logger.debug(f"Failure streak detected: {streak_length} consecutive failures")
                self.on_event(
                    "error",
                    intensity,
                    f"failure streak: {streak_length} consecutive",
                )

    def get_stats(self) -> Dict:
        """
        Get current monitoring statistics.

        Returns:
            Dictionary with monitoring stats
        """
        now = time.time()
        recent_calls = [tc for tc in self.tool_history if now - tc.timestamp < 300]

        success_count = sum(1 for tc in recent_calls if tc.success)
        failure_count = sum(1 for tc in recent_calls if not tc.success)

        return {
            "total_tool_calls": len(self.tool_history),
            "recent_success_count": success_count,
            "recent_failure_count": failure_count,
            "idle_duration": now - self.last_activity_time,
            "generation_in_progress": self.current_generation is not None,
        }
