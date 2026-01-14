#!/usr/bin/env python3
"""Interactive REPL for experimenting with emotional states.

Launch an interactive session to:
- Set valence and arousal values
- See resulting coefficients and parameters
- Process events and observe state changes
- Test homeostatic decay
"""

import cmd
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from elpis.emotion.state import EmotionalState
from elpis.emotion.regulation import HomeostasisRegulator


class EmotionREPL(cmd.Cmd):
    """Interactive REPL for emotion system."""

    intro = """
╔═══════════════════════════════════════════════════════════════╗
║           Elpis Emotional State REPL                          ║
║                                                               ║
║  Experiment with valence-arousal emotional regulation         ║
║  Type 'help' for commands, 'quit' to exit                    ║
╚═══════════════════════════════════════════════════════════════╝
"""
    prompt = "emotion> "

    def __init__(self):
        """Initialize REPL with neutral state."""
        super().__init__()
        self.state = EmotionalState()
        self.regulator = HomeostasisRegulator(self.state)
        self._show_state()

    def _show_state(self):
        """Display current emotional state."""
        print(f"\n{'─' * 60}")
        print(f"Valence:  {self.state.valence:+.3f}  {'█' * int((self.state.valence + 1) * 15)}")
        print(f"Arousal:  {self.state.arousal:+.3f}  {'█' * int((self.state.arousal + 1) * 15)}")
        print(f"Quadrant: {self.state.get_quadrant()}")

        dominant, strength = self.state.get_dominant_emotion()
        print(f"Dominant: {dominant} (strength: {strength:.3f})")

        params = self.state.get_modulated_params()
        print(f"\nSampling: temp={params['temperature']:.2f}, top_p={params['top_p']:.2f}")

        coeffs = self.state.get_steering_coefficients()
        print("\nSteering Coefficients:")
        for emotion, coeff in coeffs.items():
            bar = "█" * int(coeff * 40)
            print(f"  {emotion:12s}: {coeff:.3f}  {bar}")

        print(f"\nUpdates: {self.state.update_count}, "
              f"Distance from baseline: {self.state.distance_from_baseline():.3f}")
        print(f"{'─' * 60}")

    def do_set(self, arg):
        """
        Set emotional state directly.
        Usage: set <valence> <arousal>
        Example: set 0.5 -0.3
        """
        try:
            parts = arg.split()
            if len(parts) != 2:
                print("Usage: set <valence> <arousal>")
                return

            valence = float(parts[0])
            arousal = float(parts[1])

            if not (-1.0 <= valence <= 1.0) or not (-1.0 <= arousal <= 1.0):
                print("Error: Values must be in range [-1.0, 1.0]")
                return

            # Calculate deltas and apply
            v_delta = valence - self.state.valence
            a_delta = arousal - self.state.arousal
            self.state.shift(v_delta, a_delta)

            print(f"✓ State updated to ({valence:.2f}, {arousal:.2f})")
            self._show_state()

        except ValueError:
            print("Error: Invalid numbers")

    def do_shift(self, arg):
        """
        Shift emotional state by deltas.
        Usage: shift <valence_delta> <arousal_delta>
        Example: shift +0.2 -0.1
        """
        try:
            parts = arg.split()
            if len(parts) != 2:
                print("Usage: shift <valence_delta> <arousal_delta>")
                return

            v_delta = float(parts[0])
            a_delta = float(parts[1])

            self.state.shift(v_delta, a_delta)

            print(f"✓ Shifted by ({v_delta:+.2f}, {a_delta:+.2f})")
            self._show_state()

        except ValueError:
            print("Error: Invalid numbers")

    def do_event(self, arg):
        """
        Process an emotional event.
        Usage: event <event_type> [intensity]
        Example: event success 1.5

        Available events: success, failure, test_passed, test_failed,
                         insight, novelty, frustration, blocked, error
        """
        parts = arg.split()
        if not parts:
            print("Usage: event <event_type> [intensity]")
            print(f"\nAvailable events: {', '.join(self.regulator.get_available_events())}")
            return

        event_type = parts[0]
        intensity = float(parts[1]) if len(parts) > 1 else 1.0

        self.regulator.process_event(event_type, intensity)

        print(f"✓ Processed '{event_type}' event (intensity={intensity:.1f})")
        self._show_state()

    def do_reset(self, arg):
        """
        Reset emotional state to baseline.
        Usage: reset
        """
        self.state.reset()
        print("✓ State reset to baseline")
        self._show_state()

    def do_decay(self, arg):
        """
        Simulate time passing and homeostatic decay.
        Usage: decay <seconds>
        Example: decay 5.0
        """
        try:
            seconds = float(arg) if arg else 1.0
            if seconds < 0:
                print("Error: Seconds must be positive")
                return

            # Manually trigger decay by setting last_update backwards
            self.state.last_update = time.time() - seconds

            # Process a neutral event to trigger decay calculation
            self.regulator.process_event("idle", intensity=0.0)

            print(f"✓ Simulated {seconds:.1f}s of decay")
            self._show_state()

        except ValueError:
            print("Error: Invalid number")

    def do_baseline(self, arg):
        """
        Set baseline (homeostasis target).
        Usage: baseline <valence> <arousal>
        Example: baseline 0.1 -0.1
        """
        try:
            parts = arg.split()
            if len(parts) != 2:
                print("Usage: baseline <valence> <arousal>")
                return

            valence = float(parts[0])
            arousal = float(parts[1])

            if not (-1.0 <= valence <= 1.0) or not (-1.0 <= arousal <= 1.0):
                print("Error: Values must be in range [-1.0, 1.0]")
                return

            self.state.baseline_valence = valence
            self.state.baseline_arousal = arousal

            print(f"✓ Baseline set to ({valence:.2f}, {arousal:.2f})")
            self._show_state()

        except ValueError:
            print("Error: Invalid numbers")

    def do_strength(self, arg):
        """
        Set steering strength multiplier.
        Usage: strength <value>
        Example: strength 1.5
        """
        try:
            strength = float(arg)
            if strength < 0:
                print("Error: Strength must be non-negative")
                return

            self.state.steering_strength = strength

            print(f"✓ Steering strength set to {strength:.2f}")
            self._show_state()

        except ValueError:
            print("Error: Invalid number")

    def do_info(self, arg):
        """
        Show current state information (same as startup display).
        Usage: info
        """
        self._show_state()

    def do_dict(self, arg):
        """
        Show state as dictionary (API format).
        Usage: dict
        """
        import json
        print("\nState Dictionary:")
        print(json.dumps(self.state.to_dict(), indent=2))

    def do_quadrants(self, arg):
        """
        Show examples of each quadrant.
        Usage: quadrants
        """
        print("\n=== Emotional Quadrants ===\n")

        examples = [
            ("Excited", 0.8, 0.8, "High valence, high arousal"),
            ("Frustrated", -0.8, 0.8, "Low valence, high arousal"),
            ("Calm", 0.8, -0.8, "High valence, low arousal"),
            ("Depleted", -0.8, -0.8, "Low valence, low arousal"),
        ]

        for name, v, a, desc in examples:
            state = EmotionalState(valence=v, arousal=a)
            coeffs = state.get_steering_coefficients()

            print(f"{name:12s} ({v:+.1f}, {a:+.1f}): {desc}")
            print(f"  Coefficients: E:{coeffs['excited']:.2f} "
                  f"F:{coeffs['frustrated']:.2f} "
                  f"C:{coeffs['calm']:.2f} "
                  f"D:{coeffs['depleted']:.2f}\n")

    def do_events(self, arg):
        """
        List available emotional events.
        Usage: events
        """
        print("\nAvailable Events:")
        events = self.regulator.get_available_events()
        for event in sorted(events):
            print(f"  - {event}")
        print("\nUsage: event <event_type> [intensity]")

    def do_quit(self, arg):
        """Exit the REPL."""
        print("\nGoodbye!")
        return True

    def do_exit(self, arg):
        """Exit the REPL (alias for quit)."""
        return self.do_quit(arg)

    def do_EOF(self, arg):
        """Exit on Ctrl+D."""
        print()  # Newline after ^D
        return self.do_quit(arg)

    def default(self, line):
        """Handle unknown commands."""
        print(f"Unknown command: {line}")
        print("Type 'help' for available commands")

    def emptyline(self):
        """Don't repeat last command on empty line."""
        pass


def main():
    """Main entry point."""
    try:
        EmotionREPL().cmdloop()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")


if __name__ == "__main__":
    main()
