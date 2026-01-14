#!/usr/bin/env python3
"""Debug and visualize emotional state mappings.

This script helps developers understand how valence-arousal coordinates
map to steering coefficients and sampling parameters.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from elpis.emotion.state import EmotionalState


def visualize_grid(resolution: int = 5) -> None:
    """
    Print a grid showing how different valence/arousal combinations
    map to steering coefficients and sampling parameters.

    Args:
        resolution: Number of steps per dimension (higher = more detail)
    """
    print("\n=== Emotional State Grid ===\n")
    print("Format: (valence, arousal) -> quadrant | dominant_emotion (strength)")
    print("         temperature, top_p | steering_coefficients\n")

    values = [round(-1.0 + (2.0 * i / (resolution - 1)), 2) for i in range(resolution)]

    for arousal in reversed(values):  # High arousal at top
        print(f"\nArousal = {arousal:+.2f}")
        print("-" * 80)

        for valence in values:
            state = EmotionalState(valence=valence, arousal=arousal)
            params = state.get_modulated_params()
            coeffs = state.get_steering_coefficients()
            dominant, strength = state.get_dominant_emotion()

            print(f"  ({valence:+.2f}, {arousal:+.2f}) -> {state.get_quadrant():10s} | "
                  f"{dominant:10s} ({strength:.2f})")
            print(f"    temp={params['temperature']:.2f}, top_p={params['top_p']:.2f} | "
                  f"E:{coeffs['excited']:.2f} F:{coeffs['frustrated']:.2f} "
                  f"C:{coeffs['calm']:.2f} D:{coeffs['depleted']:.2f}")


def test_specific_state(valence: float, arousal: float, steering_strength: float = 1.0) -> None:
    """
    Analyze a specific emotional state in detail.

    Args:
        valence: Valence value (-1.0 to +1.0)
        arousal: Arousal value (-1.0 to +1.0)
        steering_strength: Global steering multiplier (default 1.0)
    """
    state = EmotionalState(valence=valence, arousal=arousal, steering_strength=steering_strength)

    print(f"\n=== Emotional State Analysis ===\n")
    print(f"Valence:           {state.valence:+.3f}  (pleasant ← → unpleasant)")
    print(f"Arousal:           {state.arousal:+.3f}  (low energy ← → high energy)")
    print(f"Quadrant:          {state.get_quadrant()}")
    print(f"Steering Strength: {state.steering_strength:.2f}")
    print(f"Distance from baseline: {state.distance_from_baseline():.3f}")

    dominant, strength = state.get_dominant_emotion()
    print(f"\nDominant Emotion:  {dominant} (strength: {strength:.3f})")

    params = state.get_modulated_params()
    print(f"\n--- Sampling Parameters ---")
    print(f"Temperature:       {params['temperature']:.2f}")
    print(f"Top-p:             {params['top_p']:.2f}")

    coeffs = state.get_steering_coefficients()
    print(f"\n--- Steering Coefficients ---")
    for emotion, coeff in coeffs.items():
        bar = "█" * int(coeff * 50)
        print(f"{emotion:12s}: {coeff:.3f}  {bar}")

    total = sum(coeffs.values())
    print(f"\nSum of coefficients: {total:.3f} (should be {steering_strength:.2f})")


def test_transitions() -> None:
    """
    Show how emotional state transitions affect coefficients.
    Demonstrates smooth interpolation between quadrants.
    """
    print("\n=== Transition Paths ===\n")

    # Path 1: Neutral → Excited (increasing valence and arousal)
    print("Path 1: Neutral → Excited")
    print("-" * 60)
    for step in range(6):
        val = aro = step / 5.0  # 0.0 to 1.0
        state = EmotionalState(valence=val, arousal=aro)
        coeffs = state.get_steering_coefficients()
        print(f"  Step {step}: ({val:.1f}, {aro:.1f}) -> "
              f"E:{coeffs['excited']:.2f} F:{coeffs['frustrated']:.2f} "
              f"C:{coeffs['calm']:.2f} D:{coeffs['depleted']:.2f}")

    # Path 2: Excited → Frustrated (decreasing valence)
    print("\nPath 2: Excited → Frustrated")
    print("-" * 60)
    for step in range(6):
        val = 1.0 - (step / 5.0)  # 1.0 to 0.0
        aro = 1.0
        state = EmotionalState(valence=val, arousal=aro)
        coeffs = state.get_steering_coefficients()
        print(f"  Step {step}: ({val:.1f}, {aro:.1f}) -> "
              f"E:{coeffs['excited']:.2f} F:{coeffs['frustrated']:.2f} "
              f"C:{coeffs['calm']:.2f} D:{coeffs['depleted']:.2f}")


def test_steering_strength() -> None:
    """
    Show how steering_strength parameter scales coefficients.
    """
    print("\n=== Steering Strength Scaling ===\n")
    print("State: (valence=0.8, arousal=0.6)")
    print("-" * 60)

    for strength in [0.0, 0.5, 1.0, 1.5, 2.0]:
        state = EmotionalState(valence=0.8, arousal=0.6, steering_strength=strength)
        coeffs = state.get_steering_coefficients()
        total = sum(coeffs.values())
        print(f"Strength {strength:.1f}: E:{coeffs['excited']:.2f} "
              f"F:{coeffs['frustrated']:.2f} C:{coeffs['calm']:.2f} "
              f"D:{coeffs['depleted']:.2f} | Sum: {total:.2f}")


def validate_coefficients() -> None:
    """
    Run validation tests on coefficient calculations.
    """
    print("\n=== Coefficient Validation ===\n")

    import random

    errors = []
    for _ in range(100):
        v = random.uniform(-1.0, 1.0)
        a = random.uniform(-1.0, 1.0)
        strength = random.uniform(0.0, 2.0)

        state = EmotionalState(valence=v, arousal=a, steering_strength=strength)
        coeffs = state.get_steering_coefficients()
        total = sum(coeffs.values())

        expected = strength
        if abs(total - expected) > 0.001:
            errors.append((v, a, strength, total))

    if errors:
        print(f"❌ Found {len(errors)} errors:\n")
        for v, a, strength, total in errors[:5]:
            print(f"  ({v:.2f}, {a:.2f}) with strength {strength:.2f} -> sum={total:.3f}")
    else:
        print("✅ All 100 random states have correct coefficient sums")

    # Test bounds
    print("\n--- Bounds Testing ---")
    extremes = [
        (-1.0, -1.0, "depleted"),
        (1.0, 1.0, "excited"),
        (-1.0, 1.0, "frustrated"),
        (1.0, -1.0, "calm"),
        (0.0, 0.0, "neutral"),
    ]

    for v, a, expected_quadrant in extremes:
        state = EmotionalState(valence=v, arousal=a)
        coeffs = state.get_steering_coefficients()
        dominant, strength = state.get_dominant_emotion()

        print(f"  ({v:+.1f}, {a:+.1f}): {dominant:10s} ({strength:.2f}) - "
              f"Coeffs sum to {sum(coeffs.values()):.2f}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Show full valence-arousal grid",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=5,
        help="Grid resolution (default: 5)",
    )
    parser.add_argument(
        "--state",
        nargs=2,
        type=float,
        metavar=("VALENCE", "AROUSAL"),
        help="Analyze specific state (e.g., --state 0.5 -0.3)",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Steering strength for --state (default: 1.0)",
    )
    parser.add_argument(
        "--transitions",
        action="store_true",
        help="Show transition paths between quadrants",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Test steering strength scaling",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation tests",
    )

    args = parser.parse_args()

    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nQuick examples:")
        print("  python debug_emotion_state.py --state 0.8 0.5")
        print("  python debug_emotion_state.py --grid")
        print("  python debug_emotion_state.py --transitions")
        print("  python debug_emotion_state.py --validate")
        return

    if args.grid:
        visualize_grid(args.resolution)

    if args.state:
        valence, arousal = args.state
        test_specific_state(valence, arousal, args.strength)

    if args.transitions:
        test_transitions()

    if args.scaling:
        test_steering_strength()

    if args.validate:
        validate_coefficients()


if __name__ == "__main__":
    main()
