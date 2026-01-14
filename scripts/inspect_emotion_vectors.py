#!/usr/bin/env python3
"""Inspect and validate trained emotion steering vectors.

This script helps verify that steering vectors are properly trained
and have expected properties.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available. Install with: pip install torch")


def load_vectors(vector_dir: Path) -> Dict[str, torch.Tensor]:
    """
    Load all emotion steering vectors from a directory.

    Args:
        vector_dir: Directory containing .pt files

    Returns:
        Dictionary mapping emotion names to tensors
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to load vectors")

    vectors = {}
    expected = ["excited", "frustrated", "calm", "depleted"]

    for emotion in expected:
        vector_path = vector_dir / f"{emotion}.pt"
        if not vector_path.exists():
            print(f"⚠️  Missing vector: {vector_path}")
            continue

        vector = torch.load(vector_path, map_location="cpu")
        vectors[emotion] = vector
        print(f"✅ Loaded {emotion:12s}: shape={vector.shape}, dtype={vector.dtype}")

    return vectors


def analyze_vector(name: str, vector: torch.Tensor) -> None:
    """
    Analyze properties of a single steering vector.

    Args:
        name: Emotion name
        vector: Steering vector tensor
    """
    print(f"\n=== {name.upper()} ===")
    print(f"Shape:       {vector.shape}")
    print(f"Dtype:       {vector.dtype}")
    print(f"Device:      {vector.device}")
    print(f"Norm (L2):   {vector.norm().item():.4f}")
    print(f"Mean:        {vector.mean().item():.6f}")
    print(f"Std:         {vector.std().item():.6f}")
    print(f"Min:         {vector.min().item():.6f}")
    print(f"Max:         {vector.max().item():.6f}")
    print(f"Non-zero:    {(vector != 0).sum().item()} / {vector.numel()}")


def compare_vectors(vectors: Dict[str, torch.Tensor]) -> None:
    """
    Compare vectors to check for orthogonality and distinguishability.

    Args:
        vectors: Dictionary of emotion vectors
    """
    print("\n=== Vector Comparisons ===\n")

    emotions = list(vectors.keys())

    # Compute pairwise cosine similarities
    print("Cosine Similarities (closer to 0 = more orthogonal):")
    print("-" * 60)

    similarities = {}
    for i, emo1 in enumerate(emotions):
        for emo2 in emotions[i+1:]:
            v1 = vectors[emo1].flatten()
            v2 = vectors[emo2].flatten()

            # Normalize and compute cosine similarity
            v1_norm = v1 / (v1.norm() + 1e-8)
            v2_norm = v2 / (v2.norm() + 1e-8)
            similarity = (v1_norm * v2_norm).sum().item()

            similarities[(emo1, emo2)] = similarity
            print(f"  {emo1:12s} ↔ {emo2:12s}: {similarity:+.4f}")

    # Check for opposite emotions having negative correlation
    print("\n--- Expected Relationships ---")
    opposites = [
        ("excited", "depleted"),
        ("frustrated", "calm"),
    ]

    for emo1, emo2 in opposites:
        if emo1 in vectors and emo2 in vectors:
            sim = similarities.get((emo1, emo2)) or similarities.get((emo2, emo1))
            if sim is not None:
                status = "✅" if sim < 0 else "⚠️"
                print(f"  {status} {emo1} ↔ {emo2}: {sim:+.4f} "
                      f"(expected negative for opposites)")


def check_vector_quality(vectors: Dict[str, torch.Tensor]) -> None:
    """
    Run quality checks on steering vectors.

    Args:
        vectors: Dictionary of emotion vectors
    """
    print("\n=== Quality Checks ===\n")

    issues = []

    # Check 1: All vectors should have same shape
    shapes = set(v.shape for v in vectors.values())
    if len(shapes) > 1:
        issues.append(f"❌ Inconsistent shapes: {shapes}")
    else:
        print(f"✅ All vectors have consistent shape: {list(shapes)[0]}")

    # Check 2: Vectors should have reasonable norms
    norms = {name: v.norm().item() for name, v in vectors.items()}
    for name, norm in norms.items():
        if norm < 0.01:
            issues.append(f"⚠️  {name} has very small norm: {norm:.6f}")
        elif norm > 100:
            issues.append(f"⚠️  {name} has very large norm: {norm:.6f}")

    if not issues:
        print(f"✅ All norms in reasonable range")
        for name, norm in norms.items():
            print(f"   {name:12s}: {norm:.4f}")

    # Check 3: Vectors should not be all zeros
    for name, vector in vectors.items():
        if (vector == 0).all():
            issues.append(f"❌ {name} is all zeros!")

    if not any("all zeros" in issue for issue in issues):
        print(f"✅ No zero vectors found")

    # Check 4: Vectors should have some variation
    for name, vector in vectors.items():
        std = vector.std().item()
        if std < 1e-6:
            issues.append(f"⚠️  {name} has very low variance (std={std:.6f})")

    if not any("variance" in issue for issue in issues):
        print(f"✅ All vectors have sufficient variance")

    # Print all issues
    if issues:
        print("\n--- Issues Detected ---")
        for issue in issues:
            print(issue)
    else:
        print("\n✅ All quality checks passed!")


def simulate_blending(vectors: Dict[str, torch.Tensor]) -> None:
    """
    Simulate blending vectors with different coefficients.

    Args:
        vectors: Dictionary of emotion vectors
    """
    if not all(emo in vectors for emo in ["excited", "frustrated", "calm", "depleted"]):
        print("\n⚠️  Cannot simulate blending: missing vectors")
        return

    print("\n=== Blending Simulation ===\n")

    # Test cases: (valence, arousal, description)
    test_cases = [
        (1.0, 1.0, "Pure Excited"),
        (0.5, 0.5, "Moderately Excited"),
        (0.0, 0.0, "Neutral (all equal)"),
        (-1.0, 1.0, "Pure Frustrated"),
        (0.8, -0.6, "Mostly Calm"),
    ]

    for valence, arousal, description in test_cases:
        # Compute coefficients (same as EmotionalState.get_steering_coefficients)
        v = (valence + 1.0) / 2.0
        a = (arousal + 1.0) / 2.0

        coeffs = {
            "excited": v * a,
            "frustrated": (1.0 - v) * a,
            "calm": v * (1.0 - a),
            "depleted": (1.0 - v) * (1.0 - a),
        }

        # Blend vectors
        blended = (
            coeffs["excited"] * vectors["excited"] +
            coeffs["frustrated"] * vectors["frustrated"] +
            coeffs["calm"] * vectors["calm"] +
            coeffs["depleted"] * vectors["depleted"]
        )

        print(f"{description:20s} | ({valence:+.1f}, {arousal:+.1f})")
        print(f"  Coefficients: E:{coeffs['excited']:.2f} F:{coeffs['frustrated']:.2f} "
              f"C:{coeffs['calm']:.2f} D:{coeffs['depleted']:.2f}")
        print(f"  Blended norm: {blended.norm().item():.4f}")
        print()


def export_metadata(vectors: Dict[str, torch.Tensor], output_path: Path) -> None:
    """
    Export vector metadata to JSON for documentation.

    Args:
        vectors: Dictionary of emotion vectors
        output_path: Path to save JSON metadata
    """
    import json

    metadata = {}
    for name, vector in vectors.items():
        metadata[name] = {
            "shape": list(vector.shape),
            "dtype": str(vector.dtype),
            "norm": float(vector.norm().item()),
            "mean": float(vector.mean().item()),
            "std": float(vector.std().item()),
            "min": float(vector.min().item()),
            "max": float(vector.max().item()),
            "non_zero": int((vector != 0).sum().item()),
            "total_elements": int(vector.numel()),
        }

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Metadata exported to {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "vector_dir",
        type=Path,
        help="Directory containing emotion vector .pt files",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Show detailed analysis of each vector",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare vectors for orthogonality",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Run quality checks",
    )
    parser.add_argument(
        "--blend",
        action="store_true",
        help="Simulate coefficient blending",
    )
    parser.add_argument(
        "--export",
        type=Path,
        metavar="JSON_PATH",
        help="Export metadata to JSON file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all analyses",
    )

    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    if not args.vector_dir.exists():
        print(f"Error: Directory not found: {args.vector_dir}")
        sys.exit(1)

    # Load vectors
    print(f"Loading vectors from {args.vector_dir}...\n")
    vectors = load_vectors(args.vector_dir)

    if not vectors:
        print("\n❌ No vectors loaded. Exiting.")
        sys.exit(1)

    print(f"\n✅ Loaded {len(vectors)}/4 expected vectors")

    # Run requested analyses
    if args.all:
        args.analyze = args.compare = args.quality = args.blend = True

    if args.analyze:
        for name, vector in vectors.items():
            analyze_vector(name, vector)

    if args.compare:
        if len(vectors) >= 2:
            compare_vectors(vectors)
        else:
            print("\n⚠️  Need at least 2 vectors for comparison")

    if args.quality:
        check_vector_quality(vectors)

    if args.blend:
        simulate_blending(vectors)

    if args.export:
        export_metadata(vectors, args.export)

    # If no specific analysis requested, show summary
    if not any([args.analyze, args.compare, args.quality, args.blend, args.export, args.all]):
        print("\nTo run analyses, use:")
        print("  --all         Run all analyses")
        print("  --analyze     Detailed per-vector statistics")
        print("  --compare     Vector similarity comparison")
        print("  --quality     Quality checks")
        print("  --blend       Blending simulation")


if __name__ == "__main__":
    main()
