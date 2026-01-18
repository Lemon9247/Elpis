#!/usr/bin/env python3
"""Analyze emotional distribution of memories in Psyche's database."""

import argparse
import json
from collections import Counter
from typing import Dict, List, Any

from memory_db import get_client, get_collections


def get_all_memories_with_emotions() -> List[Dict[str, Any]]:
    """Get all memories that have emotional context."""
    client = get_client()
    collections = get_collections(client)

    memories_with_emotions = []

    for name, col in collections.items():
        count = col.count()
        if count == 0:
            continue

        result = col.get(
            limit=count,
            include=["metadatas"],
        )

        for i, metadata in enumerate(result["metadatas"]):
            ec_raw = metadata.get("emotional_context")
            if ec_raw and ec_raw != "null":
                ec = json.loads(ec_raw)
                if ec:
                    memories_with_emotions.append({
                        "id": result["ids"][i],
                        "collection": name,
                        "memory_type": metadata.get("memory_type", "unknown"),
                        "valence": ec["valence"],
                        "arousal": ec["arousal"],
                        "quadrant": ec["quadrant"],
                    })

    return memories_with_emotions


def analyze_quadrants(memories: List[Dict[str, Any]]) -> None:
    """Analyze distribution across emotional quadrants."""
    quadrant_counts = Counter(m["quadrant"] for m in memories)
    total = len(memories)

    print("\n=== EMOTIONAL QUADRANT DISTRIBUTION ===\n")

    # Define quadrant descriptions
    quadrant_info = {
        "excited": ("High valence, high arousal", "+", "+"),
        "frustrated": ("Low valence, high arousal", "-", "+"),
        "calm": ("High valence, low arousal", "+", "-"),
        "depleted": ("Low valence, low arousal", "-", "-"),
    }

    for quadrant, (desc, v_sign, a_sign) in quadrant_info.items():
        count = quadrant_counts.get(quadrant, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"{quadrant.capitalize():12} ({v_sign}V, {a_sign}A): {count:4} ({pct:5.1f}%) {bar}")

    print(f"\nTotal memories with emotions: {total}")


def analyze_valence_arousal(memories: List[Dict[str, Any]]) -> None:
    """Analyze valence and arousal statistics."""
    if not memories:
        print("No memories with emotional context found.")
        return

    valences = [m["valence"] for m in memories]
    arousals = [m["arousal"] for m in memories]

    avg_valence = sum(valences) / len(valences)
    avg_arousal = sum(arousals) / len(arousals)

    min_v, max_v = min(valences), max(valences)
    min_a, max_a = min(arousals), max(arousals)

    print("\n=== VALENCE-AROUSAL STATISTICS ===\n")
    print(f"Valence (emotional tone):")
    print(f"  Average: {avg_valence:+.3f}")
    print(f"  Range:   [{min_v:+.2f}, {max_v:+.2f}]")
    print(f"  Interpretation: {'positive' if avg_valence > 0 else 'negative'} overall tone")

    print(f"\nArousal (emotional intensity):")
    print(f"  Average: {avg_arousal:+.3f}")
    print(f"  Range:   [{min_a:+.2f}, {max_a:+.2f}]")
    print(f"  Interpretation: {'high' if avg_arousal > 0 else 'low'} energy state")


def analyze_by_type(memories: List[Dict[str, Any]]) -> None:
    """Analyze emotions by memory type."""
    print("\n=== EMOTIONS BY MEMORY TYPE ===\n")

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for m in memories:
        mem_type = m["memory_type"]
        if mem_type not in by_type:
            by_type[mem_type] = []
        by_type[mem_type].append(m)

    for mem_type, type_memories in sorted(by_type.items()):
        avg_v = sum(m["valence"] for m in type_memories) / len(type_memories)
        avg_a = sum(m["arousal"] for m in type_memories) / len(type_memories)
        dominant_quadrant = Counter(m["quadrant"] for m in type_memories).most_common(1)[0][0]

        print(f"{mem_type.capitalize():12}: {len(type_memories):3} memories")
        print(f"              Avg valence: {avg_v:+.2f}, Avg arousal: {avg_a:+.2f}")
        print(f"              Dominant quadrant: {dominant_quadrant}")
        print()


def visualize_scatter(memories: List[Dict[str, Any]]) -> None:
    """Print a simple ASCII scatter plot of valence vs arousal."""
    if not memories:
        return

    print("\n=== VALENCE-AROUSAL SCATTER (ASCII) ===\n")

    # Create a 21x21 grid (-1 to 1 mapped to 0 to 20)
    grid_size = 21
    grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]

    def to_grid(val):
        return int((val + 1) * 10)  # -1 -> 0, 0 -> 10, 1 -> 20

    # Plot memories
    for m in memories:
        x = min(20, max(0, to_grid(m["valence"])))
        y = min(20, max(0, 20 - to_grid(m["arousal"])))  # Invert y for display

        # Use different characters for quadrants
        char = {
            "excited": "+",
            "frustrated": "x",
            "calm": "o",
            "depleted": "-",
        }.get(m["quadrant"], ".")

        grid[y][x] = char

    # Draw axes
    for i in range(grid_size):
        grid[10][i] = "-"  # horizontal axis
        grid[i][10] = "|"  # vertical axis
    grid[10][10] = "+"  # origin

    # Print grid
    print("  Arousal (+1)")
    print("      ^")
    for row in grid:
        print("      " + "".join(row))
    print("  (-1)<---+--->(+1) Valence")
    print("      v")
    print("  Arousal (-1)")
    print()
    print("Legend: + excited, x frustrated, o calm, - depleted")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze emotional distribution of Psyche's memories"
    )
    parser.add_argument(
        "--scatter", "-s",
        action="store_true",
        help="Show ASCII scatter plot"
    )
    parser.add_argument(
        "--by-type", "-t",
        action="store_true",
        help="Show breakdown by memory type"
    )

    args = parser.parse_args()

    memories = get_all_memories_with_emotions()

    if not memories:
        print("No memories with emotional context found in the database.")
        return

    analyze_quadrants(memories)
    analyze_valence_arousal(memories)

    if args.by_type:
        analyze_by_type(memories)

    if args.scatter:
        visualize_scatter(memories)


if __name__ == "__main__":
    main()
