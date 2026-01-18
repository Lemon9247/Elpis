#!/usr/bin/env python3
"""Get a specific memory by ID from Psyche's database."""

import argparse
import json
import sys

from memory_db import get_client, get_collections, parse_metadata


def get_memory(memory_id: str) -> None:
    """Retrieve and display a specific memory by ID."""
    client = get_client()
    collections = get_collections(client)

    # Try to find the memory in both collections
    for name, col in collections.items():
        try:
            result = col.get(
                ids=[memory_id],
                include=["documents", "metadatas", "embeddings"],
            )

            if result["ids"]:
                print(f"Found in: {name}\n")
                print("=" * 60)

                metadata = parse_metadata(result["metadatas"][0])

                print(f"ID: {result['ids'][0]}")
                print(f"Type: {metadata.get('memory_type', 'unknown')}")
                print(f"Status: {metadata.get('status', 'unknown')}")
                print(f"Created: {metadata.get('created_at', 'unknown')}")
                print(f"Importance: {metadata.get('importance_score', 0.5):.3f}")

                ec = metadata.get("emotional_context")
                if ec:
                    print(f"\nEmotional Context:")
                    print(f"  Quadrant: {ec['quadrant']}")
                    print(f"  Valence: {ec['valence']:+.3f}")
                    print(f"  Arousal: {ec['arousal']:+.3f}")

                tags = metadata.get("tags", [])
                if tags:
                    print(f"\nTags: {', '.join(tags)}")

                print(f"\nContent:\n{result['documents'][0]}")

                if metadata.get("summary"):
                    print(f"\nSummary:\n{metadata['summary']}")

                # Show embedding dimensions if requested
                if result["embeddings"] and result["embeddings"][0]:
                    embedding = result["embeddings"][0]
                    print(f"\nEmbedding: {len(embedding)} dimensions")
                    print(f"  First 5: {embedding[:5]}")

                return
        except Exception as e:
            continue

    print(f"Memory with ID '{memory_id}' not found.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Get a specific memory by ID"
    )
    parser.add_argument(
        "memory_id",
        help="Memory ID (full or partial)"
    )

    args = parser.parse_args()
    get_memory(args.memory_id)


if __name__ == "__main__":
    main()
