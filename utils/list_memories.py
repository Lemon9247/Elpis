#!/usr/bin/env python3
"""List and count memories in Psyche's memory database."""

import argparse
import sys
from typing import Optional

from memory_db import (
    get_client,
    get_collections,
    format_memory,
    format_datetime,
    truncate,
)


def count_memories(verbose: bool = False) -> None:
    """Print memory counts by collection."""
    client = get_client()
    collections = get_collections(client)

    short_count = collections["short_term"].count()
    long_count = collections["long_term"].count()

    print(f"Short-term memories: {short_count}")
    print(f"Long-term memories:  {long_count}")
    print(f"Total:               {short_count + long_count}")


def list_memories(
    collection_name: Optional[str] = None,
    limit: int = 20,
    show_content: bool = False,
    memory_type: Optional[str] = None,
) -> None:
    """List memories from the database."""
    client = get_client()
    collections = get_collections(client)

    # Determine which collections to query
    if collection_name == "short":
        cols_to_query = [("short_term", collections["short_term"])]
    elif collection_name == "long":
        cols_to_query = [("long_term", collections["long_term"])]
    else:
        cols_to_query = [
            ("short_term", collections["short_term"]),
            ("long_term", collections["long_term"]),
        ]

    total_shown = 0

    for name, col in cols_to_query:
        count = col.count()
        if count == 0:
            continue

        result = col.get(
            limit=min(limit - total_shown, count),
            include=["documents", "metadatas"],
        )

        if not result["ids"]:
            continue

        print(f"\n=== {name.upper().replace('_', ' ')} ({count} total) ===\n")

        for i, memory_id in enumerate(result["ids"]):
            metadata = result["metadatas"][i]
            document = result["documents"][i]

            # Filter by memory type if specified
            if memory_type and metadata.get("memory_type") != memory_type:
                continue

            # Format output
            created = format_datetime(metadata.get("created_at", ""))
            mem_type = metadata.get("memory_type", "unknown")
            importance = metadata.get("importance_score", 0.5)

            print(f"ID: {memory_id[:8]}...")
            print(f"  Type: {mem_type} | Importance: {importance:.2f} | Created: {created}")

            if show_content:
                # Show full content
                print(f"  Content: {document}")
            else:
                # Show truncated content
                print(f"  Content: {truncate(document, 70)}")

            # Show emotional context if present
            ec_raw = metadata.get("emotional_context")
            if ec_raw and ec_raw != "null":
                import json
                ec = json.loads(ec_raw)
                if ec:
                    print(f"  Emotion: {ec['quadrant']} (v={ec['valence']:.2f}, a={ec['arousal']:.2f})")

            print()
            total_shown += 1

            if total_shown >= limit:
                break

        if total_shown >= limit:
            break

    if total_shown == 0:
        print("No memories found.")


def main():
    parser = argparse.ArgumentParser(
        description="List memories from Psyche's memory database"
    )
    parser.add_argument(
        "--count", "-c",
        action="store_true",
        help="Only show memory counts"
    )
    parser.add_argument(
        "--collection",
        choices=["short", "long", "all"],
        default="all",
        help="Which collection to query (default: all)"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=20,
        help="Maximum number of memories to show (default: 20)"
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Show full content instead of truncated"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["episodic", "semantic", "procedural", "emotional"],
        help="Filter by memory type"
    )

    args = parser.parse_args()

    if args.count:
        count_memories()
    else:
        collection = None if args.collection == "all" else args.collection
        list_memories(
            collection_name=collection,
            limit=args.limit,
            show_content=args.full,
            memory_type=args.type,
        )


if __name__ == "__main__":
    main()
