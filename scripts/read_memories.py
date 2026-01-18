#!/usr/bin/env python3
"""Simple interface for reading Psyche's memories."""

import argparse
import json
import sys
from typing import Optional, List, Dict, Any

from memory_db import get_client, get_collections, parse_metadata, format_datetime


def get_all_memories(collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all memories, sorted by creation date."""
    client = get_client()
    collections = get_collections(client)

    if collection_name == "short":
        cols = [("short_term", collections["short_term"])]
    elif collection_name == "long":
        cols = [("long_term", collections["long_term"])]
    else:
        cols = [
            ("short_term", collections["short_term"]),
            ("long_term", collections["long_term"]),
        ]

    memories = []
    for name, col in cols:
        count = col.count()
        if count == 0:
            continue

        result = col.get(limit=count, include=["documents", "metadatas"])
        for i in range(len(result["ids"])):
            memories.append({
                "id": result["ids"][i],
                "content": result["documents"][i],
                "collection": name,
                **parse_metadata(result["metadatas"][i]),
            })

    # Sort by created_at
    memories.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return memories


def print_memory(memory: Dict[str, Any], index: int, total: int) -> None:
    """Print a single memory in full."""
    print("\n" + "=" * 70)
    print(f"MEMORY {index + 1} of {total}")
    print("=" * 70)

    print(f"ID:         {memory['id']}")
    print(f"Collection: {memory['collection']}")
    print(f"Type:       {memory.get('memory_type', 'unknown')}")
    print(f"Created:    {format_datetime(memory.get('created_at', ''))}")
    print(f"Importance: {memory.get('importance_score', 0.5):.2f}")

    ec = memory.get("emotional_context")
    if ec:
        print(f"Emotion:    {ec['quadrant']} (valence={ec['valence']:+.2f}, arousal={ec['arousal']:+.2f})")

    tags = memory.get("tags", [])
    if tags:
        print(f"Tags:       {', '.join(tags)}")

    print("\n--- CONTENT ---\n")
    print(memory["content"])

    if memory.get("summary"):
        print("\n--- SUMMARY ---\n")
        print(memory["summary"])

    print()


def interactive_reader(memories: List[Dict[str, Any]]) -> None:
    """Interactive mode for browsing memories."""
    if not memories:
        print("No memories found.")
        return

    total = len(memories)
    current = 0

    print(f"\nLoaded {total} memories. Commands: [n]ext, [p]rev, [q]uit, [g]o to #, [s]earch")

    while True:
        print_memory(memories[current], current, total)

        try:
            cmd = input("[n/p/q/g/s] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if cmd in ("n", "next", ""):
            current = (current + 1) % total
        elif cmd in ("p", "prev"):
            current = (current - 1) % total
        elif cmd in ("q", "quit", "exit"):
            break
        elif cmd.startswith("g") or cmd.isdigit():
            try:
                num = int(cmd[1:].strip() if cmd.startswith("g") else cmd)
                if 1 <= num <= total:
                    current = num - 1
                else:
                    print(f"Enter a number between 1 and {total}")
            except ValueError:
                print("Invalid number")
        elif cmd.startswith("s"):
            query = cmd[1:].strip() if len(cmd) > 1 else input("Search: ").strip()
            if query:
                matches = [
                    (i, m) for i, m in enumerate(memories)
                    if query.lower() in m["content"].lower()
                ]
                if matches:
                    print(f"\nFound {len(matches)} matches:")
                    for i, (idx, m) in enumerate(matches[:10]):
                        preview = m["content"][:60].replace("\n", " ")
                        print(f"  {idx + 1}: {preview}...")
                    print()
                else:
                    print("No matches found.")


def dump_all(memories: List[Dict[str, Any]]) -> None:
    """Dump all memories to stdout."""
    for i, memory in enumerate(memories):
        print_memory(memory, i, len(memories))


def main():
    parser = argparse.ArgumentParser(
        description="Read Psyche's memories"
    )
    parser.add_argument(
        "--collection", "-c",
        choices=["short", "long", "all"],
        default="all",
        help="Which collection to read (default: all)"
    )
    parser.add_argument(
        "--dump", "-d",
        action="store_true",
        help="Dump all memories (non-interactive)"
    )
    parser.add_argument(
        "--latest", "-l",
        type=int,
        metavar="N",
        help="Show only the N most recent memories"
    )
    parser.add_argument(
        "--search", "-s",
        type=str,
        metavar="QUERY",
        help="Filter to memories containing QUERY"
    )

    args = parser.parse_args()

    collection = None if args.collection == "all" else args.collection
    memories = get_all_memories(collection)

    # Apply search filter
    if args.search:
        query = args.search.lower()
        memories = [m for m in memories if query in m["content"].lower()]
        print(f"Found {len(memories)} memories matching '{args.search}'")

    # Apply latest filter
    if args.latest:
        memories = memories[:args.latest]

    if args.dump:
        dump_all(memories)
    else:
        interactive_reader(memories)


if __name__ == "__main__":
    main()
