#!/usr/bin/env python3
"""Export memories from Psyche's database to JSON or CSV."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from memory_db import get_client, get_collections, parse_metadata


def get_all_memories(collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all memories from the database."""
    client = get_client()
    collections = get_collections(client)

    if collection_name == "short":
        cols_to_query = [("short_term", collections["short_term"])]
    elif collection_name == "long":
        cols_to_query = [("long_term", collections["long_term"])]
    else:
        cols_to_query = [
            ("short_term", collections["short_term"]),
            ("long_term", collections["long_term"]),
        ]

    all_memories = []

    for name, col in cols_to_query:
        count = col.count()
        if count == 0:
            continue

        result = col.get(
            limit=count,
            include=["documents", "metadatas"],
        )

        for i in range(len(result["ids"])):
            memory = {
                "id": result["ids"][i],
                "content": result["documents"][i],
                "collection": name,
                **parse_metadata(result["metadatas"][i]),
            }
            all_memories.append(memory)

    return all_memories


def export_json(memories: List[Dict[str, Any]], output_path: Path) -> None:
    """Export memories to JSON format."""
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "count": len(memories),
        "memories": memories,
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    print(f"Exported {len(memories)} memories to {output_path}")


def export_csv(memories: List[Dict[str, Any]], output_path: Path) -> None:
    """Export memories to CSV format."""
    if not memories:
        print("No memories to export.")
        return

    # Flatten emotional context for CSV
    rows = []
    for m in memories:
        row = {
            "id": m["id"],
            "content": m["content"],
            "collection": m["collection"],
            "memory_type": m.get("memory_type", ""),
            "status": m.get("status", ""),
            "importance_score": m.get("importance_score", ""),
            "created_at": m.get("created_at", ""),
            "summary": m.get("summary", ""),
            "tags": ",".join(m.get("tags", [])),
        }

        # Flatten emotional context
        ec = m.get("emotional_context")
        if ec:
            row["valence"] = ec.get("valence", "")
            row["arousal"] = ec.get("arousal", "")
            row["quadrant"] = ec.get("quadrant", "")
        else:
            row["valence"] = ""
            row["arousal"] = ""
            row["quadrant"] = ""

        rows.append(row)

    fieldnames = [
        "id", "content", "collection", "memory_type", "status",
        "importance_score", "created_at", "summary", "tags",
        "valence", "arousal", "quadrant",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(memories)} memories to {output_path}")


def export_markdown(memories: List[Dict[str, Any]], output_path: Path) -> None:
    """Export memories to Markdown format for human reading."""
    with open(output_path, "w") as f:
        f.write(f"# Psyche Memory Export\n\n")
        f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Total memories: {len(memories)}\n\n")
        f.write("---\n\n")

        for m in memories:
            f.write(f"## Memory {m['id'][:8]}...\n\n")
            f.write(f"**Collection:** {m['collection']}  \n")
            f.write(f"**Type:** {m.get('memory_type', 'unknown')}  \n")
            f.write(f"**Created:** {m.get('created_at', 'unknown')}  \n")
            f.write(f"**Importance:** {m.get('importance_score', 0.5):.2f}  \n")

            ec = m.get("emotional_context")
            if ec:
                f.write(f"**Emotion:** {ec['quadrant']} (v={ec['valence']:.2f}, a={ec['arousal']:.2f})  \n")

            tags = m.get("tags", [])
            if tags:
                f.write(f"**Tags:** {', '.join(tags)}  \n")

            f.write(f"\n### Content\n\n")
            f.write(f"{m['content']}\n\n")

            if m.get("summary"):
                f.write(f"### Summary\n\n{m['summary']}\n\n")

            f.write("---\n\n")

    print(f"Exported {len(memories)} memories to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Psyche's memories to various formats"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "csv", "markdown"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path (default: memories_TIMESTAMP.{format})"
    )
    parser.add_argument(
        "--collection",
        choices=["short", "long", "all"],
        default="all",
        help="Which collection to export (default: all)"
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "md" if args.format == "markdown" else args.format
        output_path = Path(f"memories_{timestamp}.{ext}")

    collection = None if args.collection == "all" else args.collection
    memories = get_all_memories(collection)

    if not memories:
        print("No memories found to export.")
        return

    if args.format == "json":
        export_json(memories, output_path)
    elif args.format == "csv":
        export_csv(memories, output_path)
    elif args.format == "markdown":
        export_markdown(memories, output_path)


if __name__ == "__main__":
    main()
