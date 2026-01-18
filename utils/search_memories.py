#!/usr/bin/env python3
"""Semantic search for memories in Psyche's memory database."""

import argparse
import json
import sys
from typing import Optional

from sentence_transformers import SentenceTransformer

from memory_db import (
    get_client,
    get_collections,
    format_datetime,
    truncate,
)


def search(
    query: str,
    n_results: int = 10,
    collection_name: Optional[str] = None,
    show_content: bool = False,
) -> None:
    """Search memories semantically."""
    # Load embedding model
    print("Loading embedding model...", file=sys.stderr)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query).tolist()

    client = get_client()
    collections = get_collections(client)

    # Determine which collections to search
    if collection_name == "short":
        cols_to_query = [("short_term", collections["short_term"])]
    elif collection_name == "long":
        cols_to_query = [("long_term", collections["long_term"])]
    else:
        cols_to_query = [
            ("short_term", collections["short_term"]),
            ("long_term", collections["long_term"]),
        ]

    # Collect results with distances
    all_results = []

    for name, col in cols_to_query:
        count = col.count()
        if count == 0:
            continue

        results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )

        for i in range(len(results["ids"][0])):
            all_results.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "collection": name,
            })

    # Sort by distance (lower = more relevant)
    all_results.sort(key=lambda x: x["distance"])

    if not all_results:
        print("No memories found.")
        return

    print(f"\nSearch results for: \"{query}\"\n")
    print("=" * 60)

    for i, result in enumerate(all_results[:n_results], 1):
        metadata = result["metadata"]
        document = result["document"]
        distance = result["distance"]

        # Compute similarity from distance (L2 distance to similarity)
        similarity = max(0, 1 - (distance / 2))

        created = format_datetime(metadata.get("created_at", ""))
        mem_type = metadata.get("memory_type", "unknown")
        collection = result["collection"].replace("_", " ")

        print(f"\n[{i}] Similarity: {similarity:.1%} | {collection}")
        print(f"    ID: {result['id'][:8]}... | Type: {mem_type} | Created: {created}")

        if show_content:
            print(f"    Content: {document}")
        else:
            print(f"    Content: {truncate(document, 65)}")

        # Show emotional context
        ec_raw = metadata.get("emotional_context")
        if ec_raw and ec_raw != "null":
            ec = json.loads(ec_raw)
            if ec:
                print(f"    Emotion: {ec['quadrant']} (v={ec['valence']:.2f}, a={ec['arousal']:.2f})")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Semantic search for memories in Psyche's database"
    )
    parser.add_argument(
        "query",
        help="Search query"
    )
    parser.add_argument(
        "--results", "-n",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )
    parser.add_argument(
        "--collection",
        choices=["short", "long", "all"],
        default="all",
        help="Which collection to search (default: all)"
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Show full content instead of truncated"
    )

    args = parser.parse_args()

    collection = None if args.collection == "all" else args.collection
    search(
        query=args.query,
        n_results=args.results,
        collection_name=collection,
        show_content=args.full,
    )


if __name__ == "__main__":
    main()
