#!/usr/bin/env python3
"""Base module for connecting to Psyche's memory database."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import chromadb
from chromadb.config import Settings


# Default path relative to project root
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "memory"


def get_client(db_path: Optional[Path] = None) -> chromadb.PersistentClient:
    """Get a ChromaDB client connected to the memory database."""
    path = db_path or DEFAULT_DB_PATH
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=False,
        ),
    )


def get_collections(client: chromadb.PersistentClient) -> Dict[str, Any]:
    """Get the short-term and long-term memory collections."""
    return {
        "short_term": client.get_collection("short_term_memory"),
        "long_term": client.get_collection("long_term_memory"),
    }


def parse_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Parse JSON-encoded fields in metadata."""
    parsed = metadata.copy()

    # Parse JSON fields
    if "tags" in parsed:
        parsed["tags"] = json.loads(parsed["tags"]) if parsed["tags"] else []
    if "metadata_json" in parsed:
        parsed["extra_metadata"] = json.loads(parsed["metadata_json"]) if parsed["metadata_json"] else {}
        del parsed["metadata_json"]
    if "emotional_context" in parsed:
        parsed["emotional_context"] = json.loads(parsed["emotional_context"]) if parsed["emotional_context"] else None

    return parsed


def format_memory(
    memory_id: str,
    document: str,
    metadata: Dict[str, Any],
    distance: Optional[float] = None,
) -> Dict[str, Any]:
    """Format a memory record for display."""
    parsed = parse_metadata(metadata)

    result = {
        "id": memory_id,
        "content": document,
        **parsed,
    }

    if distance is not None:
        result["relevance_distance"] = distance

    return result


def format_datetime(iso_string: str) -> str:
    """Format ISO datetime string for display."""
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return iso_string or "unknown"


def truncate(text: str, max_length: int = 80) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
