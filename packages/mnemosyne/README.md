# Mnemosyne

MCP memory server with emotional salience and semantic search.

## Overview

Mnemosyne is a standalone MCP server that provides persistent memory storage with:

- 🧠 **Semantic Search** - ChromaDB vector database with embeddings
- 🎭 **Emotional Salience** - Memories tagged with emotional context
- 📊 **Memory Types** - Episodic, semantic, procedural, emotional
- 🔄 **Lifecycle Management** - Short-term → Long-term consolidation
- 🎯 **MCP Protocol** - Standard MCP server for easy integration

## Features

- Vector-based semantic search using sentence transformers
- Emotional context tracking (valence-arousal model)
- Importance scoring based on salience, recency, and access frequency
- Persistent storage via ChromaDB
- MCP tools for storing and retrieving memories

## Installation

```bash
pip install ./packages/mnemosyne
```

## Quick Start

### 1. Start the Server

```bash
mnemosyne-server
```

### 2. Configure MCP Client

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "mnemosyne": {
      "command": "mnemosyne-server"
    }
  }
}
```

### 3. Use MCP Tools

Available tools:

- `store_memory` - Store a new memory with optional emotional context
- `search_memories` - Semantic search for relevant memories
- `get_memory_stats` - Get memory statistics

## Memory Model

### Memory Types

- **Episodic** - Specific events/conversations
- **Semantic** - General knowledge/facts
- **Procedural** - How to do things
- **Emotional** - Emotional associations/patterns

### Memory Status

- **Short-term** - Recent, not yet consolidated
- **Consolidating** - Being processed
- **Long-term** - Permanently stored
- **Archived** - Old, rarely accessed

### Emotional Context

Memories can be tagged with emotional state at encoding:

```json
{
  "valence": 0.7,
  "arousal": 0.5,
  "quadrant": "excited"
}
```

This affects the memory's **salience score**, making emotionally significant
moments more retrievable (like in human memory).

## API Reference

### store_memory

Store a new memory.

**Arguments:**
- `content` (string, required) - Memory content
- `summary` (string) - Brief summary
- `memory_type` (string) - Type: episodic, semantic, procedural, emotional
- `tags` (array) - Tags for categorization
- `emotional_context` (object) - Emotional state at encoding

**Returns:**
- `id` (string) - Memory ID
- `importance_score` (float) - Computed importance
- `status` (string) - Storage status

### search_memories

Semantic search for memories.

**Arguments:**
- `query` (string, required) - Search query
- `n_results` (int) - Number of results (default: 10)

**Returns:**
- `query` (string) - Original query
- `results` (array) - Matching memories with scores

### get_memory_stats

Get memory statistics.

**Returns:**
- `total_memories` (int) - Total count
- `short_term` (int) - Short-term count
- `long_term` (int) - Long-term count

## Development

### Running Tests

```bash
pytest tests/
```

### Project Structure

```
packages/mnemosyne/
├── src/mnemosyne/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point
│   ├── server.py           # MCP server
│   ├── core/
│   │   └── models.py       # Memory data structures
│   ├── storage/
│   │   └── chroma_store.py # ChromaDB storage
│   └── consolidation/      # (Future) sleep consolidation
├── tests/
├── data/                   # Database storage
├── pyproject.toml
└── README.md
```

## License

GPL-3.0

## Credits

Built with:
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) - Embeddings
- [MCP SDK](https://github.com/anthropics/mcp) - Model Context Protocol
