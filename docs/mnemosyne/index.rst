=========
Mnemosyne
=========

Mnemosyne is a semantic memory server that provides intelligent memory storage and retrieval
through the Model Context Protocol (MCP). Named after the Greek goddess of memory, Mnemosyne
enables AI systems to store, search, and recall memories with emotional context awareness.

Key Features
------------

**Semantic Memory Storage**
    Store memories with rich metadata including emotional context, importance scoring,
    and relationship tracking. Memories are embedded using sentence transformers for
    powerful semantic search capabilities.

**ChromaDB Vector Backend**
    Uses ChromaDB for efficient vector storage and retrieval. Memories are automatically
    embedded using the ``all-MiniLM-L6-v2`` model from SentenceTransformers.

**Memory Types**
    Supports four memory types modeled after human cognition:

    - **Episodic**: Specific events and conversations
    - **Semantic**: General knowledge and facts
    - **Procedural**: How-to knowledge and processes
    - **Emotional**: Emotional associations and patterns

**Importance Scoring**
    Automatically computes memory importance using a weighted algorithm:

    - Emotional salience (40%)
    - Recency (30%)
    - Access frequency (30%)

**MCP Integration**
    Exposes memory operations as MCP tools, making it easy to integrate with any
    MCP-compatible AI system.

Quick Start
-----------

Start the Mnemosyne server:

.. code-block:: bash

   mnemosyne-server --persist-dir ./data/memory

The server exposes three MCP tools:

- ``store_memory``: Store a new memory with optional emotional context
- ``search_memories``: Semantic search across all memories
- ``get_memory_stats``: Get memory statistics (counts by status)

Example: Storing a Memory
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Via MCP tool call
   {
       "content": "User prefers dark mode interfaces",
       "summary": "UI preference: dark mode",
       "memory_type": "semantic",
       "tags": ["preferences", "ui"],
       "emotional_context": {
           "valence": 0.3,
           "arousal": 0.1,
           "quadrant": "calm"
       }
   }

Example: Searching Memories
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Via MCP tool call
   {
       "query": "user interface preferences",
       "n_results": 5
   }

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   architecture
   memory-types
   api/index

See Also
--------

- :doc:`/elpis/index` - Inference server with emotional modulation
- :doc:`/psyche/index` - TUI client for interacting with the system
- :doc:`/getting-started/installation` - Installation guide
