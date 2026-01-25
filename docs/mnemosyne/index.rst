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

**Long-term Memory Consolidation**
    Biologically-inspired memory consolidation promotes important short-term memories
    to long-term storage. Features include:

    - Clustering-based consolidation using cosine similarity on embeddings
    - Importance-threshold promotion (configurable, default 0.6)
    - Automatic archival of redundant cluster members
    - Lineage tracking via ``source_memory_ids`` field

**ChromaDB Vector Backend**
    Uses ChromaDB for efficient vector storage and retrieval. Memories are automatically
    embedded using the ``all-MiniLM-L6-v2`` model from SentenceTransformers.

**Hybrid Search**
    Combines vector similarity with BM25 keyword matching using Reciprocal Rank
    Fusion (RRF). Quality scoring factors in recency, importance, and access frequency.
    Optional emotional reranking for mood-congruent retrieval.

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

The server exposes eight MCP tools:

**Memory Storage**

- ``store_memory``: Store a new memory with optional emotional context
- ``search_memories``: Semantic search across all memories
- ``get_memory_stats``: Get memory statistics (counts by status)
- ``delete_memory``: Delete a memory by ID
- ``get_recent_memories``: Get memories from the last N hours

**Consolidation**

- ``consolidate_memories``: Run clustering-based memory consolidation
- ``should_consolidate``: Check if consolidation is recommended
- ``get_memory_context``: Get relevant memories for context injection

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

Example: Memory Consolidation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Check if consolidation is recommended
   result = await mnemosyne.call_tool("should_consolidate", {})
   # Returns: {"should_consolidate": true, "reason": "Buffer size (150) exceeds threshold", ...}

   # Run consolidation
   report = await mnemosyne.call_tool("consolidate_memories", {
       "importance_threshold": 0.6,
       "similarity_threshold": 0.85
   })
   # Returns: {"clusters_formed": 5, "memories_promoted": 3, "memories_archived": 12, ...}

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   architecture
   consolidation
   memory-types
   api/index

See Also
--------

- :doc:`/elpis/index` - Inference server with emotional modulation
- :doc:`/psyche/index` - Core server for coordination
- :doc:`/hermes/index` - TUI client for interacting with the system
- :doc:`/getting-started/installation` - Installation guide
