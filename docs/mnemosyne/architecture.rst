============
Architecture
============

Mnemosyne is designed as a modular memory system with clear separation between storage,
embedding, and the MCP server layer. This document describes the internal architecture
and how memories flow through the system.

System Overview
---------------

.. code-block:: text

                    +------------------+
                    |   MCP Client     |
                    |     (Psyche)     |
                    +--------+---------+
                             |
                             | stdio (JSON-RPC)
                             v
                    +------------------+
                    |  MCP Server      |
                    |  (server.py)     |
                    +--------+---------+
                             |
            +----------------+----------------+
            |                                 |
            v                                 v
    +------------------+              +------------------+
    | ChromaMemoryStore|              | MemoryConsolidator|
    +--------+---------+              +--------+---------+
             |                                 |
             +----------------+----------------+
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
     +---------------+                 +---------------+
     | Short-term    |  --promote-->   | Long-term     |
     | Collection    |                 | Collection    |
     +---------------+                 +---------------+
             |                                 |
             +----------------+----------------+
                              |
                              v
                     +------------------+
                     |    ChromaDB      |
                     | (Vector Storage) |
                     +------------------+

Storage Design
--------------

Short-term vs Long-term Collections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mnemosyne maintains two separate ChromaDB collections to model memory consolidation:

**Short-term Memory** (``short_term_memory``)
    Recent memories that have not yet been consolidated. This is the default
    destination for new memories. Memories here are:

    - Newly created memories
    - Memories in ``CONSOLIDATING`` status
    - Memories that may be forgotten if not reinforced

**Long-term Memory** (``long_term_memory``)
    Consolidated memories that have been deemed important enough to persist.
    Memories here:

    - Have ``LONG_TERM`` status
    - Have higher importance scores
    - Are more likely to be retrieved in searches

Memory Storage Format
^^^^^^^^^^^^^^^^^^^^^

Each memory is stored in ChromaDB with:

- **ID**: UUID string for unique identification
- **Document**: The memory content (used for display and backup)
- **Embedding**: 384-dimensional vector from ``all-MiniLM-L6-v2``
- **Metadata**: JSON-serialized fields including:

  - ``summary``: Brief summary of the memory
  - ``memory_type``: One of episodic, semantic, procedural, emotional
  - ``status``: Current lifecycle status
  - ``importance_score``: Computed importance (0-1)
  - ``created_at``: ISO timestamp
  - ``tags``: JSON array of tags
  - ``metadata_json``: Additional custom metadata
  - ``emotional_context``: JSON object with valence, arousal, quadrant

Embedding Pipeline
------------------

Model Selection
^^^^^^^^^^^^^^^

Mnemosyne uses the ``all-MiniLM-L6-v2`` model from SentenceTransformers:

- **Dimensions**: 384
- **Model Size**: ~80MB
- **Speed**: Fast inference, suitable for real-time use
- **Quality**: Good balance of performance and accuracy for semantic similarity

The model is loaded once during initialization and reused for all embedding operations.

Embedding Process
^^^^^^^^^^^^^^^^^

1. **Input**: Memory content string
2. **Tokenization**: Text is tokenized using the model's tokenizer
3. **Encoding**: Tokens are passed through the transformer
4. **Pooling**: Mean pooling produces a fixed-size vector
5. **Storage**: Vector is stored in ChromaDB alongside the document

.. code-block:: python

   # Simplified embedding process
   embedding = self.embedding_model.encode(memory.content).tolist()

Search Mechanisms
-----------------

Mnemosyne supports two search modes: basic vector search and hybrid search
combining vector similarity with BM25 keyword matching.

Basic Vector Search
^^^^^^^^^^^^^^^^^^^

When a search query is received:

1. **Query Embedding**: The query string is embedded using the same model
2. **Collection Selection**: Determines which collections to search:

   - Both collections if no status filter
   - Only matching collection if status filter provided

3. **Vector Search**: ChromaDB performs approximate nearest neighbor search
4. **Result Merging**: Results from multiple collections are combined
5. **Ranking**: Results are sorted by importance score
6. **Return**: Top N results are returned as Memory objects

.. code-block:: python

   # Basic vector search
   query_embedding = self.embedding_model.encode(query).tolist()
   results = collection.query(
       query_embeddings=[query_embedding],
       n_results=n_results,
   )

Hybrid Search (Vector + BM25)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hybrid search combines semantic vector search with BM25 keyword matching using
Reciprocal Rank Fusion (RRF). This approach improves retrieval quality by:

- Capturing semantic similarity (vector) for conceptually related content
- Capturing exact keyword matches (BM25) for specific terms

**BM25 Index**

Mnemosyne maintains a lazy-built BM25 index from ``rank-bm25``:

- Index rebuilds automatically when marked dirty (after add/delete operations)
- Tokenization with stop word removal for better relevance
- Lazy initialization defers work until first search

**Reciprocal Rank Fusion**

RRF combines the two ranking lists using the formula:

.. code-block:: text

   RRF_score = sum(weight / (k + rank)) for each ranking list

Where ``k=60`` (standard RRF constant), and weights default to 0.5 each.

**Hybrid Search Process**

1. Embed query and run vector search
2. Run BM25 keyword search (if available)
3. Combine rankings with RRF
4. Optionally apply quality scoring (recency, importance, access count)
5. Optionally apply emotional similarity reranking
6. Return top N results

.. code-block:: python

   # Hybrid search with quality scoring
   memories = store.search_memories_hybrid(
       query="user interface preferences",
       n_results=10,
       vector_weight=0.5,
       bm25_weight=0.5,
       use_quality_scoring=True,
       emotional_context=current_emotion,
       emotion_weight=0.3,
   )

**Quality Scoring Factors**

When ``use_quality_scoring=True``, results are reranked by:

- **Recency**: More recently accessed memories score higher
- **Importance**: Higher importance_score increases ranking
- **Access frequency**: Frequently accessed memories score higher

**Emotional Reranking**

When ``emotional_context`` is provided, memories with similar emotional
states are boosted. This implements mood-congruent memory retrieval.

Distance Metrics
^^^^^^^^^^^^^^^^

ChromaDB uses cosine similarity by default, which measures the angle between vectors.
Memories with similar semantic content will have embeddings pointing in similar
directions, resulting in higher similarity scores.

Memory Flow
-----------

Creation Flow
^^^^^^^^^^^^^

.. code-block:: text

   1. MCP tool call: store_memory
   2. Parse arguments, create Memory object
   3. Compute importance score
   4. Generate embedding via SentenceTransformer
   5. Select collection based on status
   6. Store in ChromaDB with metadata
   7. Return memory ID and status

Retrieval Flow
^^^^^^^^^^^^^^

.. code-block:: text

   1. MCP tool call: search_memories
   2. Generate query embedding
   3. Query both collections (or filtered)
   4. Convert results to Memory objects
   5. Sort by importance score
   6. Return top N results

Memory Consolidation
--------------------

The ``MemoryConsolidator`` class implements clustering-based memory consolidation,
promoting important short-term memories to long-term storage.

Consolidation Algorithm
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   1. MCP tool call: consolidate_memories
   2. Get candidates: short-term memories older than min_age_hours
   3. Recompute importance scores for all candidates
   4. Cluster memories using cosine similarity on embeddings
   5. For each cluster with avg_importance >= threshold:
      a. Select highest-importance memory as representative
      b. Store source_memory_ids for lineage tracking
      c. Promote representative to long-term collection
      d. Delete other cluster members
   6. Return consolidation report

Clustering Process
^^^^^^^^^^^^^^^^^^

The greedy clustering algorithm groups semantically similar memories:

1. **Get embeddings**: Retrieve stored embeddings from ChromaDB (no re-embedding needed)
2. **Initialize**: Start with the first unassigned memory as a cluster seed
3. **Expand cluster**: Add memories with cosine similarity >= threshold (default 0.85)
4. **Update centroid**: Compute running average of cluster embeddings
5. **Repeat**: Process remaining unassigned memories

.. code-block:: python

   def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
       """Compute cosine similarity between two vectors."""
       norm_a = np.linalg.norm(a)
       norm_b = np.linalg.norm(b)
       if norm_a == 0 or norm_b == 0:
           return 0.0
       return float(np.dot(a, b) / (norm_a * norm_b))

Promotion Flow
^^^^^^^^^^^^^^

.. code-block:: text

   1. Get memory from short_term collection (with embedding)
   2. Update status to LONG_TERM
   3. Add to long_term collection with same embedding
   4. Delete from short_term collection

Consolidation Trigger
^^^^^^^^^^^^^^^^^^^^^

Consolidation can be triggered in two ways:

**Manual**: Call ``consolidate_memories`` tool directly

**Automatic**: Psyche calls ``should_consolidate`` during idle periods and runs
consolidation when the short-term buffer exceeds the threshold (default: 100 memories)

Persistence
-----------

ChromaDB Configuration
^^^^^^^^^^^^^^^^^^^^^^

Mnemosyne uses ChromaDB's ``PersistentClient`` for durable storage:

.. code-block:: python

   self.client = chromadb.PersistentClient(
       path=str(self.persist_directory),
       settings=Settings(
           anonymized_telemetry=False,
           allow_reset=True,
       ),
   )

Data is stored at the configured ``persist_directory`` (default: ``./data/memory``).
The directory is created automatically if it doesn't exist.

Data Files
^^^^^^^^^^

ChromaDB stores data in SQLite and binary files:

- ``chroma.sqlite3``: Metadata and index
- ``*.bin``: Vector embeddings
- ``*.pkl``: Serialized Python objects

See Also
--------

- :doc:`consolidation` - Detailed consolidation documentation
- :doc:`memory-types` - Understanding memory types and lifecycle
- :doc:`api/storage` - ChromaMemoryStore API reference
- :doc:`api/models` - Memory data structures
