======================
Memory Consolidation
======================

Mnemosyne implements biologically-inspired memory consolidation to promote important
short-term memories to long-term storage. This document provides detailed documentation
on the consolidation system.

Overview
--------

Memory consolidation is the process by which temporary memories are converted into
stable, long-lasting memories. In Mnemosyne, this is implemented through:

1. **Short-term buffer**: New memories are stored in the short-term collection
2. **Importance scoring**: Memories are scored based on salience, recency, and access
3. **Clustering**: Similar memories are grouped using semantic similarity
4. **Promotion**: High-importance cluster representatives are moved to long-term storage
5. **Archival**: Redundant cluster members are deleted

Configuration
-------------

Consolidation behavior is controlled by ``ConsolidationConfig``:

.. code-block:: python

   @dataclass
   class ConsolidationConfig:
       importance_threshold: float = 0.6      # Min importance for promotion
       min_age_hours: int = 1                 # Min age before eligible
       max_batch_size: int = 50               # Max memories per consolidation
       buffer_threshold: int = 100            # Trigger recommendation threshold
       similarity_threshold: float = 0.85     # Cosine similarity for clustering
       min_cluster_size: int = 2              # Min memories to form cluster

Parameters
^^^^^^^^^^

**importance_threshold** (default: 0.6)
    Minimum average importance score for a cluster to be promoted.
    Higher values mean only very important memories are kept.

**min_age_hours** (default: 1)
    Minimum age in hours before a memory is eligible for consolidation.
    Prevents very recent memories from being processed.

**max_batch_size** (default: 50)
    Maximum number of memories to process in a single consolidation cycle.
    Limits processing time and resource usage.

**buffer_threshold** (default: 100)
    Number of short-term memories that triggers a consolidation recommendation.
    When ``should_consolidate`` is called, it returns true if this threshold is exceeded.

**similarity_threshold** (default: 0.85)
    Cosine similarity threshold for clustering. Memories with similarity >= this
    value are grouped together. Higher values create more granular clusters.

**min_cluster_size** (default: 2)
    Minimum number of memories to form a cluster. Singletons (size 1) are still
    evaluated but represent unique memories.

MCP Tools
---------

consolidate_memories
^^^^^^^^^^^^^^^^^^^^

Runs a memory consolidation cycle with clustering.

**Input Schema:**

.. code-block:: json

   {
       "type": "object",
       "properties": {
           "importance_threshold": {
               "type": "number",
               "description": "Minimum importance for promotion (0.0 to 1.0)",
               "default": 0.6
           },
           "similarity_threshold": {
               "type": "number",
               "description": "Similarity threshold for clustering (0.0 to 1.0)",
               "default": 0.85
           }
       }
   }

**Response:**

.. code-block:: json

   {
       "clusters_formed": 5,
       "memories_promoted": 3,
       "memories_archived": 12,
       "memories_skipped": 2,
       "duration_seconds": 0.45,
       "cluster_summaries": [
           {
               "promoted_id": "uuid-1234",
               "source_ids": ["uuid-5678", "uuid-9012"],
               "cluster_size": 3,
               "avg_importance": 0.75
           }
       ]
   }

should_consolidate
^^^^^^^^^^^^^^^^^^

Checks if memory consolidation is recommended based on buffer size.

**Input Schema:**

.. code-block:: json

   {
       "type": "object",
       "properties": {}
   }

**Response:**

.. code-block:: json

   {
       "should_consolidate": true,
       "reason": "Buffer size (150) exceeds threshold (100)",
       "short_term_count": 150,
       "long_term_count": 42
   }

get_memory_context
^^^^^^^^^^^^^^^^^^

Gets relevant memories formatted for context injection with a token budget.

**Input Schema:**

.. code-block:: json

   {
       "type": "object",
       "properties": {
           "query": {
               "type": "string",
               "description": "Query to search for relevant memories"
           },
           "max_tokens": {
               "type": "integer",
               "description": "Maximum tokens for the result",
               "default": 2000
           }
       },
       "required": ["query"]
   }

**Response:**

.. code-block:: json

   {
       "memories": [
           {
               "id": "uuid-1234",
               "summary": "User prefers dark mode",
               "content": "User mentioned they prefer dark mode interfaces...",
               "importance": 0.75
           }
       ],
       "token_count": 1850,
       "truncated": false
   }

Clustering Algorithm
--------------------

The consolidation system uses a greedy clustering algorithm based on cosine
similarity of memory embeddings.

Algorithm Steps
^^^^^^^^^^^^^^^

1. **Candidate Selection**

   .. code-block:: python

      # Get memories older than min_age_hours
      cutoff = datetime.now() - timedelta(hours=config.min_age_hours)
      candidates = [m for m in short_term_memories if m.created_at <= cutoff]

      # Recompute importance scores
      for memory in candidates:
          memory.importance_score = memory.compute_importance()

      # Sort by importance, take top max_batch_size
      candidates = sorted(candidates, key=lambda m: m.importance_score, reverse=True)
      candidates = candidates[:config.max_batch_size]

2. **Embedding Retrieval**

   Embeddings are retrieved directly from ChromaDB without re-embedding:

   .. code-block:: python

      memory_ids = [m.id for m in candidates]
      embeddings = store.get_embeddings_batch(memory_ids)
      # Returns: {memory_id: [float, ...], ...}

3. **Greedy Clustering**

   .. code-block:: python

      clusters = []
      assigned = set()

      for memory in candidates:
          if memory.id in assigned:
              continue

          cluster_members = [memory]
          cluster_embedding = np.array(embeddings[memory.id])
          assigned.add(memory.id)

          # Find similar memories
          for other in candidates:
              if other.id in assigned:
                  continue

              other_emb = np.array(embeddings[other.id])
              similarity = cosine_similarity(cluster_embedding, other_emb)

              if similarity >= config.similarity_threshold:
                  cluster_members.append(other)
                  assigned.add(other.id)
                  # Update centroid (running average)
                  cluster_embedding = (cluster_embedding + other_emb) / 2

          clusters.append(MemoryCluster(
              memories=cluster_members,
              centroid_embedding=cluster_embedding.tolist(),
              avg_importance=mean(m.importance_score for m in cluster_members),
              dominant_type=mode(m.memory_type for m in cluster_members)
          ))

4. **Promotion Decision**

   For each cluster, if the average importance meets the threshold:

   .. code-block:: python

      if cluster.avg_importance >= config.importance_threshold:
          # Select highest-importance memory as representative
          representative = max(cluster.memories, key=lambda m: m.importance_score)

          # Record lineage
          representative.source_memory_ids = [
              m.id for m in cluster.memories if m.id != representative.id
          ]

          # Promote to long-term
          store.promote_memory(representative.id)

          # Delete other cluster members
          for m in cluster.memories:
              if m.id != representative.id:
                  store.delete_memory(m.id)

Lineage Tracking
----------------

When memories are consolidated, the ``source_memory_ids`` field tracks which
original memories contributed to the promoted memory:

.. code-block:: python

   @dataclass
   class Memory:
       id: str
       content: str
       source_memory_ids: List[str] = field(default_factory=list)  # Lineage

This enables:

- Tracing the origin of long-term memories
- Understanding how clusters were formed
- Debugging consolidation decisions

Psyche Integration
------------------

Psyche automatically triggers consolidation during idle periods:

.. code-block:: python

   # Psyche triggers consolidation during idle periods
   async def _maybe_consolidate_memories(self) -> None:
       if not self.config.enable_consolidation:
           return

       if not self.mnemosyne_client or not self.mnemosyne_client.is_connected:
           return

       # Check interval
       elapsed = time.time() - self._last_consolidation_check
       if elapsed < self.config.consolidation_check_interval:
           return

       self._last_consolidation_check = time.time()

       # Check if consolidation is recommended
       should, reason, st_count, lt_count = await self.mnemosyne_client.should_consolidate()

       if should:
           result = await self.mnemosyne_client.consolidate_memories(
               importance_threshold=self.config.consolidation_importance_threshold,
               similarity_threshold=self.config.consolidation_similarity_threshold,
           )
           logger.info(f"Consolidation complete: {result.memories_promoted} promoted")

Configuration in Psyche
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   ServerConfig(
       enable_consolidation=True,              # Enable/disable consolidation
       consolidation_check_interval=300.0,     # Check every 5 minutes
       consolidation_importance_threshold=0.6, # Min importance for promotion
       consolidation_similarity_threshold=0.85 # Clustering threshold
   )

CLI Options
^^^^^^^^^^^

.. code-block:: bash

   # Default (with consolidation)
   psyche

   # Disable consolidation
   psyche --no-consolidation

   # Custom Mnemosyne command
   psyche --mnemosyne-command "mnemosyne-server --persist-dir ./custom/path"

Best Practices
--------------

Threshold Tuning
^^^^^^^^^^^^^^^^

- **High importance threshold (0.7-0.9)**: Only very important memories are kept.
  Good for systems with limited storage or when quality over quantity is needed.

- **Low importance threshold (0.4-0.6)**: More memories are promoted.
  Good for comprehensive memory retention.

- **High similarity threshold (0.9+)**: Only very similar memories cluster together.
  Creates more granular clusters, promotes more representatives.

- **Low similarity threshold (0.7-0.85)**: More aggressive clustering.
  Reduces redundancy but may merge slightly different memories.

Timing Considerations
^^^^^^^^^^^^^^^^^^^^^

- **min_age_hours**: Set higher (2-4 hours) to allow memories to accumulate
  access counts before consolidation decisions.

- **consolidation_check_interval**: Balance between responsiveness (lower values)
  and resource usage (higher values).

Monitoring
^^^^^^^^^^

The ``ConsolidationReport`` provides metrics for monitoring:

- ``clusters_formed``: Number of semantic clusters identified
- ``memories_promoted``: Memories moved to long-term storage
- ``memories_archived``: Redundant memories deleted
- ``memories_skipped``: Low-importance clusters not promoted
- ``duration_seconds``: Processing time

See Also
--------

- :doc:`architecture` - System architecture overview
- :doc:`memory-types` - Understanding memory types
- :doc:`api/models` - Data model reference
