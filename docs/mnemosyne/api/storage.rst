==============
Storage Module
==============

.. module:: mnemosyne.storage.chroma_store
   :synopsis: ChromaDB-based memory storage

The storage module provides the ChromaDB-backed storage implementation for
Mnemosyne memories. It handles embedding generation, vector storage, and
semantic search.

ChromaMemoryStore
-----------------

.. autoclass:: mnemosyne.storage.chroma_store.ChromaMemoryStore
   :members:
   :undoc-members:
   :show-inheritance:

The main storage class that wraps ChromaDB for memory persistence.

Initialization
^^^^^^^^^^^^^^

.. code-block:: python

   from mnemosyne.storage.chroma_store import ChromaMemoryStore

   # Default configuration
   store = ChromaMemoryStore()

   # Custom configuration
   store = ChromaMemoryStore(
       persist_directory="./my_data/memories",
       embedding_model="all-MiniLM-L6-v2",
   )

**Constructor Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Default
     - Description
   * - ``persist_directory``
     - ``"./data/memory"``
     - Path to store ChromaDB data
   * - ``embedding_model``
     - ``"all-MiniLM-L6-v2"``
     - SentenceTransformer model name

**Dependencies:**

The storage module requires optional dependencies:

.. code-block:: bash

   pip install chromadb sentence-transformers

If these are not installed, initialization will raise a ``RuntimeError`` with
installation instructions.

Collections
^^^^^^^^^^^

The store maintains two ChromaDB collections:

.. attribute:: short_term
   :type: chromadb.Collection

   Collection for short-term memories (status: SHORT_TERM, CONSOLIDATING).

.. attribute:: long_term
   :type: chromadb.Collection

   Collection for long-term memories (status: LONG_TERM).

Adding Memories
---------------

.. method:: add_memory(memory: Memory) -> None

   Add a memory to the appropriate collection.

   The collection is selected based on the memory's status:

   - ``LONG_TERM``: Added to ``long_term`` collection
   - All others: Added to ``short_term`` collection

**Example:**

.. code-block:: python

   from mnemosyne.core.models import Memory, MemoryType, MemoryStatus

   memory = Memory(
       content="Important fact to remember",
       memory_type=MemoryType.SEMANTIC,
       status=MemoryStatus.SHORT_TERM,
   )

   store.add_memory(memory)

**What happens internally:**

1. Generate embedding using SentenceTransformer
2. Select collection based on status
3. Store document, embedding, and metadata in ChromaDB

Retrieving Memories
-------------------

By ID
^^^^^

.. method:: get_memory(memory_id: str) -> Optional[Memory]

   Retrieve a memory by its unique ID.

   Searches both collections, checking short-term first.

**Example:**

.. code-block:: python

   memory = store.get_memory("550e8400-e29b-41d4-a716-446655440000")
   if memory:
       print(f"Found: {memory.content}")
   else:
       print("Memory not found")

Semantic Search
^^^^^^^^^^^^^^^

.. method:: search_memories(query: str, n_results: int = 10, status_filter: Optional[MemoryStatus] = None) -> List[Memory]

   Search memories semantically using vector similarity.

**Parameters:**

- ``query``: Search query string
- ``n_results``: Maximum number of results (default: 10)
- ``status_filter``: Optional filter by memory status

**Example:**

.. code-block:: python

   # Search all memories
   results = store.search_memories("user preferences", n_results=5)

   # Search only long-term memories
   from mnemosyne.core.models import MemoryStatus

   results = store.search_memories(
       "user preferences",
       n_results=5,
       status_filter=MemoryStatus.LONG_TERM,
   )

   for memory in results:
       print(f"[{memory.importance_score:.2f}] {memory.summary}")

**How search works:**

1. Query is embedded using the same SentenceTransformer model
2. Collections are selected based on status filter
3. ChromaDB performs approximate nearest neighbor search
4. Results are converted to Memory objects
5. Results are sorted by importance score
6. Top N results are returned

Counting Memories
-----------------

.. method:: count_memories(status: Optional[MemoryStatus] = None) -> int

   Count memories, optionally filtered by status.

**Examples:**

.. code-block:: python

   # Total count
   total = store.count_memories()
   print(f"Total memories: {total}")

   # By status
   short_term = store.count_memories(MemoryStatus.SHORT_TERM)
   long_term = store.count_memories(MemoryStatus.LONG_TERM)

   print(f"Short-term: {short_term}")
   print(f"Long-term: {long_term}")

Internal Methods
----------------

These methods handle conversion between ChromaDB results and Memory objects:

.. method:: _result_to_memory(result: Dict, idx: int) -> Memory

   Convert a ChromaDB ``get()`` result to a Memory object.

.. method:: _query_result_to_memory(results: Dict, query_idx: int, result_idx: int) -> Memory

   Convert a ChromaDB ``query()`` result to a Memory object.

Embedding Model
---------------

The store uses SentenceTransformers for embedding generation:

.. attribute:: embedding_model
   :type: SentenceTransformer

   The loaded embedding model instance.

**Default Model:** ``all-MiniLM-L6-v2``

- Produces 384-dimensional embeddings
- Good balance of speed and quality
- ~80MB model size

**Using a Different Model:**

.. code-block:: python

   # Use a larger, more accurate model
   store = ChromaMemoryStore(
       embedding_model="all-mpnet-base-v2",  # 768 dimensions
   )

   # Use a multilingual model
   store = ChromaMemoryStore(
       embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
   )

ChromaDB Configuration
----------------------

The store configures ChromaDB with:

.. code-block:: python

   self.client = chromadb.PersistentClient(
       path=str(self.persist_directory),
       settings=Settings(
           anonymized_telemetry=False,
           allow_reset=True,
       ),
   )

**Settings:**

- ``anonymized_telemetry=False``: Disable usage tracking
- ``allow_reset=True``: Allow database reset (useful for testing)

Data Persistence
^^^^^^^^^^^^^^^^

Data is persisted to the specified directory:

.. code-block:: text

   persist_directory/
   ├── chroma.sqlite3          # Metadata database
   └── [collection-uuid]/      # Vector data
       ├── data_level0.bin
       ├── header.bin
       ├── index_metadata.json
       └── length.bin

Error Handling
--------------

The module handles import errors gracefully:

.. code-block:: python

   try:
       import chromadb
       CHROMADB_AVAILABLE = True
   except ImportError:
       CHROMADB_AVAILABLE = False

If dependencies are missing, initialization raises ``RuntimeError`` with
helpful installation instructions.

See Also
--------

- :doc:`models` - Memory data structures
- :doc:`server` - MCP server that uses the store
- :doc:`../architecture` - System architecture overview
