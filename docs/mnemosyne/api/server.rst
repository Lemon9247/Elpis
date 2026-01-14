=============
Server Module
=============

.. module:: mnemosyne.server
   :synopsis: MCP server for memory management

The server module implements the MCP server that exposes memory operations as tools.
It handles initialization, tool registration, and request processing.

Module Overview
---------------

The server provides:

- MCP server instance using the ``mcp`` library
- Three MCP tools for memory operations
- Initialization function for setting up the memory store
- Async server runner for stdio communication

Server Initialization
---------------------

.. autofunction:: mnemosyne.server.initialize

This function must be called before running the server to set up the ChromaDB
memory store.

.. code-block:: python

   from mnemosyne.server import initialize, run_server

   # Initialize with custom directory
   initialize(persist_directory="./my_memories")

   # Run the server
   import asyncio
   asyncio.run(run_server())

Server Runner
-------------

.. autofunction:: mnemosyne.server.run_server

The server uses stdio for MCP communication. It should be started as a subprocess
by an MCP client.

MCP Tools
---------

The server exposes three tools via the MCP protocol:

store_memory
^^^^^^^^^^^^

Store a new memory in the database.

**Input Schema:**

.. code-block:: json

   {
     "type": "object",
     "properties": {
       "content": {
         "type": "string",
         "description": "Memory content"
       },
       "summary": {
         "type": "string",
         "description": "Brief summary"
       },
       "memory_type": {
         "type": "string",
         "enum": ["episodic", "semantic", "procedural", "emotional"],
         "default": "episodic"
       },
       "tags": {
         "type": "array",
         "items": {"type": "string"},
         "description": "Memory tags"
       },
       "emotional_context": {
         "type": "object",
         "properties": {
           "valence": {"type": "number"},
           "arousal": {"type": "number"},
           "quadrant": {"type": "string"}
         }
       }
     },
     "required": ["content"]
   }

**Example:**

.. code-block:: python

   # Tool call arguments
   {
       "content": "User prefers Python over JavaScript",
       "summary": "Language preference",
       "memory_type": "semantic",
       "tags": ["preferences", "languages"],
       "emotional_context": {
           "valence": 0.5,
           "arousal": 0.2,
           "quadrant": "calm"
       }
   }

**Response:**

.. code-block:: json

   {
     "id": "550e8400-e29b-41d4-a716-446655440000",
     "importance_score": 0.65,
     "status": "stored"
   }

search_memories
^^^^^^^^^^^^^^^

Perform semantic search across all memories.

**Input Schema:**

.. code-block:: json

   {
     "type": "object",
     "properties": {
       "query": {
         "type": "string",
         "description": "Search query"
       },
       "n_results": {
         "type": "integer",
         "default": 10,
         "description": "Number of results"
       }
     },
     "required": ["query"]
   }

**Example:**

.. code-block:: python

   # Tool call arguments
   {
       "query": "programming language preferences",
       "n_results": 5
   }

**Response:**

.. code-block:: json

   {
     "query": "programming language preferences",
     "results": [
       {
         "id": "550e8400-e29b-41d4-a716-446655440000",
         "content": "User prefers Python over JavaScript",
         "summary": "Language preference",
         "importance_score": 0.65,
         "created_at": "2024-01-15T10:30:00"
       }
     ]
   }

get_memory_stats
^^^^^^^^^^^^^^^^

Get statistics about stored memories.

**Input Schema:**

.. code-block:: json

   {
     "type": "object",
     "properties": {}
   }

**Response:**

.. code-block:: json

   {
     "total_memories": 42,
     "short_term": 15,
     "long_term": 27
   }

Internal Functions
------------------

These functions handle the actual tool implementations:

.. function:: _handle_store_memory(args: Dict[str, Any]) -> Dict[str, Any]
   :async:

   Process a store_memory tool call. Creates a Memory object from the arguments,
   computes importance score, and stores it in the appropriate collection.

.. function:: _handle_search_memories(args: Dict[str, Any]) -> Dict[str, Any]
   :async:

   Process a search_memories tool call. Performs semantic search and returns
   matching memories.

.. function:: _handle_get_stats() -> Dict[str, Any]
   :async:

   Process a get_memory_stats tool call. Returns counts from both collections.

.. function:: _ensure_initialized() -> None

   Check that the server has been initialized. Raises RuntimeError if not.

Global State
------------

.. data:: memory_store
   :type: Optional[ChromaMemoryStore]

   The initialized memory store instance. Set by :func:`initialize`.

.. data:: server
   :type: Server

   The MCP Server instance, created at module load time.

See Also
--------

- :doc:`storage` - ChromaMemoryStore API
- :doc:`models` - Memory data structures
- :doc:`../architecture` - System architecture
