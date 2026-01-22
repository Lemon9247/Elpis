=============
API Reference
=============

This section contains the complete API reference for the Mnemosyne memory server,
generated from docstrings in the source code.

Modules
-------

.. toctree::
   :maxdepth: 2
   :caption: API Modules:

   server
   models
   storage
   constants

Quick Reference
---------------

Core Classes
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`~mnemosyne.core.models.Memory`
     - Core memory data structure
   * - :class:`~mnemosyne.core.models.MemoryType`
     - Enum for memory classification
   * - :class:`~mnemosyne.core.models.MemoryStatus`
     - Enum for memory lifecycle status
   * - :class:`~mnemosyne.core.models.EmotionalContext`
     - Emotional state at encoding time
   * - :class:`~mnemosyne.storage.chroma_store.ChromaMemoryStore`
     - ChromaDB storage backend

Server Functions
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`~mnemosyne.server.initialize`
     - Initialize server components
   * - :func:`~mnemosyne.server.run_server`
     - Run the MCP server

MCP Tools
^^^^^^^^^

The server exposes these tools via MCP:

``store_memory``
    Store a new memory with content, type, tags, and emotional context.

``search_memories``
    Semantic search across all stored memories.

``get_memory_stats``
    Get statistics about memory counts by status.

See Also
--------

- :doc:`../architecture` - System architecture overview
- :doc:`../memory-types` - Memory types and lifecycle guide
