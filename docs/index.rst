Elpis Documentation
===================

.. image:: ../assets/elpis-inline.png
   :alt: Elpis Banner
   :align: center

*Do robots dream of electric sheep?*

Welcome to the Elpis documentation. Elpis is a system for giving AI persistent
memory and emotional state, consisting of four integrated components:

**Elpis** - Inference MCP server with emotional modulation via valence-arousal model.
Supports sampling parameter modulation and experimental steering vectors.

**Mnemosyne** - Memory MCP server with ChromaDB backend. Provides semantic search,
short/long-term memory stores, and clustering-based consolidation.

**Psyche** - Core library and HTTP server for memory coordination, tool execution,
and dreaming. Can run as a standalone server for remote access.

**Hermes** - TUI (Terminal User Interface) client built with Textual. Connects
to a Psyche server and executes file/bash/search tools locally.

Features
--------

- **Persistent Memory**: Semantic memory storage with automatic consolidation
- **Emotional Regulation**: Valence-arousal model modulating LLM generation
- **Server Architecture**: Psyche server with OpenAI-compatible API for remote access
- **Dreaming**: Memory-based introspection when no clients are connected
- **Tool System**: File operations, bash execution, codebase search with safety controls
- **MCP Architecture**: Modular design using Model Context Protocol

Architecture
------------

Psyche runs as a persistent HTTP server with an OpenAI-compatible API. It spawns
Elpis (inference) and Mnemosyne (memory) as MCP subprocesses and manages context,
memory retrieval, and dreaming.

Hermes (or other HTTP clients) connects to the Psyche server. Memory tools execute
server-side while file/bash/search tools execute client-side on Hermes.

Getting Started
---------------

New to Elpis? Start here:

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/index

Package Documentation
---------------------

Detailed documentation for each package:

.. toctree::
   :maxdepth: 2
   :caption: Packages

   elpis/index
   mnemosyne/index
   psyche/index
   hermes/index

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
