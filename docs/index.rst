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

**Hermes** - TUI (Terminal User Interface) client built with Textual. Supports
both local mode (spawns servers) and remote mode (connects to Psyche server).

Features
--------

- **Persistent Memory**: Semantic memory storage with automatic consolidation
- **Emotional Regulation**: Valence-arousal model modulating LLM generation
- **Local & Remote Modes**: Run everything locally or as a server for remote access
- **Dreaming**: Memory-based introspection when no clients are connected
- **Tool System**: File operations, bash execution, codebase search with safety controls
- **MCP Architecture**: Modular design using Model Context Protocol

Architecture
------------

**Local Mode** (default): Hermes spawns Elpis and Mnemosyne as MCP subprocesses,
managing everything in a single terminal session.

**Server Mode**: Psyche runs as a persistent HTTP server with OpenAI-compatible API.
Hermes or other clients connect remotely. Memory tools execute server-side while
file/bash tools execute client-side.

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
