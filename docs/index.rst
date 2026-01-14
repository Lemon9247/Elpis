Elpis Documentation
===================

.. image:: ../assets/elpis-inline.png
   :alt: Elpis Banner
   :align: center

*Do robots dream of electric sheep?*

Welcome to the Elpis documentation. Elpis is an MCP (Model Context Protocol) ecosystem
for emotionally-aware AI inference, consisting of three integrated components:

**Elpis** - An MCP inference server with emotional regulation that modulates LLM
generation based on a valence-arousal emotional model.

**Mnemosyne** - A memory MCP server providing persistent vector-based memory storage
using ChromaDB.

**Psyche** - A TUI (Text User Interface) client that brings everything together with
an interactive interface for working with the AI system.

Features
--------

- Emotional regulation using valence-arousal model
- Dual modulation approaches: sampling parameters or steering vectors
- Persistent memory with semantic search
- MCP-based architecture for modularity
- Interactive TUI client with rich output

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

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
