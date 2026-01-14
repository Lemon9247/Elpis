Getting Started
===============

This section will help you get up and running with the Elpis ecosystem.

The Elpis project consists of three main components:

- **elpis-server** - The inference MCP server with emotional modulation
- **mnemosyne-server** - The memory MCP server with ChromaDB storage
- **psyche** - The TUI client for interacting with the system

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   quickstart

Overview
--------

Elpis uses the Model Context Protocol (MCP) to provide a modular architecture
where different components can be combined as needed:

- Run the inference server standalone for emotional LLM generation
- Add the memory server for persistent context across sessions
- Use the Psyche client for a complete interactive experience

The emotional regulation system uses a valence-arousal model to modulate
LLM generation, creating more nuanced and contextually appropriate responses.
