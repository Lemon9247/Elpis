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

Project Structure
-----------------

.. code-block:: text

   src/
     elpis/           # Inference MCP server
      - config/        # Settings management
      - emotion/       # Valence-arousal state and regulation
      - llm/           # Inference backends (llama-cpp, transformers)
      - server.py      # MCP server entry point

     mnemosyne/       # Memory MCP server
      - core/          # Memory models and consolidator
      - storage/       # ChromaDB storage backend
      - server.py      # MCP server entry point

     psyche/          # TUI client
      - client/        # Textual TUI components
      - memory/        # Inference server with consolidation
      - tools/         # Tool definitions and implementations
      - mcp/           # MCP clients for Elpis and Mnemosyne
