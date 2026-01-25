Getting Started
===============

This section will help you get up and running with the Elpis ecosystem.

The Elpis project consists of four main components:

- **Elpis** - Inference MCP server with emotional modulation via valence-arousal model
- **Mnemosyne** - Memory MCP server with ChromaDB backend and short/long-term consolidation
- **Psyche** - Core server coordinating memory, inference, and dreaming with an OpenAI-compatible API
- **Hermes** - TUI (Terminal User Interface) client that connects to Psyche and executes tools locally

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   quickstart

Overview
--------

Elpis uses a server-client architecture with the Model Context Protocol (MCP) for
internal component communication:

- **Psyche server** manages context, memory, and emotional state, spawning Elpis and
  Mnemosyne as MCP subprocesses
- **Hermes** (or any HTTP client) connects to Psyche's OpenAI-compatible API
- Memory tools execute server-side while file/bash/search tools execute client-side
- The server dreams (memory-based introspection) when no clients are connected

The emotional regulation system uses a valence-arousal model with trajectory tracking
to modulate LLM generation, creating more nuanced and contextually appropriate responses.

Project Structure
-----------------

.. code-block:: text

   src/
     elpis/           # Inference MCP server
       config/        # Settings management (Pydantic)
       emotion/       # Valence-arousal state, trajectory, and regulation
       llm/           # Inference backends
         backends/    # llama-cpp and transformers implementations
         base.py      # InferenceEngine abstract base class
       server.py      # MCP server entry point

     mnemosyne/       # Memory MCP server
       core/          # Memory models and consolidator
         models.py    # Memory, MemoryType, MemoryStatus
         consolidator.py  # Clustering-based consolidation
       storage/       # ChromaDB storage backend with hybrid search
       server.py      # MCP server entry point

     psyche/          # Core server
       core/          # PsycheCore, ContextManager, MemoryHandler
       server/        # PsycheDaemon, PsycheHTTPServer
       handlers/      # ReactHandler, IdleHandler, DreamHandler
       mcp/           # ElpisClient, MnemosyneClient
       memory/        # Importance scoring, reasoning extraction

     hermes/          # TUI client
       app.py         # Main Textual application
       widgets/       # ChatView, UserInput, Sidebar, ThoughtPanel
       tools/         # ToolEngine for local tool execution
       handlers/      # RemotePsycheClient for HTTP connection
