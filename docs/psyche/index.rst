======
Psyche
======

Psyche is the core library and HTTP server for the Elpis system. It coordinates
memory handling, context management, and tool execution, providing an
OpenAI-compatible API for remote access.

Overview
--------

Psyche serves as the "mind" of the Elpis ecosystem, providing:

- **PsycheCore**: Memory coordination layer managing context, memory retrieval,
  and importance scoring for automatic storage.

- **Handlers**: Specialized components for different operational modes:

  - **ReactHandler**: Processes user input with ReAct (Reasoning + Acting) loops
  - **IdleHandler**: Background processing during silence, triggers consolidation
  - **DreamHandler**: Memory-based introspection when no clients connected

- **HTTP Server**: OpenAI-compatible API for remote access via ``psyche-server``

- **Memory Tools**: Server-side recall, storage, and consolidation operations.
  File/bash/search tools execute on the client (Hermes).

Architecture
------------

Psyche is organized into these key components:

**Core** (``psyche.core``)
    The central coordination layer:

    - ``PsycheCore``: Context management, memory handling, Elpis inference
    - ``ContextManager``: Working memory buffer with token tracking
    - ``MemoryHandler``: Long-term storage via Mnemosyne, retrieval, compaction

**Handlers** (``psyche.handlers``)
    Operational modes for different contexts:

    - ``ReactHandler``: ReAct loop for user input processing
    - ``IdleHandler``: Background processing during client silence
    - ``DreamHandler``: Server-side dreaming when no clients connected
    - ``RemotePsycheClient``: HTTP client for connecting to Psyche server

**Server** (``psyche.server``)
    HTTP server infrastructure:

    - ``PsycheDaemon``: Server lifecycle and MCP client management
    - ``PsycheHTTPServer``: FastAPI server with OpenAI-compatible API

**Memory** (``psyche.memory``)
    Memory integration components:

    - ``MemoryHandler``: Long-term storage, retrieval, and compaction
    - ``tool_schemas``: Memory tool definitions for server-side execution

**MCP Clients** (``psyche.mcp``)
    Clients for connecting to backend MCP servers:

    - ``ElpisClient``: Text generation with streaming and emotional modulation
    - ``MnemosyneClient``: Memory storage, search, and consolidation

Server Architecture
-------------------

Psyche runs as a persistent server via ``psyche-server``:

.. code-block:: text

    PsycheDaemon
      ├── PsycheCore
      │     ├── ElpisClient (MCP subprocess for inference)
      │     └── MnemosyneClient (MCP subprocess for memory)
      ├── PsycheHTTPServer (/v1/chat/completions)
      └── DreamHandler (when no clients)

The server manages MCP subprocesses for inference and memory, provides an
OpenAI-compatible HTTP API, and dreams when idle. Memory tools execute
server-side while file/bash/search tools are returned to clients for local
execution.

Key Features
------------

ReAct Loop
    The ReactHandler implements a Reasoning + Acting loop allowing the LLM to:

    1. Think about the user's request
    2. Optionally call tools to gather information
    3. Continue reasoning with tool results
    4. Repeat until a final response is ready

Memory Integration
    PsycheCore automatically:

    - Retrieves relevant memories before generating responses
    - Scores message importance for automatic storage
    - Compacts context when approaching token limits
    - Stores compacted messages to long-term memory

Dreaming
    When the server has no connected clients, DreamHandler:

    - Retrieves random memories for context
    - Generates introspective content
    - Potentially stores insights as new memories
    - Runs on a configurable interval (default: 5 minutes)

Tool Safety (via Hermes)
    The client-side tool system (``hermes.tools``) provides:

    - Pydantic validation for all tool inputs
    - Path sanitization to prevent workspace escapes
    - Configurable approval workflow (auto/ask/deny)
    - Timeout controls for bash execution

Quick Start
-----------

Start the Psyche server:

.. code-block:: bash

   # Start the server
   psyche-server

   # With custom port
   psyche-server --port 8080

   # With debug logging
   psyche-server --debug

Then connect with Hermes (see :doc:`../hermes/index`).

**Programmatic Usage:**

.. code-block:: python

   from psyche.core import PsycheCore, CoreConfig
   from psyche.mcp.client import ElpisClient, MnemosyneClient

   async def main():
       async with ElpisClient("elpis-server") as elpis:
           async with MnemosyneClient("mnemosyne-server") as mnemosyne:
               core = PsycheCore(
                   config=CoreConfig(),
                   elpis_client=elpis,
                   mnemosyne_client=mnemosyne,
               )

               # Add a message and generate response
               core.add_message("user", "Hello!")
               async for token in core.generate_stream():
                   print(token, end="", flush=True)

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   features
   tools

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
