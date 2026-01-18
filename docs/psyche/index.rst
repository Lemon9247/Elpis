======
Psyche
======

Psyche is the core library and HTTP server for the Elpis system. It coordinates
memory handling, context management, tool execution, and provides both local
and remote operation modes.

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

- **Tool System**: File operations, bash execution, and codebase search with
  safety controls and approval workflows.

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
    - ``PsycheClient``: Abstract client interface (local and remote implementations)

**Server** (``psyche.server``)
    HTTP server infrastructure for remote operation:

    - ``PsycheDaemon``: Server lifecycle and MCP client management
    - ``PsycheHTTPServer``: FastAPI server with OpenAI-compatible API

**Tools** (``psyche.tools``)
    Async tool execution with safety controls:

    - ``ToolEngine``: Orchestrates tool execution with approval workflow
    - Implementations: file ops, bash, directory listing, codebase search

**MCP Clients** (``psyche.mcp``)
    Clients for connecting to backend MCP servers:

    - ``ElpisClient``: Text generation with streaming and emotional modulation
    - ``MnemosyneClient``: Memory storage, search, and consolidation

Local vs Remote Mode
--------------------

**Local Mode** (via Hermes):

.. code-block:: text

    Hermes
      └── PsycheCore
            ├── ReactHandler (processes user input)
            ├── IdleHandler (background processing)
            ├── ElpisClient (MCP subprocess)
            └── MnemosyneClient (MCP subprocess)

In local mode, Hermes instantiates PsycheCore directly and manages the full
lifecycle including MCP server subprocesses.

**Server Mode** (via psyche-server):

.. code-block:: text

    PsycheDaemon
      ├── PsycheCore
      │     ├── ElpisClient (MCP subprocess)
      │     └── MnemosyneClient (MCP subprocess)
      ├── PsycheHTTPServer (/v1/chat/completions)
      └── DreamHandler (when no clients)

In server mode, PsycheDaemon manages the HTTP server and dreams when idle.
Memory tools execute server-side; file/bash tools return to the client.

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

Tool Safety
    The tool system provides:

    - Pydantic validation for all tool inputs
    - Path sanitization to prevent workspace escapes
    - Configurable approval workflow (auto/ask/deny)
    - Timeout controls for bash execution

Quick Start
-----------

**Server Mode:**

.. code-block:: bash

   # Start the server
   psyche-server

   # With custom port
   psyche-server --port 8080

   # With debug logging
   psyche-server --debug

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
