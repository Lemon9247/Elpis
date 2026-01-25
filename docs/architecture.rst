============
Architecture
============

This document provides a comprehensive overview of the Elpis system architecture,
explaining how the components interact, data flows through the system, and the key
design decisions that shape the implementation.

System Overview
---------------

Elpis is a modular system for giving AI persistent memory and emotional state.
It uses a server-client architecture with the Model Context Protocol (MCP) for
internal component communication.

.. code-block:: text

    +-------------------------------------------------------------+
    |                      PSYCHE SERVER                          |
    |  +---------------+  +---------------+  +---------------+    |
    |  |  PsycheCore   |  | ElpisClient   |  | MnemosyneCli  |    |
    |  |               |  | (MCP stdio)   |  | (MCP stdio)   |    |
    |  | - Context     |  |               |  |               |    |
    |  | - Memory      |  | - Inference   |  | - Storage     |    |
    |  | - Dreams      |  | - Emotion     |  | - Recall      |    |
    |  +---------------+  +---------------+  +---------------+    |
    +---------------------------|----------------------------------+
                                | HTTP API (OpenAI-compatible)
             +------------------+------------------+
             |                                     |
      +------+------+                       +------+------+
      |   Hermes    |                       |  External   |
      |    (TUI)    |                       |  Clients    |
      |             |                       |             |
      | - Chat view |                       | - Any HTTP  |
      | - Tools     |                       |   client    |
      +-------------+                       +-------------+

The Four Components
-------------------

Elpis
^^^^^

**Purpose**: LLM inference with emotional modulation

Elpis is an MCP server that provides text generation with built-in emotional state
tracking. It supports two modulation approaches:

- **Sampling parameter modulation** (llama-cpp): Adjusts temperature and top_p based
  on emotional state
- **Steering vector injection** (transformers): Directly modifies model activations
  for more nuanced emotional expression

Key features:

- Valence-arousal emotional model with trajectory tracking
- Homeostatic regulation with decay toward baseline
- Event-based emotional shifts (success, failure, frustration, etc.)
- Multiple inference backends (llama-cpp for GGUF, transformers for HuggingFace)

See :doc:`elpis/index` for detailed documentation.

Mnemosyne
^^^^^^^^^

**Purpose**: Semantic memory storage and retrieval

Mnemosyne is an MCP server providing intelligent memory storage with ChromaDB
as the vector backend. It implements biologically-inspired memory consolidation:

- **Short-term memory**: Recent memories awaiting consolidation
- **Long-term memory**: Important memories promoted after clustering
- **Hybrid search**: Combines vector similarity with BM25 keyword matching
- **Automatic consolidation**: Clustering-based promotion with importance scoring

Key features:

- Four memory types: episodic, semantic, procedural, emotional
- Emotional context attached to memories (valence, arousal, quadrant)
- Importance scoring based on salience, recency, and access frequency
- Lineage tracking for consolidated memories

See :doc:`mnemosyne/index` for detailed documentation.

Psyche
^^^^^^

**Purpose**: Core coordination server

Psyche is the central coordinator that brings everything together. It runs as an
HTTP server with an OpenAI-compatible API, spawning Elpis and Mnemosyne as MCP
subprocesses.

Architecture layers:

- **PsycheCore**: Central coordination for context, memory, and emotional state
- **PsycheDaemon**: Server lifecycle management and MCP process spawning
- **PsycheHTTPServer**: FastAPI server with ``/v1/chat/completions`` endpoint
- **Handlers**: Specialized components for different operational modes
  - **ReactHandler**: ReAct (Reasoning + Acting) loop for user input
  - **IdleHandler**: Background processing during client silence
  - **DreamHandler**: Memory-based introspection when no clients connected

Key features:

- Context management with automatic compaction
- Memory retrieval before response generation
- Importance scoring for automatic memory storage
- Dreaming (memory consolidation and introspection)

See :doc:`psyche/index` for detailed documentation.

Hermes
^^^^^^

**Purpose**: Terminal user interface

Hermes is the TUI client that connects to a Psyche server. It executes
file/bash/search tools locally while memory tools execute server-side.

Components:

- **Textual application**: Rich terminal UI with widgets
- **RemotePsycheClient**: HTTP client for server communication
- **ToolEngine**: Local execution of file, bash, and search operations

Key features:

- Streaming response display with markdown rendering
- Emotional state visualization in sidebar
- Tool activity panel showing execution details
- Slash commands for control (``/help``, ``/status``, ``/emotion``, etc.)

See :doc:`hermes/index` for detailed documentation.

Communication Patterns
----------------------

MCP Protocol (Elpis, Mnemosyne)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Elpis and Mnemosyne run as stdio-based MCP servers spawned by Psyche. The MCP
protocol provides:

- **Tools**: Named functions that can be called with arguments
- **Resources**: Read-only data endpoints (e.g., ``emotion://state``)
- **JSON-RPC**: Message format over stdio

Example tool call flow:

.. code-block:: text

    PsycheCore                    ElpisClient                    Elpis Server
        |                              |                              |
        |--- generate(messages) ------>|                              |
        |                              |--- MCP tool call (stdio) --->|
        |                              |                              |
        |                              |<-- JSON-RPC response --------|
        |<-- (response, emotion) ------|                              |

HTTP API (Psyche)
^^^^^^^^^^^^^^^^^

Psyche exposes an OpenAI-compatible HTTP API for external clients:

- **Endpoint**: ``POST /v1/chat/completions``
- **Streaming**: SSE (Server-Sent Events) for real-time token delivery
- **Format**: OpenAI chat completion request/response format

.. code-block:: bash

    curl http://localhost:8741/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": true
      }'

Tool Execution Model
--------------------

Elpis separates tool execution between server and client:

Server-Side Tools (Memory)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Memory tools execute on the Psyche server because memory is part of the AI's
"self". These are not returned to clients for execution:

- ``recall_memory``: Search for relevant memories
- ``store_memory``: Store a new memory
- ``consolidate_memories``: Trigger memory consolidation

Client-Side Tools (File/Bash/Search)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File and bash tools are returned to clients (Hermes) for local execution:

- ``read_file``: Read file contents
- ``create_file``: Create a new file
- ``edit_file``: Modify an existing file
- ``execute_bash``: Run shell commands
- ``search_codebase``: Full-text code search
- ``list_directory``: Directory listing

This separation ensures:

- Memory operations are consistent across all clients
- File operations execute in the client's local context
- Security controls can be applied at the client level

Data Flow Example
-----------------

Here's how a user message flows through the system:

.. code-block:: text

    1. User types message in Hermes TUI
                |
                v
    2. Hermes sends HTTP POST to Psyche /v1/chat/completions
                |
                v
    3. PsycheCore receives message
       |
       +-- Recalls relevant memories (MnemosyneClient.search_memories)
       |
       +-- Builds context (ContextManager.add_message)
       |
       +-- Constructs system prompt with tools
       |
       +-- Gets completion (ElpisClient.generate)
           |
           +-- Elpis gets current emotional state
           |
           +-- Computes modulated parameters or steering coefficients
           |
           +-- Runs LLM inference
           |
           +-- Returns response + emotional state
                |
                v
    4. PsycheCore processes response
       |
       +-- Extracts tool calls (if any)
       |
       +-- Memory tools execute server-side
       |
       +-- File/bash tools returned to client
       |
       +-- Computes importance score
       |
       +-- Auto-stores high-importance content
                |
                v
    5. Response streamed back to Hermes via SSE
                |
                v
    6. Hermes displays response
       |
       +-- Executes any file/bash tools locally
       |
       +-- Sends tool results back to server (if applicable)
       |
       +-- Updates emotional state display

Emotional State Flow
--------------------

Emotional state flows through the system as follows:

.. code-block:: text

    Event (success, failure, etc.)
                |
                v
    HomeostasisRegulator.process_event()
       |
       +-- Look up event deltas (e.g., success: +0.2 valence, +0.1 arousal)
       |
       +-- Apply intensity multiplier
       |
       +-- Clamp deltas to max_delta
       |
       +-- Apply decay toward baseline
       |
       +-- Shift EmotionalState
                |
                v
    EmotionalState records to history
                |
                v
    Trajectory computed on next query
       |
       +-- Velocity (linear regression over recent history)
       |
       +-- Trend (improving/declining/stable/oscillating)
       |
       +-- Spiral detection (sustained movement from baseline)
                |
                v
    State returned with response metadata

Memory Consolidation Flow
-------------------------

Memory consolidation follows a biologically-inspired process:

.. code-block:: text

    1. Memories stored in short-term collection
                |
                v
    2. Psyche triggers should_consolidate() during idle
       |
       +-- Checks buffer size vs threshold
       |
       +-- Returns recommendation
                |
                v
    3. If recommended, calls consolidate_memories()
                |
                v
    4. MemoryConsolidator runs clustering
       |
       +-- Selects candidates (age >= min_age_hours)
       |
       +-- Recomputes importance scores
       |
       +-- Retrieves embeddings from ChromaDB
       |
       +-- Greedy clustering by cosine similarity
                |
                v
    5. For each cluster meeting importance threshold:
       |
       +-- Selects highest-importance memory as representative
       |
       +-- Records source_memory_ids for lineage
       |
       +-- Promotes representative to long-term collection
       |
       +-- Deletes other cluster members

Dreaming
--------

When no clients are connected, Psyche enters a dreaming state:

.. code-block:: text

    1. No clients connected for interval period
                |
                v
    2. DreamHandler activates
       |
       +-- Selects dream intention based on emotional quadrant:
           - Frustrated: Seek resolution patterns
           - Depleted: Seek restoration (joy, meaning)
           - Excited: Seek exploration (curiosity)
           - Calm: Seek synthesis (patterns, integration)
                |
                v
    3. Retrieves random memories matching intention
                |
                v
    4. Generates introspective response
                |
                v
    5. Potentially stores insights as new semantic memories
                |
                v
    6. Triggers consolidation if buffer is full

Key Architectural Decisions
---------------------------

MCP for Internal Communication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using MCP for Elpis and Mnemosyne provides:

- **Decoupling**: Services can be upgraded independently
- **Isolation**: Resource management per process
- **Testability**: Components can be tested in isolation
- **Flexibility**: Could be replaced with different implementations

HTTP for External API
^^^^^^^^^^^^^^^^^^^^^

Using HTTP with OpenAI compatibility provides:

- **Ecosystem compatibility**: Works with existing tools expecting OpenAI API
- **Multiple clients**: Any HTTP client can connect
- **Streaming**: SSE support for real-time responses
- **Remote access**: Server can run on different machine than client

Two-Level Tool Execution
^^^^^^^^^^^^^^^^^^^^^^^^

Separating server-side (memory) from client-side (file/bash) tools:

- **Consistency**: Memory state is shared across all clients
- **Security**: File operations execute in controlled client context
- **Latency**: Local operations don't require network round-trip
- **Flexibility**: Different clients can have different tool capabilities

Emotional Trajectory Tracking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Beyond raw valence-arousal, tracking trajectory enables:

- **Early warning**: Detect emotional spirals before they intensify
- **Personality modeling**: Different baselines and decay rates
- **Adaptive behavior**: Responses can consider momentum and trend
- **Debugging**: Understand how emotional state evolved

Memory Consolidation
^^^^^^^^^^^^^^^^^^^^

Clustering-based consolidation provides:

- **Compression**: Similar memories merge into representatives
- **Quality**: Importance threshold filters noise
- **Lineage**: source_memory_ids maintain traceability
- **Gradual**: Short-term buffer allows reinforcement before consolidation

Configuration Overview
----------------------

Each component has its own configuration namespace:

- **Elpis**: ``ELPIS_MODEL_*``, ``ELPIS_EMOTION_*``
- **Mnemosyne**: ``MNEMOSYNE_STORAGE_*``, ``MNEMOSYNE_RETRIEVAL_*``
- **Psyche**: ``PSYCHE_SERVER_*``, ``PSYCHE_CONTEXT_*``
- **Hermes**: ``HERMES_*``

All use Pydantic settings with environment variable support. See individual
component documentation for detailed configuration options.

See Also
--------

- :doc:`elpis/index` - Inference server documentation
- :doc:`mnemosyne/index` - Memory server documentation
- :doc:`psyche/index` - Core server documentation
- :doc:`hermes/index` - TUI client documentation
- :doc:`getting-started/quickstart` - Quick start guide
