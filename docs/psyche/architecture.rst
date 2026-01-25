============
Architecture
============

This document describes the internal architecture of the Psyche server,
including its layered design, component relationships, and data flow.

Module Overview
---------------

.. code-block:: text

    psyche/
      cli.py                # CLI entry point (psyche-server command)

      config/
        settings.py         # Pydantic settings classes
        constants.py        # Configuration constants

      core/
        server.py           # PsycheCore - central coordination
        context_manager.py  # Working memory buffer management
        memory_handler.py   # Long-term memory integration

      server/
        daemon.py           # PsycheDaemon - lifecycle management
        http.py             # PsycheHTTPServer - FastAPI/OpenAI API
        mcp.py              # MCP server implementation (optional)

      handlers/
        dream_handler.py    # Memory-based introspection

      mcp/
        client.py           # ElpisClient, MnemosyneClient

      memory/
        importance.py       # Importance scoring algorithm
        reasoning.py        # Reasoning tag extraction
        compaction.py       # Context compaction strategies
        tool_schemas.py     # Memory tool definitions

Three-Layer Architecture
------------------------

Psyche is organized into three distinct layers:

.. code-block:: text

    +------------------------------------------------------------------+
    |                    Layer 3: Server Infrastructure                 |
    |  +------------------+  +------------------+  +------------------+ |
    |  |  PsycheDaemon    |  | PsycheHTTPServer |  |  DreamHandler    | |
    |  |  - Lifecycle     |  |  - FastAPI       |  |  - Introspection | |
    |  |  - Connections   |  |  - OpenAI API    |  |  - Idle behavior | |
    |  +--------+---------+  +--------+---------+  +--------+---------+ |
    +-----------|----------------------|----------------------|---------+
                |                      |                      |
                +----------+-----------+----------------------+
                           |
                           v
    +------------------------------------------------------------------+
    |                    Layer 2: Core Coordination                     |
    |  +------------------+  +------------------+  +------------------+ |
    |  |   PsycheCore     |  | ContextManager   |  | MemoryHandler    | |
    |  |  - Message flow  |  |  - Token buffer  |  |  - Storage       | |
    |  |  - Generation    |  |  - Compaction    |  |  - Retrieval     | |
    |  +--------+---------+  +--------+---------+  +--------+---------+ |
    +-----------|----------------------|----------------------|---------+
                |                      |                      |
                +----------+-----------+----------------------+
                           |
                           v
    +------------------------------------------------------------------+
    |                    Layer 1: MCP Clients                          |
    |  +-------------------------+  +-----------------------------+    |
    |  |      ElpisClient        |  |      MnemosyneClient        |    |
    |  |  - generate()           |  |  - store_memory()           |    |
    |  |  - function_call()      |  |  - search_memories()        |    |
    |  |  - get_emotion()        |  |  - consolidate_memories()   |    |
    |  +-------------------------+  +-----------------------------+    |
    +------------------------------------------------------------------+
                |                              |
                v                              v
    +------------------+              +------------------+
    |  Elpis Server    |              | Mnemosyne Server |
    |  (MCP subprocess)|              | (MCP subprocess) |
    +------------------+              +------------------+

Layer 1: MCP Clients
--------------------

MCP clients provide the interface to backend servers (Elpis and Mnemosyne).
They handle process spawning, message transport, and tool invocation.

ElpisClient
^^^^^^^^^^^

**Module**: ``mcp/client.py``

Manages connection to the Elpis inference server:

.. code-block:: python

    class ElpisClient:
        async def generate(
            self,
            messages: List[Dict],
            max_tokens: int = 4096,
            temperature: float = 0.7,
        ) -> Tuple[str, Optional[Dict]]:
            """Generate completion with emotional state."""
            ...

        async def function_call(
            self,
            messages: List[Dict],
            tools: List[Dict],
        ) -> Optional[List[Dict]]:
            """Request tool calls from the model."""
            ...

        async def get_emotion(self) -> Dict[str, Any]:
            """Get current emotional state and trajectory."""
            ...

        async def update_emotion(self, event_type: str, intensity: float) -> Dict:
            """Trigger emotional event."""
            ...

Key features:

- Thread-safe session management with async lock
- Automatic process spawning via MCP stdio transport
- Streaming support for real-time token delivery

MnemosyneClient
^^^^^^^^^^^^^^^

**Module**: ``mcp/client.py``

Manages connection to the Mnemosyne memory server:

.. code-block:: python

    class MnemosyneClient:
        async def store_memory(
            self,
            content: str,
            summary: str,
            memory_type: str = "episodic",
            emotional_context: Optional[Dict] = None,
        ) -> str:
            """Store a new memory, returns memory ID."""
            ...

        async def search_memories(
            self,
            query: str,
            n_results: int = 10,
        ) -> List[Dict]:
            """Semantic search across memories."""
            ...

        async def consolidate_memories(
            self,
            importance_threshold: float = 0.6,
            similarity_threshold: float = 0.85,
        ) -> Dict:
            """Trigger memory consolidation."""
            ...

        async def should_consolidate(self) -> Tuple[bool, str, int, int]:
            """Check if consolidation is recommended."""
            ...

Layer 2: Core Coordination
--------------------------

The core layer handles message processing, context management, and memory integration.

PsycheCore
^^^^^^^^^^

**Module**: ``core/server.py``

Central coordination for the Psyche substrate:

.. code-block:: python

    @dataclass
    class CoreConfig:
        context: ContextConfig           # Token limits, compaction
        memory: MemoryHandlerConfig      # Storage settings
        reasoning_enabled: bool = True   # Enable <reasoning> tags
        auto_storage: bool = True        # Auto-store important content
        emotional_modulation: bool = True

    class PsycheCore:
        def __init__(
            self,
            config: CoreConfig,
            elpis_client: ElpisClient,
            mnemosyne_client: Optional[MnemosyneClient],
        ): ...

        async def process_message(self, content: str) -> AsyncIterator[str]:
            """Process user message and generate response."""
            ...

        async def generate_stream(
            self,
            messages: Optional[List[Dict]] = None,
        ) -> AsyncIterator[str]:
            """Generate response with streaming."""
            ...

PsycheCore responsibilities:

- Building system prompts with tool schemas
- Recalling relevant memories before generation
- Computing importance scores for responses
- Auto-storing high-importance content
- Extracting reasoning from responses

ContextManager
^^^^^^^^^^^^^^

**Module**: ``core/context_manager.py``

Manages the working memory buffer:

.. code-block:: python

    @dataclass
    class ContextConfig:
        max_tokens: int = 24000       # Maximum context size
        reserve_tokens: int = 4000    # Reserved for response
        min_messages_to_keep: int = 4 # Always keep recent messages
        checkpoint_enabled: bool = False

    class ContextManager:
        def add_message(self, role: str, content: str) -> None:
            """Add message to context."""
            ...

        def get_messages(self) -> List[Dict[str, str]]:
            """Get all messages in context."""
            ...

        def should_compact(self) -> bool:
            """Check if context needs compaction."""
            ...

        def compact(self, summarize_fn: Optional[Callable] = None) -> List[Dict]:
            """Compact context, optionally summarizing old messages."""
            ...

Compaction strategies:

- **Sliding Window**: Drops oldest messages (default)
- **Summarization**: Compresses old messages into summary

MemoryHandler
^^^^^^^^^^^^^

**Module**: ``core/memory_handler.py``

Integrates long-term memory via Mnemosyne:

.. code-block:: python

    class MemoryHandler:
        async def recall_relevant(
            self,
            query: str,
            n_results: int = 5,
        ) -> List[Dict]:
            """Recall memories relevant to query."""
            ...

        async def store_if_important(
            self,
            content: str,
            importance_score: float,
            emotional_context: Optional[Dict] = None,
        ) -> Optional[str]:
            """Store content if above importance threshold."""
            ...

        def format_memories_for_context(
            self,
            memories: List[Dict],
        ) -> str:
            """Format memories for inclusion in system prompt."""
            ...

Layer 3: Server Infrastructure
------------------------------

The server layer handles HTTP API, lifecycle management, and background processing.

PsycheDaemon
^^^^^^^^^^^^

**Module**: ``server/daemon.py``

Server lifecycle manager:

.. code-block:: python

    @dataclass
    class ServerConfig:
        http_host: str = "127.0.0.1"
        http_port: int = 8741
        elpis_command: str = "elpis-server"
        mnemosyne_command: Optional[str] = "mnemosyne-server"
        dream_enabled: bool = True
        dream_delay_seconds: float = 60.0
        consolidation_enabled: bool = True
        consolidation_interval: float = 300.0

    class PsycheDaemon:
        async def start(self) -> None:
            """Start daemon with all services."""
            ...

        async def stop(self) -> None:
            """Gracefully shutdown all services."""
            ...

        def register_client(self, client_id: str) -> None:
            """Track client connection."""
            ...

        def unregister_client(self, client_id: str) -> None:
            """Track client disconnection."""
            ...

Daemon responsibilities:

- Spawning Elpis and Mnemosyne as subprocesses
- Managing HTTP server lifecycle
- Tracking client connections
- Scheduling dreams when idle
- Triggering periodic consolidation

PsycheHTTPServer
^^^^^^^^^^^^^^^^

**Module**: ``server/http.py``

FastAPI-based HTTP server with OpenAI-compatible API:

.. code-block:: python

    class PsycheHTTPServer:
        def __init__(self, core: PsycheCore, config: HTTPServerConfig):
            self.app = FastAPI()
            self._setup_routes()

        async def chat_completions(self, request: ChatCompletionRequest):
            """Handle POST /v1/chat/completions."""
            ...

Endpoints:

- ``POST /v1/chat/completions`` - Chat completion (OpenAI format)
- ``GET /health`` - Health check
- ``GET /status`` - Server status

Streaming uses Server-Sent Events (SSE) for real-time token delivery.

DreamHandler
^^^^^^^^^^^^

**Module**: ``handlers/dream_handler.py``

Memory-based introspection when idle:

.. code-block:: python

    class DreamHandler:
        async def dream(self) -> Optional[str]:
            """Generate introspective content."""
            ...

        def get_dream_intention(self, quadrant: str) -> str:
            """Get intention based on emotional state."""
            ...

Dream intentions by quadrant:

+------------+--------------------------------------------+
| Quadrant   | Dream Intention                            |
+============+============================================+
| Frustrated | Seek resolution patterns from past success |
+------------+--------------------------------------------+
| Depleted   | Seek restoration (joy, meaning, connection)|
+------------+--------------------------------------------+
| Excited    | Seek exploration (curiosity, complexity)   |
+------------+--------------------------------------------+
| Calm       | Seek synthesis (patterns, integration)     |
+------------+--------------------------------------------+

Message Processing Flow
-----------------------

.. code-block:: text

    1. HTTP request arrives at PsycheHTTPServer
              |
              v
    2. PsycheCore.process_message() called
              |
              +-- Recall relevant memories (MnemosyneClient)
              |
              +-- Build context (ContextManager.add_message)
              |
              +-- Construct system prompt with memories + tools
              |
              v
    3. Generate response (ElpisClient.generate_stream)
              |
              +-- Get emotional state
              |
              +-- Stream tokens back
              |
              v
    4. Process response
              |
              +-- Extract tool calls
              |
              +-- Parse reasoning tags
              |
              +-- Compute importance score
              |
              v
    5. Auto-store if important (MnemosyneClient.store_memory)
              |
              v
    6. Stream response to client via SSE

Importance Scoring
------------------

**Module**: ``memory/importance.py``

Messages are scored for automatic storage:

.. code-block:: python

    def calculate_importance(
        content: str,
        emotional_context: Optional[Dict] = None,
        recency_weight: float = 0.3,
        relevance_weight: float = 0.4,
        emotional_weight: float = 0.3,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate importance score (0.0 to 1.0).

        Returns:
            Tuple of (score, breakdown_dict)
        """

Factors:

- **Content quality**: Length, complexity, information density
- **Emotional salience**: Higher arousal increases importance
- **Recency**: Recent content weighted higher
- **Uniqueness**: Novel content scores higher

Threshold (default 0.7) determines auto-storage.

Configuration Hierarchy
-----------------------

.. code-block:: text

    ServerConfig (daemon.py)
      |
      +-- HTTPServerConfig
      |     +-- host, port
      |     +-- cors settings
      |
      +-- CoreConfig (core/server.py)
            |
            +-- ContextConfig
            |     +-- max_tokens
            |     +-- reserve_tokens
            |     +-- compaction settings
            |
            +-- MemoryHandlerConfig
                  +-- tool schemas
                  +-- retrieval settings

All configuration uses Pydantic with environment variable support:

.. code-block:: bash

    PSYCHE_SERVER__HTTP_HOST=0.0.0.0
    PSYCHE_SERVER__HTTP_PORT=8741
    PSYCHE_CORE__CONTEXT__MAX_TOKENS=32000
    PSYCHE_CORE__AUTO_STORAGE_THRESHOLD=0.7

Initialization Sequence
-----------------------

.. code-block:: text

    1. CLI parses arguments (cli.py)
              |
              v
    2. ServerConfig loaded
              |
              v
    3. PsycheDaemon created
              |
              v
    4. daemon.start() called
              |
              +-- Spawn Elpis subprocess
              |
              +-- Spawn Mnemosyne subprocess (optional)
              |
              +-- Create ElpisClient, MnemosyneClient
              |
              +-- Create PsycheCore with clients
              |
              +-- Create PsycheHTTPServer
              |
              +-- Start Uvicorn HTTP server
              |
              +-- Start dream scheduler (if enabled)
              |
              v
    5. Server running, accepting connections

Error Handling
--------------

Errors are handled at each layer:

**MCP Client Layer**
    Connection failures, timeout handling, reconnection

**Core Layer**
    Generation errors, memory failures, context overflow

**Server Layer**
    HTTP errors, request validation, graceful degradation

When Mnemosyne is unavailable, Psyche continues with memory features disabled.

See Also
--------

- :doc:`features` - Feature documentation
- :doc:`tools` - Tool system details
- :doc:`api/index` - API reference
- :doc:`/mnemosyne/index` - Memory server documentation
- :doc:`/elpis/index` - Inference server documentation
