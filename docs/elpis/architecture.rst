============
Architecture
============

This document describes the internal architecture of the Elpis inference server,
including its module structure, component relationships, and data flow.

Module Overview
---------------

.. code-block:: text

    elpis/
      cli.py              # CLI entry point (elpis-server command)
      server.py           # MCP server implementation and tool handlers

      config/
        settings.py       # Pydantic settings (ModelSettings, EmotionSettings)

      emotion/
        state.py          # EmotionalState, EmotionalTrajectory, TrajectoryConfig
        regulation.py     # HomeostasisRegulator, EventHistory, event mappings
        behavioral_monitor.py  # BehavioralMonitor for usage pattern detection
        sentiment.py      # SentimentAnalyzer for LLM-based emotion inference

      llm/
        base.py           # InferenceEngine abstract base class
        prompts.py        # Prompt templates and formatting
        backends/
          __init__.py     # Backend registry and factory
          llama_cpp/
            config.py     # LlamaCppConfig
            inference.py  # LlamaInference implementation
          transformers/
            config.py     # TransformersConfig
            inference.py  # TransformersInference implementation
            steering.py   # SteeringManager for vector injection

      utils/
        exceptions.py     # Custom exception classes
        hardware.py       # Hardware detection (CUDA, ROCm, CPU)
        logging.py        # Logging configuration

Component Architecture
----------------------

.. code-block:: text

    +------------------------------------------------------------------+
    |                        MCP Server (server.py)                     |
    |  +------------------+  +------------------+  +------------------+ |
    |  |   Tool Handlers  |  |  Stream Manager  |  |  ServerContext   | |
    |  |  - generate      |  |  - StreamState   |  |  - llm           | |
    |  |  - function_call |  |  - active_streams|  |  - emotion_state | |
    |  |  - get_emotion   |  |  - TTL cleanup   |  |  - regulator     | |
    |  +--------+---------+  +--------+---------+  |  - behavioral_mon| |
    |           |                      |          |  - sentiment_anlz | |
    +-----------|----------------------|----------+--------+---------+-+
                |                      |                   |
                v                      v                   v
    +------------------+    +------------------+    +------------------+
    | InferenceEngine  |    | EmotionalState   |    | HomeostasisReg.  |
    | (llm/base.py)    |    | (emotion/state)  |    | (emotion/reg.)   |
    +--------+---------+    +------------------+    +--------+---------+
             |                                              |
    +--------+--------+                           +---------+---------+
    |                 |                           |                   |
    v                 v                           v                   v
    +----------+  +---------------+    +------------------+  +------------------+
    | LlamaCpp |  | Transformers  |    | BehavioralMonitor|  | SentimentAnalyzer|
    | Backend  |  | Backend       |    | (behavioral_mon.)|  | (sentiment.py)   |
    +----------+  +-------+-------+    +------------------+  +------------------+
                          |
                          v
                  +---------------+
                  | SteeringMgr   |
                  | (steering.py) |
                  +---------------+

Server Context
--------------

The :class:`ServerContext` dataclass provides dependency injection for all server
components:

.. code-block:: python

    @dataclass
    class ServerContext:
        llm: InferenceEngine           # Inference backend
        emotion_state: EmotionalState  # Current emotional state
        regulator: HomeostasisRegulator # Event processing
        settings: Settings             # Configuration
        active_streams: Dict[str, StreamState]  # Stream tracking
        behavioral_monitor: Optional[BehavioralMonitor]  # Pattern detection
        sentiment_analyzer: Optional[SentimentAnalyzer]  # Deep emotion inference

This pattern avoids global mutable state and makes testing easier.

The ``behavioral_monitor`` and ``sentiment_analyzer`` are optional and created
based on configuration settings.

Inference Backend System
------------------------

Backend Registry
^^^^^^^^^^^^^^^^

The backend system uses a plugin-style registry pattern in ``llm/backends/__init__.py``:

.. code-block:: python

    # Backend registration at import time
    _BACKENDS: Dict[str, Type[InferenceEngine]] = {}

    def register_backend(name: str, cls: Type[InferenceEngine]) -> None:
        _BACKENDS[name] = cls

    def create_backend(settings: ModelSettings) -> InferenceEngine:
        backend_cls = _BACKENDS[settings.backend]
        config = _create_config(settings)
        return backend_cls(config)

This allows:

- Runtime backend selection based on configuration
- Graceful handling of missing dependencies
- Easy addition of new backends

InferenceEngine Interface
^^^^^^^^^^^^^^^^^^^^^^^^^

All backends implement the :class:`InferenceEngine` abstract base class:

.. code-block:: python

    class InferenceEngine(ABC):
        # Capability flags
        SUPPORTS_STEERING: bool = False
        MODULATION_TYPE: Literal["none", "sampling", "steering"] = "none"

        @abstractmethod
        async def chat_completion(self, messages, ...) -> str: ...

        @abstractmethod
        async def chat_completion_stream(self, messages, ...) -> AsyncIterator[str]: ...

        @abstractmethod
        async def function_call(self, messages, tools, ...) -> Optional[List[Dict]]: ...

The capability flags allow the server to adapt behavior based on backend features.

llama-cpp Backend
^^^^^^^^^^^^^^^^^

**Module**: ``llm/backends/llama_cpp/inference.py``

The llama-cpp backend wraps ``llama-cpp-python`` for GGUF model inference:

- **Hardware Detection**: Auto-detects CUDA, ROCm, or CPU
- **Thread Safety**: Disables CUDA graphs, forces single-threaded OpenMP
- **Chat Formatting**: Uses model-specific chat templates
- **Modulation**: Adjusts temperature and top_p based on emotional state

.. code-block:: text

    LlamaInference
      |
      +-- _load_model()      # Load GGUF with hardware config
      +-- chat_completion()  # Non-streaming generation
      +-- chat_completion_stream()  # Token-by-token streaming
      +-- function_call()    # Tool call generation

Transformers Backend
^^^^^^^^^^^^^^^^^^^^

**Module**: ``llm/backends/transformers/inference.py``

The transformers backend uses HuggingFace Transformers with steering vector support:

- **Model Loading**: HuggingFace model IDs or local paths
- **Precision Control**: auto, float16, bfloat16, float32
- **Steering Vectors**: Activation-level emotional modulation
- **Hook-based Injection**: Uses PyTorch forward hooks

.. code-block:: text

    TransformersInference
      |
      +-- SteeringManager
      |     +-- load_vectors()        # Load .pt files
      |     +-- compute_blended_vector()  # Bilinear interpolation
      |     +-- apply_hook()          # Register forward hook
      |     +-- remove_hook()         # Clean up hook
      |
      +-- chat_completion()
      +-- chat_completion_stream()
      +-- function_call()

Steering Vector Flow
""""""""""""""""""""

1. Emotional state provides blend coefficients
2. SteeringManager computes blended vector from 4 emotion vectors
3. Forward hook registered at target layer
4. During generation, hook adds blended vector to activations
5. Hook removed after generation completes

Emotional State System
----------------------

EmotionalState
^^^^^^^^^^^^^^

**Module**: ``emotion/state.py``

The :class:`EmotionalState` dataclass represents the 2D valence-arousal model:

.. code-block:: python

    @dataclass
    class EmotionalState:
        valence: float = 0.0  # -1.0 to +1.0
        arousal: float = 0.0  # -1.0 to +1.0

        # Homeostasis target
        baseline_valence: float = 0.0
        baseline_arousal: float = 0.0

        # Trajectory tracking
        _history: List[Tuple[datetime, float, float]]
        _trajectory_config: TrajectoryConfig

Key methods:

- ``get_quadrant()`` - Returns "excited", "frustrated", "calm", or "depleted"
- ``get_modulated_params()`` - Returns temperature/top_p for sampling modulation
- ``get_steering_coefficients()`` - Returns blend weights for steering vectors
- ``get_trajectory()`` - Computes velocity, trend, spiral detection
- ``shift()`` - Apply delta and record to history

Trajectory Tracking
^^^^^^^^^^^^^^^^^^^

The :class:`EmotionalTrajectory` tracks patterns over time:

.. code-block:: python

    @dataclass
    class EmotionalTrajectory:
        valence_velocity: float  # Rate of change per minute
        arousal_velocity: float
        trend: str              # "improving", "declining", "stable", "oscillating"
        spiral_detected: bool   # Sustained movement from baseline
        spiral_direction: str   # "positive", "negative", "escalating", "withdrawing"
        time_in_current_quadrant: float
        momentum: str           # "positive", "negative", "neutral"

Trajectory is computed from the history buffer using:

- Linear regression for velocity calculation
- Pattern matching for trend detection
- Distance tracking for spiral detection

HomeostasisRegulator
^^^^^^^^^^^^^^^^^^^^

**Module**: ``emotion/regulation.py``

The :class:`HomeostasisRegulator` handles event processing, decay, and dynamic
emotional behavior:

.. code-block:: python

    class HomeostasisRegulator:
        def __init__(self, state, decay_rate=0.1, max_delta=0.5,
                     streak_compounding_enabled=True,
                     mood_inertia_enabled=True, ...):
            self.event_history = EventHistory()  # Context-aware intensity
            self.decay_multipliers = {...}       # Per-quadrant decay

        def process_event(self, event_type: str, intensity: float = 1.0) -> None:
            # Apply time-based decay (with quadrant multiplier)
            self._apply_decay()

            # Look up event deltas
            valence_delta, arousal_delta = EVENT_MAPPINGS[event_type]

            # Apply context-aware intensity from event history
            if self.streak_compounding_enabled:
                intensity *= self.event_history.get_intensity_modifier(event_type)

            # Apply mood inertia resistance
            if self.mood_inertia_enabled:
                intensity *= self._get_inertia_modifier(valence_delta)

            # Shift emotional state
            self.state.shift(valence_delta * intensity, arousal_delta * intensity)

Key features:

- **Event History**: Tracks recent events for compounding/dampening
- **Mood Inertia**: Resists rapid emotional swings based on trajectory
- **Quadrant Decay**: Per-quadrant decay rate multipliers
- **Multi-factor Response Analysis**: Weighted keyword scoring

Event mappings define the emotional impact of each event type:

.. code-block:: python

    EVENT_MAPPINGS = {
        "success": (0.2, 0.1),       # Positive valence, slight arousal
        "failure": (-0.2, 0.15),     # Negative valence, arousal increase
        "frustration": (-0.15, 0.25), # Negative valence, high arousal
        "novelty": (0.1, 0.2),       # Slight positive, arousal increase
        "idle": (0.0, -0.1),         # Arousal decrease only
        # ... more events
    }

BehavioralMonitor
^^^^^^^^^^^^^^^^^

**Module**: ``emotion/behavioral_monitor.py``

The :class:`BehavioralMonitor` detects patterns in tool usage and generation:

.. code-block:: python

    class BehavioralMonitor:
        def __init__(self, on_event, retry_loop_threshold=3, ...):
            self.on_event = on_event  # Callback to trigger emotional events
            self.tool_history = []     # Recent tool calls

        def record_tool_call(self, tool_name, success, duration=None):
            # Check for retry loops, failure streaks
            self._check_retry_loop()
            self._check_failure_streak()

        def start_generation(self):
            # Mark generation start time

        def end_generation(self):
            # Check for long generation, trigger if needed

Detected patterns:

- **Retry loops**: Same tool called N+ times -> frustration event
- **Failure streaks**: N+ consecutive failures -> compounding error events
- **Long generations**: >30s -> mild blocked event
- **Idle periods**: >2min -> calming idle event

SentimentAnalyzer
^^^^^^^^^^^^^^^^^

**Module**: ``emotion/sentiment.py``

The :class:`SentimentAnalyzer` provides deep emotional inference:

.. code-block:: python

    class SentimentAnalyzer:
        def __init__(self, use_local_model=True, min_length=200):
            self._local_model = None  # Lazy-loaded DistilBERT

        def analyze(self, content: str) -> Optional[SentimentResult]:
            # Skip short content
            # Try local model or LLM analysis
            # Return sentiment score and emotions

        def get_emotional_event(self, result) -> Optional[Tuple[str, float]]:
            # Map sentiment to event type and intensity

Supports two modes:

- **Local model**: Fast DistilBERT-based classification
- **LLM self-analysis**: Uses inference engine for deeper understanding

MCP Server Layer
----------------

Tool Registration
^^^^^^^^^^^^^^^^^

Tools are registered using MCP decorators:

.. code-block:: python

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return [
            Tool(name="generate", description="...", inputSchema={...}),
            Tool(name="get_emotion", description="...", inputSchema={...}),
            # ...
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict) -> List[TextContent]:
        if name == "generate":
            return await handle_generate(arguments)
        # ...

Streaming Architecture
^^^^^^^^^^^^^^^^^^^^^^

Streaming uses a poll-based pattern for MCP compatibility:

.. code-block:: text

    Client                    Server
      |                          |
      |-- generate_stream_start ->|
      |<-- stream_id -------------|
      |                          |
      |-- generate_stream_read -->|  (polling)
      |<-- tokens[] --------------|
      |                          |
      |-- generate_stream_read -->|
      |<-- tokens[], complete ----|
      |                          |

Stream state is tracked per stream ID with TTL enforcement:

.. code-block:: python

    @dataclass
    class StreamState:
        buffer: List[str]        # Accumulated tokens
        cursor: int              # Last read position
        is_complete: bool        # Generation finished
        error: Optional[str]     # Error message if failed
        task: Optional[asyncio.Task]  # Producer task reference
        created_at: float        # For TTL enforcement

Resource Exposure
^^^^^^^^^^^^^^^^^

MCP resources provide read-only data access:

- ``emotion://state`` - Current emotional state as JSON
- ``emotion://events`` - Available event types and their effects

Configuration Flow
------------------

Configuration flows through Pydantic settings:

.. code-block:: text

    Environment Variables / .env / TOML
              |
              v
    +------------------+
    |     Settings     |
    |  +------------+  |
    |  |ModelSettings| --> Backend selection, model path, etc.
    |  +------------+  |
    |  +------------+  |
    |  |EmotionSettings| --> Baseline, decay, thresholds
    |  +------------+  |
    +------------------+
              |
              v
    +------------------+
    |  ServerContext   |  <-- Initialized at startup
    +------------------+

Initialization Sequence
-----------------------

.. code-block:: text

    1. CLI parses arguments (cli.py)
              |
              v
    2. Settings loaded from env/files (config/settings.py)
              |
              v
    3. Backend created via factory (llm/backends/__init__.py)
              |
              v
    4. EmotionalState initialized with baseline
              |
              v
    5. HomeostasisRegulator wraps state
              |
              v
    6. ServerContext assembled
              |
              v
    7. MCP server starts stdio loop (server.py)
              |
              v
    8. Tools registered and ready

Error Handling
--------------

Errors are handled at multiple levels:

**Backend Level**
    Inference errors are caught and returned as error responses

**Server Level**
    Tool errors return MCP error content with details

**Stream Level**
    Stream errors are stored in StreamState and returned on next read

Custom exceptions in ``utils/exceptions.py``:

- ``ElpisError`` - Base exception
- ``ConfigurationError`` - Invalid configuration
- ``InferenceError`` - Backend inference failure
- ``EmotionError`` - Emotional state errors

See Also
--------

- :doc:`emotion-system` - Emotional model details
- :doc:`backends` - Backend comparison and usage
- :doc:`configuration` - Configuration reference
- :doc:`api/index` - API reference
