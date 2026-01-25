Configuration
=============

Elpis uses Pydantic settings with environment variable support for configuration.
All settings can be configured via environment variables, a ``.env`` file, or
programmatically through Python.

Environment Variables
---------------------

Model Settings
^^^^^^^^^^^^^^

These settings control the LLM backend and model loading.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``ELPIS_MODEL_BACKEND``
     - ``llama-cpp``
     - Inference backend: ``llama-cpp`` or ``transformers``
   * - ``ELPIS_MODEL_PATH``
     - ``./data/models/...``
     - Path to GGUF file or HuggingFace model ID
   * - ``ELPIS_MODEL_CONTEXT_LENGTH``
     - ``32768``
     - Context window size (512-131072)
   * - ``ELPIS_MODEL_GPU_LAYERS``
     - ``35``
     - Layers to offload to GPU (llama-cpp)
   * - ``ELPIS_MODEL_N_THREADS``
     - ``8``
     - CPU threads for inference
   * - ``ELPIS_MODEL_TEMPERATURE``
     - ``0.7``
     - Default sampling temperature (0.0-2.0)
   * - ``ELPIS_MODEL_TOP_P``
     - ``0.9``
     - Default top-p sampling (0.0-1.0)
   * - ``ELPIS_MODEL_MAX_TOKENS``
     - ``4096``
     - Default max tokens to generate
   * - ``ELPIS_MODEL_HARDWARE_BACKEND``
     - ``auto``
     - Hardware: ``auto``, ``cuda``, ``rocm``, ``cpu``
   * - ``ELPIS_MODEL_TORCH_DTYPE``
     - ``auto``
     - Torch dtype: ``auto``, ``float16``, ``bfloat16``, ``float32``
   * - ``ELPIS_MODEL_STEERING_LAYER``
     - ``15``
     - Layer for steering vector injection (transformers)
   * - ``ELPIS_MODEL_EMOTION_VECTORS_DIR``
     - ``None``
     - Directory with trained emotion vectors

Emotion Settings
^^^^^^^^^^^^^^^^

These settings control the emotional regulation system.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``ELPIS_EMOTION_BASELINE_VALENCE``
     - ``0.0``
     - Baseline valence (-1.0 to 1.0)
   * - ``ELPIS_EMOTION_BASELINE_AROUSAL``
     - ``0.0``
     - Baseline arousal (-1.0 to 1.0)
   * - ``ELPIS_EMOTION_DECAY_RATE``
     - ``0.1``
     - Decay rate toward baseline per second
   * - ``ELPIS_EMOTION_MAX_DELTA``
     - ``0.5``
     - Maximum emotional shift per event
   * - ``ELPIS_EMOTION_STEERING_STRENGTH``
     - ``1.0``
     - Global steering strength multiplier

Trajectory Settings
^^^^^^^^^^^^^^^^^^^

These settings control emotional trajectory tracking and pattern detection.

.. list-table::
   :header-rows: 1
   :widths: 40 10 50

   * - Variable
     - Default
     - Description
   * - ``ELPIS_EMOTION_TRAJECTORY_HISTORY_SIZE``
     - ``20``
     - Number of emotional states to retain for trajectory analysis
   * - ``ELPIS_EMOTION_MOMENTUM_POSITIVE_THRESHOLD``
     - ``0.01``
     - Valence velocity threshold for positive momentum classification
   * - ``ELPIS_EMOTION_MOMENTUM_NEGATIVE_THRESHOLD``
     - ``-0.01``
     - Valence velocity threshold for negative momentum classification
   * - ``ELPIS_EMOTION_TREND_IMPROVING_THRESHOLD``
     - ``0.02``
     - Valence velocity threshold for "improving" trend
   * - ``ELPIS_EMOTION_TREND_DECLINING_THRESHOLD``
     - ``-0.02``
     - Valence velocity threshold for "declining" trend
   * - ``ELPIS_EMOTION_SPIRAL_HISTORY_COUNT``
     - ``5``
     - Number of recent states to check for spiral detection
   * - ``ELPIS_EMOTION_SPIRAL_INCREASING_THRESHOLD``
     - ``3``
     - Consecutive distance increases needed to detect a spiral

Logging Settings
^^^^^^^^^^^^^^^^

These settings control logging behavior.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``ELPIS_LOGGING_LEVEL``
     - ``INFO``
     - Log level: DEBUG, INFO, WARNING, ERROR
   * - ``ELPIS_LOGGING_OUTPUT_FILE``
     - ``./logs/elpis.log``
     - Log file path
   * - ``ELPIS_LOGGING_FORMAT``
     - ``json``
     - Log format type

Settings Classes
----------------

Elpis configuration is structured as nested Pydantic settings classes:

.. code-block:: python

    from elpis.config.settings import Settings

    # Load from environment
    settings = Settings()

    # Access nested settings
    print(settings.model.backend)      # "llama-cpp"
    print(settings.emotion.decay_rate)  # 0.1

The main :class:`~elpis.config.settings.Settings` class contains:

- :class:`~elpis.config.settings.ModelSettings` - LLM configuration
- :class:`~elpis.config.settings.EmotionSettings` - Emotional regulation
- :class:`~elpis.config.settings.LoggingSettings` - Logging configuration

.. note::

   Tool settings are configured in Hermes (the TUI client) rather than Elpis.
   See :doc:`/hermes/api/tools` for tool configuration options.

Example Configurations
----------------------

Basic llama-cpp Setup
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # .env file for basic GGUF model inference
    ELPIS_MODEL_BACKEND=llama-cpp
    ELPIS_MODEL_PATH=./data/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
    ELPIS_MODEL_GPU_LAYERS=35
    ELPIS_MODEL_CONTEXT_LENGTH=8192

Transformers with Steering Vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # .env file for HuggingFace model with emotional steering
    ELPIS_MODEL_BACKEND=transformers
    ELPIS_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
    ELPIS_MODEL_TORCH_DTYPE=bfloat16
    ELPIS_MODEL_STEERING_LAYER=15
    ELPIS_MODEL_EMOTION_VECTORS_DIR=./data/vectors/llama-3.1-8b

Custom Emotional Baseline
^^^^^^^^^^^^^^^^^^^^^^^^^

Configure a slightly positive and calm baseline personality:

.. code-block:: bash

    # Slightly positive, calm personality
    ELPIS_EMOTION_BASELINE_VALENCE=0.2
    ELPIS_EMOTION_BASELINE_AROUSAL=-0.1
    ELPIS_EMOTION_DECAY_RATE=0.05  # Slower return to baseline

CPU-Only Inference
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # CPU-only configuration
    ELPIS_MODEL_BACKEND=llama-cpp
    ELPIS_MODEL_HARDWARE_BACKEND=cpu
    ELPIS_MODEL_GPU_LAYERS=0
    ELPIS_MODEL_N_THREADS=16  # Use more CPU threads

Programmatic Configuration
--------------------------

You can also configure Elpis programmatically:

.. code-block:: python

    from elpis.config.settings import (
        Settings,
        ModelSettings,
        EmotionSettings,
    )
    from elpis.server import initialize

    # Create custom settings
    settings = Settings(
        model=ModelSettings(
            backend="transformers",
            path="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.8,
        ),
        emotion=EmotionSettings(
            baseline_valence=0.1,
            decay_rate=0.05,
        ),
    )

    # Initialize server with custom settings
    context = initialize(settings)

Backend-Specific Configuration
------------------------------

Each backend has its own configuration class with relevant options:

llama-cpp Backend
^^^^^^^^^^^^^^^^^

Uses :class:`~elpis.llm.backends.llama_cpp.config.LlamaCppConfig`:

- ``gpu_layers``: Number of layers on GPU (0 for CPU-only)
- ``n_threads``: CPU threads for matrix operations
- ``chat_format``: Chat template (e.g., ``llama-3``)

Transformers Backend
^^^^^^^^^^^^^^^^^^^^

Uses :class:`~elpis.llm.backends.transformers.config.TransformersConfig`:

- ``torch_dtype``: Model precision (``bfloat16`` recommended for GPU)
- ``steering_layer``: Layer for vector injection (typically middle layers)
- ``emotion_vectors_dir``: Path to trained ``.pt`` vector files

See :doc:`backends` for detailed backend comparison and usage.
