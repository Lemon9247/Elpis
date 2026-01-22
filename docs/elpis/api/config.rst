Configuration Module
====================

The configuration module provides Pydantic settings models for all Elpis
configuration options.

.. module:: elpis.config.settings
   :synopsis: Configuration settings for Elpis

Overview
--------

Elpis uses `pydantic-settings <https://docs.pydantic.dev/latest/concepts/pydantic_settings/>`_
for configuration management. Settings can be loaded from:

- Environment variables
- ``.env`` files
- Programmatic instantiation

All settings classes support environment variable prefixes and nested
delimiter syntax.

Root Settings
-------------

.. autoclass:: elpis.config.settings.Settings
   :members:
   :undoc-members:
   :show-inheritance:

The root ``Settings`` class aggregates all configuration sections:

.. code-block:: python

    from elpis.config.settings import Settings

    # Load from environment
    settings = Settings()

    # Access sections
    settings.model      # ModelSettings
    settings.emotion    # EmotionSettings
    settings.logging    # LoggingSettings

Model Settings
--------------

.. autoclass:: elpis.config.settings.ModelSettings
   :members:
   :undoc-members:
   :show-inheritance:

Configuration for the LLM inference backend.

**Environment prefix**: ``ELPIS_MODEL_``

Example:

.. code-block:: bash

    export ELPIS_MODEL_BACKEND=llama-cpp
    export ELPIS_MODEL_PATH=./model.gguf
    export ELPIS_MODEL_GPU_LAYERS=35

Backend Conversion
^^^^^^^^^^^^^^^^^^

``ModelSettings`` can be converted to backend-specific configs:

.. code-block:: python

    from elpis.config.settings import ModelSettings

    settings = ModelSettings(backend="llama-cpp", path="./model.gguf")

    # Convert to llama-cpp config
    llama_config = settings.to_llama_cpp_config()

    # Convert to transformers config
    tf_config = settings.to_transformers_config()

Emotion Settings
----------------

.. autoclass:: elpis.config.settings.EmotionSettings
   :members:
   :undoc-members:
   :show-inheritance:

Configuration for the emotional regulation system.

**Environment prefix**: ``ELPIS_EMOTION_``

Example:

.. code-block:: bash

    # Optimistic, calm baseline personality
    export ELPIS_EMOTION_BASELINE_VALENCE=0.2
    export ELPIS_EMOTION_BASELINE_AROUSAL=-0.1
    export ELPIS_EMOTION_DECAY_RATE=0.05

Logging Settings
----------------

.. autoclass:: elpis.config.settings.LoggingSettings
   :members:
   :undoc-members:
   :show-inheritance:

Configuration for logging behavior.

**Environment prefix**: ``ELPIS_LOGGING_``

Example:

.. code-block:: bash

    export ELPIS_LOGGING_LEVEL=DEBUG
    export ELPIS_LOGGING_OUTPUT_FILE=/var/log/elpis.log

Backend Configurations
----------------------

Each inference backend has its own configuration class.

LlamaCppConfig
^^^^^^^^^^^^^^

.. autoclass:: elpis.llm.backends.llama_cpp.config.LlamaCppConfig
   :members:
   :undoc-members:
   :show-inheritance:

TransformersConfig
^^^^^^^^^^^^^^^^^^

.. autoclass:: elpis.llm.backends.transformers.config.TransformersConfig
   :members:
   :undoc-members:
   :show-inheritance:

Environment Variable Reference
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Description
   * - ``ELPIS_MODEL_BACKEND``
     - Inference backend (llama-cpp, transformers)
   * - ``ELPIS_MODEL_PATH``
     - Path to model file or HuggingFace ID
   * - ``ELPIS_MODEL_CONTEXT_LENGTH``
     - Context window size
   * - ``ELPIS_MODEL_GPU_LAYERS``
     - GPU layers (llama-cpp)
   * - ``ELPIS_MODEL_N_THREADS``
     - CPU threads
   * - ``ELPIS_MODEL_TEMPERATURE``
     - Default temperature
   * - ``ELPIS_MODEL_TOP_P``
     - Default top-p
   * - ``ELPIS_MODEL_MAX_TOKENS``
     - Default max tokens
   * - ``ELPIS_MODEL_HARDWARE_BACKEND``
     - Hardware: auto, cuda, rocm, cpu
   * - ``ELPIS_MODEL_TORCH_DTYPE``
     - Torch dtype (transformers)
   * - ``ELPIS_MODEL_STEERING_LAYER``
     - Steering layer (transformers)
   * - ``ELPIS_MODEL_EMOTION_VECTORS_DIR``
     - Vector directory (transformers)
   * - ``ELPIS_EMOTION_BASELINE_VALENCE``
     - Baseline valence
   * - ``ELPIS_EMOTION_BASELINE_AROUSAL``
     - Baseline arousal
   * - ``ELPIS_EMOTION_DECAY_RATE``
     - Decay rate per second
   * - ``ELPIS_EMOTION_MAX_DELTA``
     - Max event delta
   * - ``ELPIS_EMOTION_STEERING_STRENGTH``
     - Global steering multiplier
   * - ``ELPIS_LOGGING_LEVEL``
     - Log level
   * - ``ELPIS_LOGGING_OUTPUT_FILE``
     - Log file path
   * - ``ELPIS_LOGGING_FORMAT``
     - Log format
