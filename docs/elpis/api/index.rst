API Reference
=============

This section contains the complete API reference for the Elpis inference server,
generated from source code docstrings.

Modules
-------

.. toctree::
   :maxdepth: 2

   server
   config
   emotion
   llm

Quick Links
-----------

**Server**
    - :class:`~elpis.server.ServerContext` - Server dependency container
    - :func:`~elpis.server.initialize` - Initialize server components
    - :func:`~elpis.server.run_server` - Run the MCP server

**Configuration**
    - :class:`~elpis.config.settings.Settings` - Root configuration
    - :class:`~elpis.config.settings.ModelSettings` - LLM settings
    - :class:`~elpis.config.settings.EmotionSettings` - Emotion settings

**Emotion System**
    - :class:`~elpis.emotion.state.EmotionalState` - Valence-arousal state
    - :class:`~elpis.emotion.regulation.HomeostasisRegulator` - Event processing

**Inference**
    - :class:`~elpis.llm.base.InferenceEngine` - Abstract inference interface
    - :func:`~elpis.llm.backends.create_backend` - Backend factory
    - :class:`~elpis.llm.backends.transformers.steering.SteeringManager` - Vector injection
