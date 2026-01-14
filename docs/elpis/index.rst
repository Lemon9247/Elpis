Elpis Inference Server
======================

Elpis is an MCP (Model Context Protocol) server for emotional inference. It provides
LLM inference capabilities with built-in emotional state modulation, enabling more
nuanced and contextually appropriate AI responses.

Key Features
------------

**Emotional Modulation**
    Elpis implements a valence-arousal emotional model that influences inference
    behavior. The system can express different emotional states (excited, frustrated,
    calm, depleted) through either sampling parameter adjustment or steering vector
    injection.

**Multiple Backends**
    Choose between two inference backends based on your needs:

    - **llama-cpp**: Fast inference with GGUF quantized models, emotional modulation
      via temperature/top_p adjustment
    - **transformers**: Full HuggingFace support with steering vector injection for
      activation-level emotional modulation

**MCP Protocol Support**
    Elpis exposes its capabilities through the MCP protocol, enabling integration
    with MCP-compatible clients like Claude Desktop or the included Psyche TUI.

**Streaming Generation**
    Both backends support streaming token generation for real-time response display.

Architecture Overview
---------------------

.. code-block:: text

    +------------------+
    |   MCP Client     |  (Psyche TUI, Claude Desktop, etc.)
    +--------+---------+
             |
             | MCP Protocol (stdio)
             v
    +------------------+
    |   Elpis Server   |
    |  +------------+  |
    |  | Emotional  |  |  <-- Valence/Arousal state
    |  | Regulator  |  |
    |  +-----+------+  |
    |        |         |
    |  +-----v------+  |
    |  | Inference  |  |  <-- llama-cpp or transformers
    |  |  Backend   |  |
    |  +------------+  |
    +------------------+

Quick Start
-----------

Start the server with default settings:

.. code-block:: bash

    # Using the CLI entry point
    elpis-server

    # Or with Python module
    python -m elpis.cli

Configure via environment variables:

.. code-block:: bash

    export ELPIS_MODEL_PATH="./data/models/model.gguf"
    export ELPIS_MODEL_BACKEND="llama-cpp"
    elpis-server

MCP Tools
---------

Elpis exposes several MCP tools:

``generate``
    Generate text completion with emotional modulation. Accepts chat messages
    in OpenAI format and returns the generated response along with current
    emotional state.

``generate_stream_start`` / ``generate_stream_read``
    Start and poll streaming generation for real-time token delivery.

``function_call``
    Generate function/tool calls with emotional modulation.

``update_emotion``
    Manually trigger an emotional event (success, failure, frustration, etc.)
    to shift the system's emotional state.

``reset_emotion``
    Reset emotional state to baseline.

``get_emotion``
    Retrieve current emotional state including valence, arousal, and steering
    coefficients.

MCP Resources
-------------

``emotion://state``
    Current emotional state as JSON, including valence, arousal, quadrant,
    and steering coefficients.

``emotion://events``
    Available emotional event types and their effects on valence/arousal.

Contents
--------

.. toctree::
   :maxdepth: 2

   configuration
   emotion-system
   backends
   api/index
