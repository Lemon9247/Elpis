Server Module
=============

The server module provides the MCP server implementation and entry point
for Elpis.

.. module:: elpis.server
   :synopsis: MCP server for emotional inference

Overview
--------

The server module exposes LLM inference capabilities through the Model Context
Protocol (MCP). It manages server dependencies, handles tool calls, and
coordinates between the emotional regulation system and inference backends.

Server Context
--------------

.. autoclass:: elpis.server.ServerContext
   :members:
   :undoc-members:
   :show-inheritance:

The ``ServerContext`` dataclass holds all initialized server components:

.. code-block:: python

    from elpis.server import initialize, get_context

    # Initialize (typically done once at startup)
    context = initialize()

    # Access components
    print(context.llm)            # InferenceEngine
    print(context.emotion_state)  # EmotionalState
    print(context.regulator)      # HomeostasisRegulator
    print(context.settings)       # Settings

Stream State
------------

.. autoclass:: elpis.server.StreamState
   :members:
   :undoc-members:
   :show-inheritance:

Tracks active streaming generation sessions.

Server Functions
----------------

.. autofunction:: elpis.server.initialize

.. autofunction:: elpis.server.get_context

.. autofunction:: elpis.server.run_server

.. autofunction:: elpis.server.main

MCP Tool Handlers
-----------------

The server exposes the following MCP tools, each handled by an internal
async function:

``generate``
    Generates text completion with optional emotional modulation.
    Handled by ``_handle_generate()``.

``generate_stream_start``
    Starts streaming generation and returns a stream ID.
    Handled by ``_handle_generate_stream_start()``.

``generate_stream_read``
    Reads new tokens from an active stream.
    Handled by ``_handle_generate_stream_read()``.

``generate_stream_cancel``
    Cancels an active stream.
    Handled by ``_handle_generate_stream_cancel()``.

``function_call``
    Generates tool/function calls.
    Handled by ``_handle_function_call()``.

``update_emotion``
    Manually triggers an emotional event.
    Handled by ``_handle_update_emotion()``.

``reset_emotion``
    Resets emotional state to baseline.
    Handled by ``_handle_reset_emotion()``.

``get_emotion``
    Returns current emotional state.
    Handled by ``_handle_get_emotion()``.

MCP Resources
-------------

The server exposes two resources:

``emotion://state``
    Current emotional state as JSON, including valence, arousal,
    quadrant, and steering coefficients.

``emotion://events``
    Available event types and their valence/arousal effects.

Usage Example
-------------

.. code-block:: python

    import asyncio
    from elpis.server import initialize, run_server
    from elpis.config.settings import Settings, ModelSettings

    # Custom configuration
    settings = Settings(
        model=ModelSettings(
            backend="llama-cpp",
            path="./model.gguf",
        )
    )

    # Initialize and run
    context = initialize(settings)
    asyncio.run(run_server())

CLI Entry Point
---------------

The server can be started via the CLI:

.. code-block:: bash

    # Using the entry point
    elpis-server

    # Using Python module
    python -m elpis.cli

    # With quiet mode (for use as subprocess)
    ELPIS_QUIET=1 elpis-server

When ``ELPIS_QUIET=1`` is set, logging is redirected to a file instead
of stderr, making the server suitable for use as a subprocess of a TUI
application.
