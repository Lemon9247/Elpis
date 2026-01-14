Quickstart
==========

This guide will get you using Elpis in just a few minutes.

Starting the Servers
--------------------

Elpis consists of multiple MCP servers that can be run independently
or together.

Elpis Inference Server
^^^^^^^^^^^^^^^^^^^^^^

The inference server provides LLM generation with emotional modulation:

.. code-block:: bash

   elpis-server

By default, this starts the server using stdio transport for MCP
communication. The server will load your configured model and be
ready to accept requests.

Mnemosyne Memory Server
^^^^^^^^^^^^^^^^^^^^^^^

The memory server provides persistent vector-based storage:

.. code-block:: bash

   mnemosyne-server

This starts the ChromaDB-backed memory server, which stores and
retrieves memories using semantic similarity.

Running the Psyche Client
-------------------------

The easiest way to use Elpis is through the Psyche TUI client, which
manages the servers automatically:

.. code-block:: bash

   psyche

Psyche will:

1. Start the elpis-server as a subprocess
2. Connect to it via MCP stdio transport
3. Provide an interactive terminal interface

.. note::

   When using Psyche, you do NOT need to start elpis-server manually.
   Psyche spawns it automatically as needed.

Basic Usage Flow
----------------

Once Psyche is running, you can interact with the AI system:

1. **Start a conversation** - Type your message and press Enter

2. **View emotional state** - The interface shows the current
   valence-arousal state

3. **Use tools** - The AI can use tools for file operations, search,
   and other tasks

4. **Memory persistence** - Important context is automatically stored
   and recalled

Example Session
^^^^^^^^^^^^^^^

.. code-block:: text

   $ psyche

   Elpis Psyche Client v0.1.0
   Connected to inference server

   You: Hello, how are you today?

   [Emotional State: valence=0.5, arousal=0.3]

   AI: I'm doing well, thank you for asking! I'm feeling
   quite pleasant and calm at the moment. How can I help
   you today?

Programmatic Usage
------------------

You can also use Elpis components directly in Python:

.. code-block:: python

   import asyncio
   from elpis.llm.inference import LlamaInference
   from elpis.config.settings import ModelSettings
   from elpis.emotion.state import EmotionalState

   async def main():
       # Configure the model
       settings = ModelSettings(
           path="./data/models/model.gguf",
           context_length=8192,
       )

       # Initialize inference engine
       llm = LlamaInference(settings)

       # Create an emotional state
       emotion = EmotionalState(valence=0.7, arousal=0.5)

       # Generate with emotional modulation
       response = await llm.chat_completion(
           messages=[{"role": "user", "content": "Hello!"}],
           **emotion.get_modulated_params()
       )

       print(response)

   asyncio.run(main())

Configuration
-------------

Elpis can be configured via YAML files or environment variables.

Configuration File
^^^^^^^^^^^^^^^^^^

Create a ``config.yaml`` file:

.. code-block:: yaml

   model:
     backend: llama-cpp
     path: ./data/models/model.gguf
     context_length: 8192
     gpu_layers: 35

   emotion:
     baseline_valence: 0.0
     baseline_arousal: 0.0
     decay_rate: 0.1
     steering_strength: 1.0

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

Use environment variables with the ``ELPIS_`` prefix:

.. code-block:: bash

   export ELPIS_MODEL__PATH=./data/models/model.gguf
   export ELPIS_MODEL__CONTEXT_LENGTH=8192
   export ELPIS_EMOTION__BASELINE_VALENCE=0.0

MCP Tools
---------

Elpis exposes these MCP tools:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Tool
     - Description
   * - ``generate``
     - Text generation with emotional modulation
   * - ``function_call``
     - Tool/function call generation
   * - ``update_emotion``
     - Trigger an emotional event
   * - ``reset_emotion``
     - Reset to baseline state
   * - ``get_emotion``
     - Get current emotional state

MCP Resources
-------------

Available MCP resources:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Resource URI
     - Description
   * - ``emotion://state``
     - Current valence-arousal state
   * - ``emotion://events``
     - Available emotional event types

Next Steps
----------

- Learn about :doc:`../elpis/index` for inference details
- Explore :doc:`../mnemosyne/index` for memory capabilities
- See :doc:`../psyche/index` for client features
