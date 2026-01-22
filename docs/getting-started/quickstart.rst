Quickstart
==========

This guide will get you using Elpis in just a few minutes.

Getting Started
---------------

Elpis uses a server architecture: Psyche runs as a persistent server managing
inference and memory, while Hermes provides the terminal interface.

.. code-block:: bash

   # Terminal 1: Start the server
   psyche-server

   # Terminal 2: Connect with Hermes
   hermes

By default, Hermes connects to ``http://127.0.0.1:8741``. The architecture:

- **Psyche** manages memory and executes memory tools server-side
- **Hermes** executes file/bash/search tools locally
- Server dreams when no clients are connected
- Multiple clients can connect (memories are shared)

Direct API Access
^^^^^^^^^^^^^^^^^

The Psyche server exposes an OpenAI-compatible API:

.. code-block:: bash

   curl http://localhost:8741/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [{"role": "user", "content": "Hello!"}],
       "stream": true
     }'

Basic Usage
-----------

Once Hermes is running, you can interact with the AI:

1. **Type messages** - Enter your message and press Enter
2. **View emotional state** - The sidebar shows valence-arousal state
3. **Use tools** - The AI can read/write files, run commands, search code
4. **Memory persistence** - Important context is stored and recalled

Slash Commands
^^^^^^^^^^^^^^

- ``/help`` - Show available commands
- ``/status`` - Display server status and context usage
- ``/clear`` - Clear conversation context
- ``/emotion`` - Show current emotional state
- ``/thoughts on|off`` - Toggle internal thought display
- ``/quit`` - Exit Hermes

Keyboard Shortcuts
^^^^^^^^^^^^^^^^^^

- ``Ctrl+C`` - Stop generation or quit (double-tap when idle)
- ``Ctrl+L`` - Clear context
- ``Ctrl+T`` - Toggle thought panel
- ``Escape`` - Focus input

Programmatic Usage
------------------

Use Elpis components directly in Python:

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

Configuration File
^^^^^^^^^^^^^^^^^^

Create a ``configs/elpis.toml`` file:

.. code-block:: toml

   [model]
   backend = "llama-cpp"
   path = "./data/models/model.gguf"
   context_length = 8192
   gpu_layers = 35

   [emotion]
   baseline_valence = 0.0
   baseline_arousal = 0.0
   decay_rate = 0.1
   steering_strength = 1.0

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

Use environment variables with nested ``__`` delimiter:

.. code-block:: bash

   # Elpis settings
   export ELPIS_MODEL__PATH=./data/models/model.gguf
   export ELPIS_MODEL__CONTEXT_LENGTH=16384

   # Mnemosyne settings
   export MNEMOSYNE_STORAGE__PATH=./data/memory

   # Psyche settings
   export PSYCHE_SERVER__PORT=8741

Running Individual Servers
--------------------------

You can run servers independently if needed:

.. code-block:: bash

   # Inference server only
   elpis-server

   # Memory server only
   mnemosyne-server

   # Psyche server (spawns Elpis and Mnemosyne)
   psyche-server --port 8741

Next Steps
----------

- Learn about :doc:`../elpis/index` for inference and emotion details
- Explore :doc:`../mnemosyne/index` for memory capabilities
- See :doc:`../psyche/index` for core library features
- Read :doc:`../hermes/index` for TUI client details
