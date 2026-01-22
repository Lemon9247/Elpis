==========
MCP Module
==========

The MCP (Model Context Protocol) module provides the client for connecting to
the Elpis inference server.

psyche.mcp.client
-----------------

Elpis Client
^^^^^^^^^^^^

The ElpisClient manages the connection to the Elpis inference server and provides
methods for text generation, function calling, and emotional state management.

.. automodule:: psyche.mcp.client
   :members:
   :undoc-members:
   :show-inheritance:

Connection Management
^^^^^^^^^^^^^^^^^^^^^

The client uses an async context manager for connection:

.. code-block:: python

    from psyche.mcp.client import ElpisClient

    client = ElpisClient(server_command="elpis-server")

    async with client.connect():
        result = await client.generate(messages=[
            {"role": "user", "content": "Hello!"}
        ])
        print(result.content)

Text Generation
^^^^^^^^^^^^^^^

Generate text with optional emotional modulation:

.. code-block:: python

    result = await client.generate(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Python?"},
        ],
        max_tokens=2048,
        temperature=None,  # Use emotionally modulated temperature
        emotional_modulation=True,
    )

    print(result.content)
    print(f"Emotional state: {result.emotional_state.quadrant}")

Streaming Generation
^^^^^^^^^^^^^^^^^^^^

Stream tokens as they are generated:

.. code-block:: python

    async for token in client.generate_stream(
        messages=[{"role": "user", "content": "Tell me a story."}],
        max_tokens=500,
        poll_interval=0.05,
    ):
        print(token, end="", flush=True)

Function Calling
^^^^^^^^^^^^^^^^

Generate function/tool calls:

.. code-block:: python

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }
    ]

    result = await client.function_call(
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=tools,
    )

    for call in result.tool_calls:
        print(f"Call: {call['function']['name']}")

Emotional State Management
^^^^^^^^^^^^^^^^^^^^^^^^^^

Query and update emotional state:

.. code-block:: python

    # Get current state
    emotion = await client.get_emotion()
    print(f"Quadrant: {emotion.quadrant}")
    print(f"Valence: {emotion.valence}")
    print(f"Arousal: {emotion.arousal}")

    # Trigger an emotional event
    emotion = await client.update_emotion(
        event_type="success",
        intensity=0.5,
        context="Task completed successfully",
    )

    # Reset to baseline
    emotion = await client.reset_emotion()

Data Classes
^^^^^^^^^^^^

.. py:class:: EmotionalState
   :noindex:

   Representation of the inference server's emotional state.

   .. py:attribute:: valence
      :type: float
      :noindex:

      Emotional valence from -1 (negative) to 1 (positive).

   .. py:attribute:: arousal
      :type: float
      :noindex:

      Emotional arousal from 0 (low) to 1 (high).

   .. py:attribute:: quadrant
      :type: str
      :noindex:

      Quadrant classification: "excited", "calm", "frustrated", or "depleted".

   .. py:attribute:: update_count
      :type: int
      :noindex:

      Number of emotional updates since last reset.

.. py:class:: GenerationResult
   :noindex:

   Result from a generation request.

   .. py:attribute:: content
      :type: str
      :noindex:

      Generated text content.

   .. py:attribute:: emotional_state
      :type: EmotionalState
      :noindex:

      Emotional state at time of generation.

   .. py:attribute:: modulated_params
      :type: Dict[str, float]
      :noindex:

      Parameters that were modulated by emotional state.

.. py:class:: FunctionCallResult
   :noindex:

   Result from a function call request.

   .. py:attribute:: tool_calls
      :type: List[Dict[str, Any]]
      :noindex:

      List of tool calls in OpenAI format.

   .. py:attribute:: emotional_state
      :type: EmotionalState
      :noindex:

      Emotional state at time of generation.

Resource Reading
^^^^^^^^^^^^^^^^

Read resources from the server:

.. code-block:: python

    # Read emotional state resource
    content = await client.read_resource("emotion://state")

    # List available emotional events
    events = await client.list_available_events()
