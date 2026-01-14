=============
Memory Module
=============

The memory module provides the continuous inference server and context management
for Psyche.

psyche.memory.server
--------------------

Memory Server
^^^^^^^^^^^^^

The memory server maintains an always-active thought process that processes user
input, generates responses, manages context, and produces idle reflections.

.. automodule:: psyche.memory.server
   :members:
   :undoc-members:
   :show-inheritance:

Server States
^^^^^^^^^^^^^

The server operates in several states:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - State
     - Description
   * - IDLE
     - No active processing
   * - THINKING
     - Generating a response or reflection
   * - WAITING_INPUT
     - Waiting for user input
   * - PROCESSING_TOOLS
     - Executing a tool call
   * - SHUTTING_DOWN
     - Server is stopping

Configuration
^^^^^^^^^^^^^

The server is configured through :class:`~psyche.memory.server.ServerConfig`:

.. code-block:: python

    @dataclass
    class ServerConfig:
        # Inference settings
        idle_think_interval: float = 30.0
        think_temperature: float = 0.7

        # Context settings
        max_context_tokens: int = 24000
        reserve_tokens: int = 4000

        # Tool settings
        workspace_dir: str = "."
        max_tool_iterations: int = 10
        max_tool_result_chars: int = 16000

        # Idle reflection settings
        allow_idle_tools: bool = True
        max_idle_tool_iterations: int = 3
        max_idle_result_chars: int = 8000

        # Rate limiting
        startup_warmup_seconds: float = 120.0
        idle_tool_cooldown_seconds: float = 300.0
        post_interaction_delay: float = 60.0

Callbacks
^^^^^^^^^

The memory server provides callbacks for UI integration:

.. code-block:: python

    server = MemoryServer(
        elpis_client=client,
        on_token=lambda token: print(token, end=""),
        on_thought=lambda thought: print(f"[{thought.thought_type}]"),
        on_response=lambda content: print("\n---"),
        on_tool_call=lambda name, result: print(f"Tool: {name}"),
    )

psyche.memory.compaction
------------------------

Context Compaction
^^^^^^^^^^^^^^^^^^

The context compactor manages conversation history within token limits.

.. automodule:: psyche.memory.compaction
   :members:
   :undoc-members:
   :show-inheritance:

Compaction Strategies
^^^^^^^^^^^^^^^^^^^^^

**Sliding Window**

The default strategy drops oldest messages when approaching limits:

.. code-block:: python

    compactor = ContextCompactor(
        max_tokens=24000,
        reserve_tokens=4000,
        min_messages_to_keep=4,
    )

**Summarization**

Optionally compress old messages into a summary:

.. code-block:: python

    def summarize(messages: List[Message]) -> str:
        # Generate a summary of the messages
        return "Summary of conversation..."

    compactor = ContextCompactor(
        max_tokens=24000,
        reserve_tokens=4000,
        summarize_fn=summarize,
    )

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

    from psyche.memory.compaction import ContextCompactor, create_message

    compactor = ContextCompactor(max_tokens=8000, reserve_tokens=2000)

    # Add messages
    compactor.add_message(create_message("user", "Hello!"))
    compactor.add_message(create_message("assistant", "Hi there!"))

    # Get messages for API call
    api_messages = compactor.get_api_messages()

    # Check token usage
    print(f"Tokens used: {compactor.total_tokens}")
    print(f"Available: {compactor.available_tokens}")
