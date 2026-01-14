========
Features
========

Psyche provides a rich set of features for interactive AI conversations with
emotional modulation and continuous inference capabilities.

TUI Interface
-------------

The primary interface is a Textual-based Terminal User Interface (TUI) that
provides a modern, responsive terminal experience.

Layout
^^^^^^

The TUI is organized into several panels:

**Main Chat Area**
    The central panel displays the conversation history with user messages,
    assistant responses, and system notifications. Messages are rendered with
    rich formatting and markdown support.

**Sidebar**
    Contains status widgets:

    - *Emotional State Display*: Visual bars showing valence and arousal with
      quadrant classification
    - *Status Display*: Server state, message count, and token usage
    - *Tool Activity*: Recent tool executions with status indicators

**Thought Panel**
    A collapsible panel showing internal AI thoughts and reflections.
    Toggle visibility with ``Ctrl+T`` or ``/thoughts on|off``.

**Input Area**
    Text input with placeholder text and command detection.

Streaming Responses
^^^^^^^^^^^^^^^^^^^

Psyche streams tokens in real-time as they are generated:

.. code-block:: python

    # Token streaming is handled automatically via callbacks
    async for token in client.generate_stream(messages):
        chat_view.append_token(token)

This provides immediate visual feedback and a more responsive experience
compared to waiting for complete responses.

Widgets
^^^^^^^

The TUI is composed of several custom Textual widgets:

:class:`~psyche.client.widgets.ChatView`
    Scrollable chat history with streaming message support. Handles user messages,
    assistant responses, and system notifications with rich formatting.

:class:`~psyche.client.widgets.Sidebar`
    Container for status widgets including emotional state, server status,
    and tool activity displays.

:class:`~psyche.client.widgets.ThoughtPanel`
    Collapsible panel for internal thoughts with type-based color coding
    (reflection, planning, idle, memory).

:class:`~psyche.client.widgets.UserInput`
    Input widget with slash command detection and placeholder text.

:class:`~psyche.client.widgets.ToolActivity`
    Displays active and recent tool executions with status indicators
    (running, complete, error).

REPL Mode
---------

For simpler interactions, Psyche provides a REPL (Read-Eval-Print Loop)
interface using prompt_toolkit:

.. code-block:: python

    from psyche.client.repl import PsycheREPL
    from psyche.memory.server import MemoryServer

    repl = PsycheREPL(server=memory_server)
    await repl.run()

Features include:

- Command history with file persistence
- Rich console output with markdown rendering
- Thinking indicators during generation
- The same slash commands as the TUI

Continuous Inference
--------------------

Unlike traditional chatbots that only respond to user input, Psyche maintains
a continuous thought process.

Idle Thinking
^^^^^^^^^^^^^

When no user input is received for a configurable interval (default: 30 seconds),
the memory server generates internal reflections:

.. code-block:: python

    @dataclass
    class ServerConfig:
        idle_think_interval: float = 30.0  # Seconds between idle thoughts
        think_temperature: float = 0.7     # Temperature for reflection

These reflections are displayed in the thought panel (if visible) but are not
added to the main conversation context.

Post-Interaction Delay
^^^^^^^^^^^^^^^^^^^^^^

To avoid the AI appearing to "continue speaking" immediately after responding,
idle thinking is delayed after user interactions:

.. code-block:: python

    post_interaction_delay: float = 60.0  # Wait 60s after user input

Idle Tool Access
^^^^^^^^^^^^^^^^

During idle reflection, the AI can optionally use read-only tools to explore
the workspace:

.. code-block:: python

    allow_idle_tools: bool = True        # Enable sandboxed tool use
    max_idle_tool_iterations: int = 3    # Max tool calls per reflection

Only safe tools are allowed (``read_file``, ``list_directory``, ``search_codebase``),
and sensitive paths are blocked. See :doc:`tools` for details.

Emotional State Display
-----------------------

Psyche visualizes the Elpis server's emotional state using a valence-arousal model.

Valence-Arousal Model
^^^^^^^^^^^^^^^^^^^^^

The emotional state is represented by two dimensions:

- **Valence** [-1, 1]: Negative to positive affect
- **Arousal** [0, 1]: Low to high activation/energy

Quadrants
^^^^^^^^^

The emotional space is divided into four quadrants:

+------------+--------------------+----------------------+
| Quadrant   | Valence            | Arousal              |
+============+====================+======================+
| Excited    | Positive (> 0)     | High (> 0.5)         |
+------------+--------------------+----------------------+
| Calm       | Positive (> 0)     | Low (<= 0.5)         |
+------------+--------------------+----------------------+
| Frustrated | Negative (<= 0)    | High (> 0.5)         |
+------------+--------------------+----------------------+
| Depleted   | Negative (<= 0)    | Low (<= 0.5)         |
+------------+--------------------+----------------------+

Visual Display
^^^^^^^^^^^^^^

The sidebar displays emotional state with visual bars:

.. code-block:: text

    Excited

    Valence: [=======---] +0.45
    Arousal: [======----] 0.60

Color coding indicates the current quadrant:

- **Excited**: Bright green
- **Calm**: Blue
- **Frustrated**: Red
- **Depleted**: Dim/gray

Memory Consolidation
--------------------

Psyche integrates with Mnemosyne for automatic memory consolidation during idle periods.

How It Works
^^^^^^^^^^^^

1. **Dual Connection**: Psyche connects to both Elpis (inference) and Mnemosyne (memory)
2. **Periodic Check**: After each idle thought, checks if consolidation is recommended
3. **Automatic Trigger**: If short-term buffer exceeds threshold, runs consolidation
4. **Background Processing**: Consolidation runs without interrupting the UI

Configuration
^^^^^^^^^^^^^

.. code-block:: python

    ServerConfig(
        enable_consolidation=True,              # Enable/disable consolidation
        consolidation_check_interval=300.0,     # Check every 5 minutes
        consolidation_importance_threshold=0.6, # Min importance for promotion
        consolidation_similarity_threshold=0.85 # Clustering threshold
    )

CLI Options
^^^^^^^^^^^

.. code-block:: bash

    # Default (with consolidation enabled)
    psyche

    # Disable consolidation
    psyche --no-consolidation

    # Custom Mnemosyne command
    psyche --mnemosyne-command "mnemosyne-server --persist-dir ./my-memories"

MnemosyneClient
^^^^^^^^^^^^^^^

The ``MnemosyneClient`` class provides the interface to Mnemosyne:

.. code-block:: python

    from psyche.mcp.client import MnemosyneClient

    client = MnemosyneClient(server_command="mnemosyne-server")

    async with client.connect():
        # Check if consolidation is recommended
        should, reason, st_count, lt_count = await client.should_consolidate()

        if should:
            # Run consolidation
            result = await client.consolidate_memories(
                importance_threshold=0.6,
                similarity_threshold=0.85
            )
            print(f"Promoted {result.memories_promoted} memories")

        # Store a memory
        await client.store_memory(
            content="User prefers dark mode",
            summary="UI preference",
            memory_type="semantic"
        )

        # Search memories
        results = await client.search_memories("user preferences", n_results=5)

Context Management
------------------

Psyche manages conversation context to stay within token limits.

Context Compaction
^^^^^^^^^^^^^^^^^^

The :class:`~psyche.memory.compaction.ContextCompactor` handles context size:

.. code-block:: python

    compactor = ContextCompactor(
        max_tokens=24000,        # Maximum context tokens
        reserve_tokens=4000,     # Reserve for response generation
        min_messages_to_keep=4,  # Always keep recent messages
    )

Compaction Strategies
^^^^^^^^^^^^^^^^^^^^^

Two strategies are available:

**Sliding Window** (default)
    Drops oldest messages when approaching limits. Simple and predictable.

**Summarization**
    Compresses old messages into a summary using a provided function.
    Preserves more context but requires additional inference.

Token Estimation
^^^^^^^^^^^^^^^^

Token counts are estimated using a word-based heuristic:

.. code-block:: python

    def estimate_tokens(text: str) -> int:
        # Rough estimate: ~1.3 tokens per word for English
        words = len(text.split())
        return int(words * 1.3)

Commands and Shortcuts
----------------------

Slash Commands
^^^^^^^^^^^^^^

+---------------+-------------------------------------------+
| Command       | Description                               |
+===============+===========================================+
| ``/help``     | Show available commands                   |
+---------------+-------------------------------------------+
| ``/status``   | Display server state and context usage    |
+---------------+-------------------------------------------+
| ``/clear``    | Clear conversation context                |
+---------------+-------------------------------------------+
| ``/emotion``  | Show current emotional state              |
+---------------+-------------------------------------------+
| ``/thoughts`` | Toggle internal thought panel             |
+---------------+-------------------------------------------+
| ``/quit``     | Exit Psyche                               |
+---------------+-------------------------------------------+

Keyboard Shortcuts
^^^^^^^^^^^^^^^^^^

+--------------+----------------------------+
| Shortcut     | Action                     |
+==============+============================+
| ``Ctrl+C``   | Quit application           |
+--------------+----------------------------+
| ``Ctrl+L``   | Clear conversation context |
+--------------+----------------------------+
| ``Ctrl+T``   | Toggle thought panel       |
+--------------+----------------------------+
| ``Escape``   | Focus input widget         |
+--------------+----------------------------+
