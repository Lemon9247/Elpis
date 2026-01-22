======
Hermes
======

Hermes is the TUI (Terminal User Interface) client for the Elpis system. Built
with Textual, it provides a rich terminal-based interface for interacting with
Psyche, featuring streaming responses, emotional state display, and tool activity
visualization.

Overview
--------

Hermes connects to a running Psyche server via HTTP and provides the user-facing
interface in the Elpis ecosystem. It handles:

- Terminal-based chat with streaming responses
- Local execution of file, bash, and search tools
- Emotional state display from the server
- Tool activity visualization

Memory operations (recall, store, consolidate) execute server-side as part of
Psyche's "self", while filesystem and command tools run locally on the client.

Features
--------

Chat Interface
    Real-time streaming responses with markdown rendering. Messages display
    as they're generated, providing immediate feedback.

Emotional State Display
    Sidebar shows current valence-arousal state with quadrant classification
    (excited, calm, frustrated, depleted).

Tool Activity
    Visual display of tool calls with expandable details. Shows tool name,
    arguments, and results for transparency.

Thought Panel
    Optional panel showing internal reasoning (``<reasoning>`` blocks) and
    idle thoughts. Toggle with ``Ctrl+T``.

Slash Commands
    Built-in commands for control: ``/help``, ``/status``, ``/clear``,
    ``/emotion``, ``/thoughts``, ``/quit``.

Quick Start
-----------

1. Start the Psyche server:

.. code-block:: bash

   psyche-server

2. Connect with Hermes:

.. code-block:: bash

   hermes

By default, Hermes connects to ``http://127.0.0.1:8741``. To connect to a
different server:

.. code-block:: bash

   hermes --server http://myserver:8741

   # With debug logging
   hermes --debug

Keyboard Shortcuts
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Shortcut
     - Action
   * - ``Ctrl+C``
     - Stop generation or quit (double-tap when idle)
   * - ``Ctrl+L``
     - Clear conversation context
   * - ``Ctrl+T``
     - Toggle thought panel
   * - ``Escape``
     - Focus input field

Slash Commands
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``/help``
     - Show available commands
   * - ``/status``
     - Display server status and context usage
   * - ``/clear``
     - Clear conversation context
   * - ``/emotion``
     - Show current emotional state
   * - ``/thoughts on|off``
     - Toggle internal thought display
   * - ``/quit``
     - Exit Hermes

Components
----------

Hermes is organized into these modules:

**App** (``hermes.app``)
    The main Textual application managing the UI layout and user interaction.

**Widgets** (``hermes.widgets``)
    Custom Textual widgets:

    - ``ChatView``: Scrollable message display with streaming support
    - ``UserInput``: Input field with history and command detection
    - ``Sidebar``: Status display, emotional state, and controls
    - ``ThoughtPanel``: Internal reasoning display
    - ``ToolActivity``: Tool call visualization

**Commands** (``hermes.commands``)
    Slash command handling and help text formatting.

**CLI** (``hermes.cli``)
    Command-line entry point with argument parsing.

**Tools** (``hermes.tools``)
    Local tool execution engine for file, bash, and search operations.

Architecture
------------

Hermes connects to a Psyche server and executes tools locally:

.. code-block:: text

   Hermes (TUI Client)
     ├── RemotePsycheClient → Psyche Server (HTTP)
     └── ToolEngine (file/bash/search tools)

   Psyche Server (separate process)
     ├── PsycheCore
     │     ├── ElpisClient (inference)
     │     └── MnemosyneClient (memory)
     └── Memory tools execute here

Hermes receives tool calls from the server, executes file/bash/search tools locally,
and sends results back. Memory tools (``recall_memory``, ``store_memory``,
``consolidate_memories``) execute server-side because memory is part of Psyche's
"self".

Configuration
-------------

Hermes inherits most configuration from the Psyche server it connects to.
Command-line options:

.. code-block:: text

   hermes --help

   Options:
     --server URL        Psyche server URL (default: http://127.0.0.1:8741)
     --workspace PATH    Working directory for tool operations (default: .)
     --debug             Enable debug logging
     --log-file PATH     Path to log file
     --help              Show help message

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
