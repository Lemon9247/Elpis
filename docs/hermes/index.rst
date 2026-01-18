======
Hermes
======

Hermes is the TUI (Terminal User Interface) client for the Elpis system. Built
with Textual, it provides a rich terminal-based interface for interacting with
Psyche, featuring streaming responses, emotional state display, and tool activity
visualization.

Overview
--------

Hermes serves as the user-facing interface in the Elpis ecosystem, supporting
two operational modes:

**Local Mode** (default)
    Hermes spawns Elpis and Mnemosyne as MCP subprocesses, managing everything
    in a single terminal session. All tools execute locally.

**Remote Mode** (``--server``)
    Hermes connects to a running Psyche server via HTTP. File/bash/search tools
    execute locally while memory tools execute server-side.

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

**Local Mode:**

.. code-block:: bash

   # Start Hermes (spawns servers automatically)
   hermes

   # With debug logging
   hermes --debug

**Remote Mode:**

.. code-block:: bash

   # First, start the Psyche server
   psyche-server

   # Then connect with Hermes
   hermes --server http://localhost:8741

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

Architecture
------------

Hermes is organized into these components:

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
    Command-line entry point with argument parsing for local/remote modes.

Local vs Remote Mode
--------------------

**Local Mode:**

.. code-block:: text

   Hermes
     ├── PsycheCore (instantiated directly)
     │     ├── ReactHandler
     │     ├── IdleHandler
     │     ├── ElpisClient → elpis-server (subprocess)
     │     └── MnemosyneClient → mnemosyne-server (subprocess)
     └── ToolEngine (all tools execute locally)

**Remote Mode:**

.. code-block:: text

   Hermes
     ├── RemotePsycheClient → Psyche Server (HTTP)
     └── ToolEngine (file/bash/search tools only)

   Psyche Server (separate process)
     ├── PsycheCore
     │     ├── ElpisClient
     │     └── MnemosyneClient
     └── Memory tools execute here

In remote mode, Hermes receives tool_calls from the server, executes file/bash/search
tools locally, and sends results back. Memory tools (``recall_memory``, ``store_memory``)
execute server-side because memory is part of Psyche's "self".

Configuration
-------------

Hermes inherits most configuration from the servers it connects to. Command-line
options include:

.. code-block:: bash

   hermes --help

   Options:
     --server URL        Connect to Psyche server instead of local mode
     --debug             Enable debug logging
     --workspace PATH    Set workspace directory for tools
     --help              Show help message

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
