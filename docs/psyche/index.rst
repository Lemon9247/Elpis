======
Psyche
======

Psyche is the interactive TUI (Terminal User Interface) and REPL client for the Elpis
system. It provides a rich terminal-based interface for interacting with the Elpis
inference server, featuring streaming responses, emotional state display, and
continuous inference capabilities.

Overview
--------

Psyche serves as the user-facing client in the Elpis ecosystem, connecting to both
the Elpis inference server and Mnemosyne memory server via MCP (Model Context Protocol)
and providing:

- **Interactive TUI**: A rich terminal interface built with Textual, featuring
  chat history, emotional state visualization, tool activity display, and
  internal thought panels.

- **REPL Mode**: A command-line Read-Eval-Print Loop for simpler interactions
  using prompt_toolkit with command history support.

- **Continuous Inference**: Unlike traditional chatbots, Psyche maintains an
  always-active thought process that generates idle reflections when not
  processing user input.

- **Memory Consolidation**: Automatic memory consolidation during idle periods,
  promoting important short-term memories to long-term storage via Mnemosyne.

- **Tool Integration**: A comprehensive tool system for file operations, bash
  command execution, and codebase searching with safety controls.

Architecture
------------

Psyche is organized into several key components:

**Client Layer** (``psyche.client``)
    The user interface components including the Textual-based TUI application,
    REPL interface, and display management.

**Memory Server** (``psyche.memory``)
    The continuous inference server that manages conversation context, processes
    user input, generates responses, and handles idle thinking. Includes context
    compaction for managing token limits and automatic memory consolidation.

**Tool Engine** (``psyche.tools``)
    Async tool execution orchestrator with validated tool definitions and
    implementations for file operations, bash execution, directory listing,
    and codebase searching.

**MCP Clients** (``psyche.mcp``)
    Clients for connecting to backend servers:

    - ``ElpisClient``: Text generation with streaming and emotional modulation
    - ``MnemosyneClient``: Memory storage, search, and consolidation

Key Features
------------

Streaming Responses
    Real-time token streaming enables immediate feedback as the AI generates
    responses, providing a more responsive user experience.

Emotional State Display
    Visual representation of the inference server's emotional state (valence and
    arousal) with quadrant classification (excited, calm, frustrated, depleted).

Continuous Inference
    Background idle thinking generates reflections and explorations during quiet
    periods, with optional read-only tool access for workspace exploration.

Memory Consolidation
    Automatic memory consolidation during idle periods. Psyche connects to Mnemosyne
    and periodically checks if consolidation is recommended, then triggers clustering-
    based promotion of important memories to long-term storage.

Context Management
    Automatic context compaction keeps conversations within token limits using
    sliding window or summarization strategies.

Tool System
    Safe, sandboxed tool execution with Pydantic validation, timeout controls,
    and path sanitization to prevent workspace escapes.

Quick Start
-----------

Launch Psyche from the command line:

.. code-block:: bash

    # Start the TUI client (with memory consolidation)
    psyche

    # With debug logging
    psyche --debug

    # Specify workspace directory
    psyche --workspace /path/to/project

    # Disable memory consolidation
    psyche --no-consolidation

    # Custom Mnemosyne server command
    psyche --mnemosyne-command "mnemosyne-server --persist-dir ./custom/memory"

Within the TUI, type messages and press Enter to interact. Use slash commands
for control:

- ``/help`` - Show available commands
- ``/status`` - Display server status and context usage
- ``/clear`` - Clear conversation context
- ``/emotion`` - Show current emotional state
- ``/thoughts on|off`` - Toggle internal thought display
- ``/quit`` - Exit Psyche

Keyboard shortcuts:

- ``Ctrl+C`` - Quit
- ``Ctrl+L`` - Clear context
- ``Ctrl+T`` - Toggle thought panel
- ``Escape`` - Focus input

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   features
   tools

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
