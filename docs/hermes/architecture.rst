============
Architecture
============

This document describes the internal architecture of the Hermes TUI client,
including its component structure, widget system, and tool execution model.

Module Overview
---------------

.. code-block:: text

    hermes/
      cli.py                # CLI entry point (hermes command)
      app.py                # Main Textual application
      app.tcss              # Textual CSS styling
      commands.py           # Slash command handling

      config/
        settings.py         # Pydantic settings classes

      handlers/
        psyche_client.py    # PsycheClient interface and RemotePsycheClient

      widgets/
        __init__.py         # Widget exports
        chat_view.py        # Message display with streaming
        user_input.py       # Input field with history
        sidebar.py          # Status, emotion, controls
        thought_panel.py    # Reasoning display
        tool_activity.py    # Tool execution visualization

      tools/
        __init__.py         # Tool exports
        tool_engine.py      # ToolEngine orchestrator
        tool_definitions.py # ToolDefinition, input models
        implementations/
          file_tools.py     # read_file, create_file, edit_file
          bash_tool.py      # execute_bash
          search_tool.py    # search_codebase
          directory_tool.py # list_directory

      formatters/
        tool_formatter.py   # Tool result formatting

Component Architecture
----------------------

.. code-block:: text

    +------------------------------------------------------------------+
    |                     Hermes Application (app.py)                   |
    |  +------------------+  +------------------+  +------------------+ |
    |  |    Widgets       |  |   Event Loop     |  |  Key Bindings    | |
    |  |  - ChatView      |  |  - Messages      |  |  - Ctrl+C        | |
    |  |  - UserInput     |  |  - Streaming     |  |  - Ctrl+L        | |
    |  |  - Sidebar       |  |  - Tool exec     |  |  - Ctrl+T        | |
    |  +--------+---------+  +--------+---------+  +------------------+ |
    +-----------|----------------------|--------------------------------+
                |                      |
                v                      v
    +------------------+    +------------------+
    | RemotePsycheClient|   |    ToolEngine    |
    | (HTTP to Psyche) |    | (Local execution)|
    +--------+---------+    +--------+---------+
             |                       |
             v                       v
    +------------------+    +------------------+
    |  Psyche Server   |    |  Local Tools     |
    |   (HTTP API)     |    |  - File I/O      |
    +------------------+    |  - Bash          |
                            |  - Search        |
                            +------------------+

Application Layer
-----------------

Hermes App
^^^^^^^^^^

**Module**: ``app.py``

The main Textual application managing UI and event flow:

.. code-block:: python

    class AppState(Enum):
        IDLE = "idle"
        PROCESSING = "processing"
        DISCONNECTED = "disconnected"

    class Hermes(App):
        BINDINGS = [
            Binding("ctrl+c", "interrupt_or_quit", "Stop/Quit"),
            Binding("ctrl+l", "clear", "Clear"),
            Binding("ctrl+t", "toggle_thoughts", "Thoughts"),
            Binding("escape", "focus_input", "Focus Input"),
        ]

        def __init__(
            self,
            client: Optional[PsycheClient] = None,
            tool_engine: Optional[ToolEngine] = None,
            workspace: Optional[Path] = None,
        ): ...

Key responsibilities:

- Widget composition and layout
- Event routing between widgets
- Message processing loop
- Tool execution coordination
- State management (idle/processing/disconnected)

Application States
^^^^^^^^^^^^^^^^^^

.. code-block:: text

    +-------+     user_submitted     +------------+
    | IDLE  | -------------------->  | PROCESSING |
    +-------+                        +------------+
        ^                                  |
        |      response_complete           |
        +----------------------------------+
        |
        |     connection_lost
        v
    +--------------+
    | DISCONNECTED |
    +--------------+

Widget System
-------------

All widgets inherit from Textual's base classes and communicate via messages.

ChatView
^^^^^^^^

**Module**: ``widgets/chat_view.py``

Scrollable message display with streaming support:

.. code-block:: python

    class ChatView(ScrollableContainer):
        async def add_message(
            self,
            role: str,
            content: str,
            stream: bool = False,
        ) -> None:
            """Add a message to the chat."""
            ...

        async def stream_token(self, token: str) -> None:
            """Append token to current streaming message."""
            ...

        async def finish_stream(self) -> None:
            """Finalize the current streaming message."""
            ...

Features:

- Markdown rendering via ``rich``
- Auto-scroll to bottom
- Streaming token accumulation
- Message role styling (user/assistant/system)

UserInput
^^^^^^^^^

**Module**: ``widgets/user_input.py``

Input field with command detection and history:

.. code-block:: python

    class UserSubmitted(Message):
        """Message sent when user submits input."""
        content: str
        is_command: bool

    class UserInput(Input):
        async def action_submit(self) -> None:
            """Handle Enter key."""
            content = self.value.strip()
            is_command = content.startswith("/")
            self.post_message(UserSubmitted(content, is_command))

Features:

- Command prefix detection (``/``)
- Input history (up/down arrows)
- Submission on Enter
- Focus management

Sidebar
^^^^^^^

**Module**: ``widgets/sidebar.py``

Status display with emotional state visualization:

.. code-block:: python

    class Sidebar(Static):
        def compose(self) -> ComposeResult:
            yield StatusDisplay()
            yield EmotionalStateDisplay()
            yield ControlsPanel()

    class EmotionalStateDisplay(Static):
        def update_state(
            self,
            valence: float,
            arousal: float,
            quadrant: str,
        ) -> None:
            """Update emotional state display."""
            ...

Components:

- Server connection status
- Context token usage
- Valence-arousal bars
- Quadrant indicator with color coding
- Trajectory information

ThoughtPanel
^^^^^^^^^^^^

**Module**: ``widgets/thought_panel.py``

Collapsible panel for internal reasoning:

.. code-block:: python

    class ThoughtPanel(Static):
        def add_thought(self, content: str, source: str = "reasoning") -> None:
            """Add thought to display."""
            ...

        def toggle(self) -> None:
            """Toggle panel visibility."""
            ...

Displays:

- Reasoning blocks (``<reasoning>`` tags)
- Idle thoughts (server introspection)
- Debug information (optional)

ToolActivity
^^^^^^^^^^^^

**Module**: ``widgets/tool_activity.py``

Tool execution visualization:

.. code-block:: python

    class ToolActivity(Static):
        def show_tool_call(
            self,
            tool_name: str,
            arguments: Dict[str, Any],
        ) -> None:
            """Display pending tool call."""
            ...

        def show_tool_result(
            self,
            tool_name: str,
            result: Dict[str, Any],
            success: bool,
        ) -> None:
            """Display tool result."""
            ...

Features:

- Expandable argument display
- Result truncation
- Success/failure indicators
- Execution timing

Client Layer
------------

PsycheClient Interface
^^^^^^^^^^^^^^^^^^^^^^

**Module**: ``handlers/psyche_client.py``

Abstract interface for Psyche connection:

.. code-block:: python

    class PsycheClient(ABC):
        @abstractmethod
        async def add_user_message(self, content: str) -> Optional[str]:
            """Add user message and get memory context."""
            ...

        @abstractmethod
        async def generate(
            self,
            max_tokens: int = 2048,
        ) -> Dict[str, Any]:
            """Generate response (non-streaming)."""
            ...

        @abstractmethod
        async def generate_stream(
            self,
            max_tokens: int = 2048,
        ) -> AsyncIterator[str]:
            """Generate response with streaming."""
            ...

        @abstractmethod
        async def get_emotional_state(self) -> Dict[str, Any]:
            """Get current emotional state."""
            ...

RemotePsycheClient
^^^^^^^^^^^^^^^^^^

HTTP implementation connecting to Psyche server:

.. code-block:: python

    class RemotePsycheClient(PsycheClient):
        def __init__(self, server_url: str = "http://127.0.0.1:8741"):
            self.server_url = server_url
            self.session: Optional[aiohttp.ClientSession] = None
            self.messages: List[Dict[str, str]] = []

        async def generate_stream(self, max_tokens: int = 2048) -> AsyncIterator[str]:
            """Stream tokens via SSE."""
            async with self.session.post(
                f"{self.server_url}/v1/chat/completions",
                json={"messages": self.messages, "stream": True},
            ) as response:
                async for line in response.content:
                    # Parse SSE data
                    yield token

Features:

- Async HTTP with aiohttp
- SSE streaming support
- Connection health checks
- Automatic reconnection

Tool Execution Layer
--------------------

ToolEngine
^^^^^^^^^^

**Module**: ``tools/tool_engine.py``

Async tool execution orchestrator:

.. code-block:: python

    @dataclass
    class ToolSettings:
        bash_timeout: int = 30
        max_file_size: int = 1_000_000
        allowed_extensions: Optional[List[str]] = None
        tool_timeout: float = 60.0

    class ToolEngine:
        def __init__(
            self,
            workspace_dir: Path,
            settings: Optional[ToolSettings] = None,
        ):
            self.workspace_dir = workspace_dir
            self.tools: Dict[str, ToolDefinition] = {}
            self._register_tools()

        async def execute_tool_call(
            self,
            tool_call: Dict[str, Any],
        ) -> Dict[str, Any]:
            """Execute a single tool call."""
            ...

        def get_tool_schemas(self) -> List[Dict[str, Any]]:
            """Get OpenAI-format tool schemas."""
            ...

ToolDefinition
^^^^^^^^^^^^^^

**Module**: ``tools/tool_definitions.py``

Schema and handler binding:

.. code-block:: python

    @dataclass
    class ToolDefinition:
        name: str
        description: str
        parameters: Dict[str, Any]  # JSON Schema
        input_model: Type[ToolInput]  # Pydantic model
        handler: Callable[..., Awaitable[Dict]]

    class ToolInput(BaseModel):
        """Base class for tool inputs with validation."""

        @field_validator('*', mode='before')
        @classmethod
        def validate_no_null_bytes(cls, v):
            if isinstance(v, str) and '\x00' in v:
                raise ValueError("Null bytes not allowed")
            return v

Tool Implementations
^^^^^^^^^^^^^^^^^^^^

**Module**: ``tools/implementations/``

Each tool is implemented as a class with an ``execute`` method:

.. code-block:: python

    # file_tools.py
    class FileTools:
        def __init__(self, workspace_dir: Path, settings: ToolSettings):
            self.workspace_dir = workspace_dir
            self.settings = settings

        def sanitize_path(self, path: str) -> Path:
            """Ensure path is within workspace."""
            resolved = (self.workspace_dir / path).resolve()
            resolved.relative_to(self.workspace_dir)
            return resolved

        async def read_file(self, file_path: str, max_lines: int = 2000) -> Dict:
            safe_path = self.sanitize_path(file_path)
            content = safe_path.read_text()
            return {"success": True, "content": content, ...}

Available tools:

+------------------+----------------------------------+
| Tool             | Implementation                   |
+==================+==================================+
| read_file        | FileTools.read_file              |
+------------------+----------------------------------+
| create_file      | FileTools.create_file            |
+------------------+----------------------------------+
| edit_file        | FileTools.edit_file              |
+------------------+----------------------------------+
| execute_bash     | BashTool.execute                 |
+------------------+----------------------------------+
| search_codebase  | SearchTool.search                |
+------------------+----------------------------------+
| list_directory   | DirectoryTool.list               |
+------------------+----------------------------------+

Message Processing Flow
-----------------------

.. code-block:: text

    1. User types message, presses Enter
              |
              v
    2. UserInput posts UserSubmitted message
              |
              v
    3. Hermes.on_user_submitted() handler
              |
              +-- Check if command (starts with /)
              |     +-- Yes: Execute command
              |     +-- No: Continue to generation
              |
              v
    4. Display user message in ChatView
              |
              v
    5. RemotePsycheClient.generate_stream()
              |
              +-- HTTP POST to Psyche server
              |
              +-- Stream SSE tokens
              |
              v
    6. For each token: ChatView.stream_token()
              |
              v
    7. Parse response for tool calls
              |
              +-- If tool calls found:
              |     +-- Display in ToolActivity
              |     +-- Execute via ToolEngine
              |     +-- Send results back to server
              |     +-- Continue generation
              |
              v
    8. ChatView.finish_stream()
              |
              v
    9. Update EmotionalStateDisplay
              |
              v
    10. Return to IDLE state

Tool Execution Flow
-------------------

.. code-block:: text

    1. Response contains tool_call JSON
              |
              v
    2. Parse tool name and arguments
              |
              v
    3. Validate arguments against Pydantic model
              |
              +-- Validation failed: Return error result
              |
              v
    4. Check tool safety (bash commands, paths)
              |
              +-- Blocked: Return security error
              |
              v
    5. Execute tool handler with timeout
              |
              v
    6. Format result
              |
              v
    7. Display in ToolActivity
              |
              v
    8. Send result back to Psyche server
              |
              v
    9. Server incorporates result, continues generation

Command System
--------------

**Module**: ``commands.py``

Slash commands are parsed and executed locally:

.. code-block:: python

    COMMANDS = {
        "help": Command(
            name="help",
            description="Show available commands",
            handler=handle_help,
        ),
        "status": Command(
            name="status",
            description="Show server status",
            handler=handle_status,
        ),
        # ...
    }

    def get_command(name: str) -> Optional[Command]:
        """Look up command by name."""
        return COMMANDS.get(name.lstrip("/"))

Available commands:

+----------------+----------------------------------+
| Command        | Description                      |
+================+==================================+
| ``/help``      | Show available commands          |
+----------------+----------------------------------+
| ``/status``    | Show server status               |
+----------------+----------------------------------+
| ``/clear``     | Clear conversation context       |
+----------------+----------------------------------+
| ``/emotion``   | Show emotional state             |
+----------------+----------------------------------+
| ``/thoughts``  | Toggle thought panel             |
+----------------+----------------------------------+
| ``/quit``      | Exit Hermes                      |
+----------------+----------------------------------+

Configuration
-------------

**Module**: ``config/settings.py``

.. code-block:: python

    @dataclass
    class HermesSettings:
        server_url: str = "http://127.0.0.1:8741"
        workspace: Optional[Path] = None
        tool_settings: ToolSettings = field(default_factory=ToolSettings)
        log_file: Optional[Path] = None
        debug: bool = False

Environment variables:

.. code-block:: bash

    HERMES_SERVER_URL=http://127.0.0.1:8741
    HERMES_WORKSPACE=/path/to/workspace
    HERMES_TOOL_SETTINGS__BASH_TIMEOUT=60
    HERMES_DEBUG=true

Styling
-------

**File**: ``app.tcss``

Textual CSS for widget styling:

.. code-block:: css

    ChatView {
        background: $surface;
        padding: 1;
    }

    .message-user {
        color: $primary;
    }

    .message-assistant {
        color: $text;
    }

    EmotionalStateDisplay {
        border: solid $primary;
    }

    .quadrant-excited { color: green; }
    .quadrant-frustrated { color: red; }
    .quadrant-calm { color: blue; }
    .quadrant-depleted { color: gray; }

Error Handling
--------------

Errors are handled at each layer:

**Widget Layer**
    Display errors in ChatView as system messages

**Client Layer**
    Connection failures trigger DISCONNECTED state

**Tool Layer**
    Execution errors returned as tool results

.. code-block:: python

    try:
        result = await tool.execute(**args)
    except ToolExecutionError as e:
        result = {"success": False, "error": str(e)}
    except asyncio.TimeoutError:
        result = {"success": False, "error": "Tool execution timed out"}

See Also
--------

- :doc:`index` - Hermes overview
- :doc:`api/index` - API reference
- :doc:`/psyche/tools` - Tool system documentation
- :doc:`/psyche/index` - Psyche server documentation
