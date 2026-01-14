=====
Tools
=====

Psyche includes a comprehensive tool system that allows the AI to interact with
the file system, execute commands, and search codebases. The tool engine provides
async execution, input validation, and safety controls.

Tool System Overview
--------------------

The tool system is built around three key components:

:class:`~psyche.tools.tool_engine.ToolEngine`
    Async tool execution orchestrator that manages tool registration, validation,
    and execution.

:class:`~psyche.tools.tool_definitions.ToolDefinition`
    Schema definition for tools including name, description, parameters, input
    model, and handler function.

**Tool Implementations**
    Individual tool classes (FileTools, BashTool, SearchTool, DirectoryTool)
    that implement the actual functionality.

Available Tools
---------------

read_file
^^^^^^^^^

Read contents of a file from the workspace.

**Parameters:**

+--------------+---------+--------------------------------------------------+
| Parameter    | Type    | Description                                      |
+==============+=========+==================================================+
| file_path    | string  | Path to file (relative to workspace or absolute) |
+--------------+---------+--------------------------------------------------+
| max_lines    | integer | Maximum lines to read (default: 2000)            |
+--------------+---------+--------------------------------------------------+

**Example:**

.. code-block:: json

    {
      "name": "read_file",
      "arguments": {
        "file_path": "src/main.py",
        "max_lines": 100
      }
    }

**Response:**

.. code-block:: json

    {
      "success": true,
      "content": "...",
      "line_count": 100,
      "total_lines": 250,
      "truncated": true,
      "file_path": "/path/to/workspace/src/main.py",
      "size_bytes": 5432
    }

create_file
^^^^^^^^^^^

Create a new file in the workspace. Fails if the file already exists.

**Parameters:**

+-------------+---------+--------------------------------------------------+
| Parameter   | Type    | Description                                      |
+=============+=========+==================================================+
| file_path   | string  | Path to file (relative to workspace or absolute) |
+-------------+---------+--------------------------------------------------+
| content     | string  | Content to write to the new file                 |
+-------------+---------+--------------------------------------------------+
| create_dirs | boolean | Create parent directories if needed (default: true) |
+-------------+---------+--------------------------------------------------+

**Example:**

.. code-block:: json

    {
      "name": "create_file",
      "arguments": {
        "file_path": "new_module.py",
        "content": "\"\"\"New module.\"\"\"\n\ndef hello():\n    pass\n"
      }
    }

edit_file
^^^^^^^^^

Edit an existing file by replacing ``old_string`` with ``new_string``. The
``old_string`` must match exactly and be unique in the file.

**Parameters:**

+------------+--------+--------------------------------------------------+
| Parameter  | Type   | Description                                      |
+============+========+==================================================+
| file_path  | string | Path to existing file                            |
+------------+--------+--------------------------------------------------+
| old_string | string | Exact text to find and replace (must be unique)  |
+------------+--------+--------------------------------------------------+
| new_string | string | Text to replace it with                          |
+------------+--------+--------------------------------------------------+

**Example:**

.. code-block:: json

    {
      "name": "edit_file",
      "arguments": {
        "file_path": "config.py",
        "old_string": "DEBUG = False",
        "new_string": "DEBUG = True"
      }
    }

**Notes:**

- The file must already exist - use ``create_file`` for new files
- Creates a backup file (``.bak``) before editing
- Fails if ``old_string`` appears multiple times

execute_bash
^^^^^^^^^^^^

Execute a bash command in the workspace directory.

**Parameters:**

+-----------+--------+---------------------------+
| Parameter | Type   | Description               |
+===========+========+===========================+
| command   | string | Bash command to execute   |
+-----------+--------+---------------------------+

**Example:**

.. code-block:: json

    {
      "name": "execute_bash",
      "arguments": {
        "command": "ls -la src/"
      }
    }

**Response:**

.. code-block:: json

    {
      "success": true,
      "stdout": "total 16\ndrwxr-xr-x ...",
      "stderr": "",
      "exit_code": 0,
      "command": "ls -la src/"
    }

**Safety:** Dangerous command patterns are blocked by default. See
`Safety Controls`_ for details.

search_codebase
^^^^^^^^^^^^^^^

Search for patterns in the codebase using regex (requires ripgrep).

**Parameters:**

+---------------+---------+-----------------------------------------------+
| Parameter     | Type    | Description                                   |
+===============+=========+===============================================+
| pattern       | string  | Regex pattern to search for                   |
+---------------+---------+-----------------------------------------------+
| file_glob     | string  | File glob pattern (e.g., ``*.py``, ``**/*.js``) |
+---------------+---------+-----------------------------------------------+
| context_lines | integer | Context lines around matches (default: 0)     |
+---------------+---------+-----------------------------------------------+

**Example:**

.. code-block:: json

    {
      "name": "search_codebase",
      "arguments": {
        "pattern": "def\\s+process_",
        "file_glob": "*.py",
        "context_lines": 2
      }
    }

**Response:**

.. code-block:: json

    {
      "success": true,
      "matches": [
        {
          "file": "src/processor.py",
          "line_number": 42,
          "line": "def process_data(input):",
          "match": {}
        }
      ],
      "match_count": 3,
      "pattern": "def\\s+process_",
      "file_glob": "*.py"
    }

list_directory
^^^^^^^^^^^^^^

List files and directories in the workspace.

**Parameters:**

+-----------+---------+--------------------------------------------------+
| Parameter | Type    | Description                                      |
+===========+=========+==================================================+
| dir_path  | string  | Directory path (default: ``.``)                  |
+-----------+---------+--------------------------------------------------+
| recursive | boolean | List subdirectories recursively (default: false) |
+-----------+---------+--------------------------------------------------+
| pattern   | string  | Glob pattern to filter results                   |
+-----------+---------+--------------------------------------------------+

**Example:**

.. code-block:: json

    {
      "name": "list_directory",
      "arguments": {
        "dir_path": "src",
        "recursive": true,
        "pattern": "*.py"
      }
    }

ReAct Loop Pattern
------------------

Psyche uses the ReAct (Reasoning + Acting) pattern for tool execution:

1. The AI explains what it wants to do and why
2. The AI emits a ``tool_call`` block in its response
3. Psyche executes the tool and returns the result
4. The AI reflects on the result and continues

**Tool Call Format:**

.. code-block:: text

    I need to check the contents of the configuration file.

    ```tool_call
    {"name": "read_file", "arguments": {"file_path": "config.py"}}
    ```

**Tool Result:**

The result is added to the conversation as a user message:

.. code-block:: text

    [Tool result for read_file]:
    {
      "success": true,
      "content": "DEBUG = False\n...",
      ...
    }

**Iteration Limits:**

.. code-block:: python

    max_tool_iterations: int = 10  # Maximum ReAct iterations per request

Safety Controls
---------------

Path Sanitization
^^^^^^^^^^^^^^^^^

All file paths are validated to prevent workspace escapes:

.. code-block:: python

    def sanitize_path(self, path: str) -> Path:
        """Ensure path is within workspace directory."""
        resolved = (self.workspace_dir / path).resolve()
        resolved.relative_to(self.workspace_dir)  # Raises if outside
        return resolved

Dangerous Command Blocking
^^^^^^^^^^^^^^^^^^^^^^^^^^

The bash tool blocks dangerous command patterns by default:

.. code-block:: python

    DANGEROUS_PATTERNS = [
        'rm -rf /',
        'rm -rf ~',
        'rm -rf *',
        'mkfs',
        ':(){:|:&};:',  # Fork bomb
        'dd if=/dev/zero',
        # ... and more
    ]

Input Validation
^^^^^^^^^^^^^^^^

All tool inputs are validated using Pydantic models:

.. code-block:: python

    class ReadFileInput(ToolInput):
        file_path: str = Field(description="Path to file")
        max_lines: Optional[int] = Field(default=2000, ge=0, le=100000)

        @field_validator('file_path')
        @classmethod
        def validate_path(cls, v: str) -> str:
            if '\x00' in v:
                raise ValueError("Null bytes not allowed")
            return v

Safe Idle Tools
---------------

During idle reflection, only safe read-only tools are allowed:

.. code-block:: python

    SAFE_IDLE_TOOLS: Set[str] = {"read_file", "list_directory", "search_codebase"}

Sensitive Path Protection
^^^^^^^^^^^^^^^^^^^^^^^^^

Sensitive paths are blocked during idle reflection:

.. code-block:: python

    SENSITIVE_PATH_PATTERNS: Set[str] = {
        ".ssh", ".gnupg", ".gpg", ".aws", ".azure", ".gcloud",
        ".config/gh", ".netrc", ".npmrc", ".pypirc",
        "id_rsa", "id_ed25519", "id_ecdsa", ".pem", ".key",
        "credentials", "secrets", "tokens", ".env",
    }

Rate Limiting
^^^^^^^^^^^^^

Idle tool use is rate-limited to prevent excessive filesystem access:

.. code-block:: python

    startup_warmup_seconds: float = 120.0   # No tools for first 2 minutes
    idle_tool_cooldown_seconds: float = 300.0  # 5 min between idle tool uses

Tool Configuration
------------------

Tool behavior can be configured through :class:`~psyche.tools.tool_engine.ToolSettings`:

.. code-block:: python

    @dataclass
    class ToolSettings:
        bash_timeout: int = 30              # Bash command timeout in seconds
        max_file_size: int = 1_000_000      # Maximum file size to read
        allowed_extensions: Optional[List[str]] = None  # File extension filter

And through :class:`~psyche.memory.server.ServerConfig`:

.. code-block:: python

    @dataclass
    class ServerConfig:
        workspace_dir: str = "."            # Working directory for tools
        max_tool_iterations: int = 10       # Max ReAct iterations
        max_tool_result_chars: int = 16000  # Truncate large results
        allow_idle_tools: bool = True       # Enable idle tool use
        max_idle_tool_iterations: int = 3   # Max idle tool calls
        max_idle_result_chars: int = 8000   # Truncate idle results

Extending the Tool System
-------------------------

To add a new tool:

1. Create an input model:

.. code-block:: python

    class MyToolInput(ToolInput):
        param1: str = Field(description="First parameter")
        param2: int = Field(default=10, description="Second parameter")

2. Create the tool implementation:

.. code-block:: python

    class MyTool:
        def __init__(self, workspace_dir: Path, settings: ToolSettings):
            self.workspace_dir = workspace_dir
            self.settings = settings

        async def execute(self, param1: str, param2: int) -> Dict[str, Any]:
            # Implementation
            return {"success": True, "result": "..."}

3. Register the tool in ``ToolEngine._register_tools()``:

.. code-block:: python

    self.tools["my_tool"] = ToolDefinition(
        name="my_tool",
        description="Description of my tool",
        parameters={
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."},
                "param2": {"type": "integer", "default": 10, "description": "..."},
            },
            "required": ["param1"],
        },
        input_model=MyToolInput,
        handler=my_tool.execute,
    )
