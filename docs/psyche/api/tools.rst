============
Tools Module
============

The tools module provides async tool execution with input validation and safety
controls.

hermes.tools.tool_engine
------------------------

Tool Engine
^^^^^^^^^^^

The tool engine orchestrates tool execution, managing registration, validation,
and async execution.

.. automodule:: hermes.tools.tool_engine
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

    from hermes.tools.tool_engine import ToolEngine, ToolSettings

    engine = ToolEngine(
        workspace_dir="/path/to/workspace",
        settings=ToolSettings(bash_timeout=60),
    )

    # Execute a tool call
    result = await engine.execute_tool_call({
        "function": {
            "name": "read_file",
            "arguments": '{"file_path": "README.md"}'
        }
    })

    if result["success"]:
        print(result["result"]["content"])

hermes.tools.tool_definitions
-----------------------------

Tool Definitions
^^^^^^^^^^^^^^^^

Tool definitions provide schemas and input validation models.

.. automodule:: hermes.tools.tool_definitions
   :members:
   :undoc-members:
   :show-inheritance:

Input Models
^^^^^^^^^^^^

Each tool has a corresponding Pydantic input model:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Description
   * - :class:`~hermes.tools.tool_definitions.ReadFileInput`
     - Input for read_file tool
   * - :class:`~hermes.tools.tool_definitions.CreateFileInput`
     - Input for create_file tool
   * - :class:`~hermes.tools.tool_definitions.EditFileInput`
     - Input for edit_file tool
   * - :class:`~hermes.tools.tool_definitions.ExecuteBashInput`
     - Input for execute_bash tool
   * - :class:`~hermes.tools.tool_definitions.SearchCodebaseInput`
     - Input for search_codebase tool
   * - :class:`~hermes.tools.tool_definitions.ListDirectoryInput`
     - Input for list_directory tool

Tool Implementations
--------------------

File Tools
^^^^^^^^^^

.. automodule:: hermes.tools.implementations.file_tools
   :members:
   :undoc-members:
   :show-inheritance:

Bash Tool
^^^^^^^^^

.. automodule:: hermes.tools.implementations.bash_tool
   :members:
   :undoc-members:
   :show-inheritance:

Search Tool
^^^^^^^^^^^

.. automodule:: hermes.tools.implementations.search_tool
   :members:
   :undoc-members:
   :show-inheritance:

Directory Tool
^^^^^^^^^^^^^^

.. automodule:: hermes.tools.implementations.directory_tool
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
----------

.. autoexception:: hermes.tools.tool_engine.ToolExecutionError
   :members:
   :show-inheritance:

.. autoexception:: hermes.tools.implementations.file_tools.PathSafetyError
   :members:
   :show-inheritance:

.. autoexception:: hermes.tools.implementations.bash_tool.CommandSafetyError
   :members:
   :show-inheritance:
