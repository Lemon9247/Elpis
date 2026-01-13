"""Async tool execution orchestrator."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from psyche.tools.tool_definitions import (
    CreateFileInput,
    EditFileInput,
    ExecuteBashInput,
    ListDirectoryInput,
    ReadFileInput,
    SearchCodebaseInput,
    ToolDefinition,
)
from psyche.tools.implementations.bash_tool import BashTool
from psyche.tools.implementations.directory_tool import DirectoryTool
from psyche.tools.implementations.file_tools import FileTools
from psyche.tools.implementations.search_tool import SearchTool


class ToolExecutionError(Exception):
    """Exception raised during tool execution."""
    pass


@dataclass
class ToolSettings:
    """Settings for tool execution."""
    bash_timeout: int = 30
    max_file_size: int = 1_000_000
    allowed_extensions: Optional[List[str]] = None


class ToolEngine:
    """Async tool execution orchestrator."""

    def __init__(self, workspace_dir: str, settings: Optional[ToolSettings] = None):
        """
        Initialize tool engine.

        Args:
            workspace_dir: Root workspace directory for tool operations
            settings: Tool settings (uses defaults if not provided)
        """
        self.workspace_dir = Path(workspace_dir).resolve()
        self.settings = settings or ToolSettings()
        self.tools: Dict[str, ToolDefinition] = {}

        # Create workspace directory if it doesn't exist
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Register all tools
        self._register_tools()
        logger.info(f"ToolEngine initialized with {len(self.tools)} tools")

    def _register_tools(self) -> None:
        """Register all available tools with their schemas."""
        # Initialize tool implementations
        file_tools = FileTools(self.workspace_dir, self.settings)
        bash_tool = BashTool(self.workspace_dir, self.settings)
        search_tool = SearchTool(self.workspace_dir, self.settings)
        directory_tool = DirectoryTool(self.workspace_dir, self.settings)

        # Register read_file
        self.tools["read_file"] = ToolDefinition(
            name="read_file",
            description="Read contents of a file from the workspace",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to file (relative to workspace or absolute)",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default: 2000)",
                        "default": 2000,
                    },
                },
                "required": ["file_path"],
            },
            input_model=ReadFileInput,
            handler=file_tools.read_file,
        )

        # Register create_file
        self.tools["create_file"] = ToolDefinition(
            name="create_file",
            description="Create a new file in the workspace. Fails if file already exists.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to file (relative to workspace or absolute)",
                    },
                    "content": {"type": "string", "description": "Content to write to the new file"},
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Create parent directories if they don't exist (default: true)",
                        "default": True,
                    },
                },
                "required": ["file_path", "content"],
            },
            input_model=CreateFileInput,
            handler=file_tools.create_file,
        )

        # Register edit_file
        self.tools["edit_file"] = ToolDefinition(
            name="edit_file",
            description=(
                "Edit an EXISTING file by replacing old_string with new_string. "
                "IMPORTANT: The file must already exist - use create_file for new files. "
                "The old_string must match EXACTLY and be unique in the file. Creates a backup."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to existing file (relative to workspace or absolute)",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The EXACT text to find and replace (must exist and be unique in file)",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace it with",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
            input_model=EditFileInput,
            handler=file_tools.edit_file,
        )

        # Register execute_bash
        self.tools["execute_bash"] = ToolDefinition(
            name="execute_bash",
            description="Execute a bash command in the workspace directory",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute",
                    }
                },
                "required": ["command"],
            },
            input_model=ExecuteBashInput,
            handler=bash_tool.execute_bash,
        )

        # Register search_codebase
        self.tools["search_codebase"] = ToolDefinition(
            name="search_codebase",
            description="Search for patterns in codebase using regex (requires ripgrep)",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "File glob pattern (e.g., '*.py', '**/*.js')",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines around matches (default: 0)",
                        "default": 0,
                    },
                },
                "required": ["pattern"],
            },
            input_model=SearchCodebaseInput,
            handler=search_tool.search_codebase,
        )

        # Register list_directory
        self.tools["list_directory"] = ToolDefinition(
            name="list_directory",
            description="List files and directories in the workspace",
            parameters={
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "Directory path (relative to workspace or absolute, default: '.')",
                        "default": ".",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List subdirectories recursively (default: false)",
                        "default": False,
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter results",
                    },
                },
                "required": [],
            },
            input_model=ListDirectoryInput,
            handler=directory_tool.list_directory,
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Return OpenAI-compatible tool schemas.

        Returns:
            List of tool schemas in OpenAI function calling format
        """
        return [tool.to_openai_schema() for tool in self.tools.values()]

    def get_tool_descriptions(self) -> str:
        """
        Return human-readable descriptions of available tools.

        Returns:
            Formatted string describing all available tools
        """
        descriptions = []
        for name, tool in self.tools.items():
            schema = tool.to_openai_schema()
            func = schema.get("function", {})
            desc = func.get("description", "No description")
            params = func.get("parameters", {}).get("properties", {})

            param_list = []
            for param_name, param_info in params.items():
                param_desc = param_info.get("description", "")
                param_type = param_info.get("type", "any")
                param_list.append(f"  - {param_name} ({param_type}): {param_desc}")

            params_str = "\n".join(param_list) if param_list else "  (no parameters)"
            descriptions.append(f"### {name}\n{desc}\nParameters:\n{params_str}")

        return "\n\n".join(descriptions)

    async def execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single tool call from LLM.

        Args:
            tool_call: Tool call dictionary from LLM with function name and arguments

        Returns:
            Dictionary with tool execution results
        """
        start_time = time.time()

        try:
            # Extract tool name and arguments
            function_data = tool_call.get("function", {})
            tool_name = function_data.get("name")
            args_json = function_data.get("arguments", "{}")

            if not tool_name:
                raise ToolExecutionError("Missing tool name in function call")

            if tool_name not in self.tools:
                raise ToolExecutionError(f"Unknown tool: {tool_name}")

            # Parse arguments
            try:
                args = json.loads(args_json) if isinstance(args_json, str) else args_json
            except json.JSONDecodeError as e:
                raise ToolExecutionError(f"Invalid JSON arguments: {e}") from e

            # Get tool definition
            tool_def = self.tools[tool_name]

            # Validate arguments with Pydantic model
            validated_args = tool_def.input_model(**args)

            logger.debug(f"Executing tool: {tool_name} with args: {args}")

            # Execute tool (async)
            result = await tool_def.handler(**validated_args.model_dump())

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Tool {tool_name} executed in {duration_ms:.2f}ms")

            return {
                "tool_call_id": tool_call.get("id"),
                "success": result.get("success", False),
                "result": result,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.exception(f"Tool execution failed: {e}")
            duration_ms = (time.time() - start_time) * 1000

            return {
                "tool_call_id": tool_call.get("id"),
                "success": False,
                "result": {"success": False, "error": str(e)},
                "duration_ms": duration_ms,
            }

    async def execute_multiple_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tool calls concurrently.

        Args:
            tool_calls: List of tool call specifications from LLM

        Returns:
            List of execution results in the same order as inputs
        """
        tasks = [self.execute_tool_call(call) for call in tool_calls]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def sanitize_path(self, path: str) -> Path:
        """
        Sanitize and validate a path is within the workspace.

        Args:
            path: Relative or absolute path string

        Returns:
            Resolved Path object guaranteed within workspace

        Raises:
            ToolExecutionError: If path escapes workspace directory
        """
        path_obj = Path(path)

        if path_obj.is_absolute():
            resolved = path_obj.resolve()
        else:
            resolved = (self.workspace_dir / path_obj).resolve()

        # Check path is within workspace
        try:
            resolved.relative_to(self.workspace_dir)
        except ValueError:
            raise ToolExecutionError(
                f"Path '{path}' escapes workspace directory"
            )

        return resolved
