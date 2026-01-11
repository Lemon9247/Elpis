"""Async tool execution orchestrator."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from elpis.config.settings import Settings
from elpis.tools.tool_definitions import (
    ExecuteBashInput,
    ListDirectoryInput,
    ReadFileInput,
    SearchCodebaseInput,
    ToolDefinition,
    WriteFileInput,
)
from elpis.tools.implementations.bash_tool import BashTool
from elpis.tools.implementations.directory_tool import DirectoryTool
from elpis.tools.implementations.file_tools import FileTools
from elpis.tools.implementations.search_tool import SearchTool
from elpis.utils.exceptions import ToolExecutionError


class ToolEngine:
    """Async tool execution orchestrator."""

    def __init__(self, workspace_dir: str, settings: Settings):
        """
        Initialize tool engine.

        Args:
            workspace_dir: Root workspace directory for tool operations
            settings: Global settings object
        """
        self.workspace_dir = Path(workspace_dir).resolve()
        self.settings = settings
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

        # Register write_file
        self.tools["write_file"] = ToolDefinition(
            name="write_file",
            description="Write content to a file in the workspace",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to file (relative to workspace or absolute)",
                    },
                    "content": {"type": "string", "description": "Content to write to file"},
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Create parent directories if they don't exist (default: true)",
                        "default": True,
                    },
                },
                "required": ["file_path", "content"],
            },
            input_model=WriteFileInput,
            handler=file_tools.write_file,
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
