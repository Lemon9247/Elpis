"""Formatter for tool display in the Psyche TUI.

Provides human-readable formatting for tool invocations and results,
transforming raw tool names and arguments into descriptive display text.
"""

from pathlib import Path
from typing import Any, Dict, Optional


class ToolDisplayFormatter:
    """
    Formats tool calls and results for human-readable display.

    Transforms generic tool names and arguments into descriptive text:
    - "read_file" with {"file_path": "src/main.py"} -> "Reading src/main.py"
    - "execute_bash" with {"command": "ls -la"} -> "$ ls -la"
    """

    # Templates for formatting tool starts (args -> display string)
    START_TEMPLATES = {
        "read_file": "Reading {file_path}",
        "create_file": "Creating {file_path}",
        "edit_file": "Editing {file_path}",
        "execute_bash": "$ {command}",
        "list_directory": "Listing {dir_path}",
        "search_codebase": "Searching: {pattern}",
        "recall_memory": "Recalling: {query}",
        "store_memory": "Storing memory",
    }

    # Maximum length for displayed arguments (prevents overflow)
    MAX_ARG_LENGTH = 40

    @classmethod
    def format_start(cls, tool_name: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Format tool display text at invocation start.

        Args:
            tool_name: Name of the tool being executed
            args: Arguments passed to the tool

        Returns:
            Human-readable description of the tool invocation

        Examples:
            >>> ToolDisplayFormatter.format_start("read_file", {"file_path": "src/main.py"})
            "Reading src/main.py"
            >>> ToolDisplayFormatter.format_start("execute_bash", {"command": "ls -la"})
            "$ ls -la"
            >>> ToolDisplayFormatter.format_start("unknown_tool", {})
            "unknown_tool"
        """
        if args is None:
            args = {}

        # Try to use template
        template = cls.START_TEMPLATES.get(tool_name)
        if template:
            try:
                # Process args for display
                display_args = cls._process_args_for_display(args)
                return template.format(**display_args)
            except KeyError:
                # Template requires args we don't have - fall back to name
                pass

        # Fallback: just return the tool name with underscores as spaces
        return tool_name.replace("_", " ").title()

    @classmethod
    def format_result(cls, tool_name: str, result: Optional[Dict[str, Any]]) -> str:
        """
        Format a brief summary of the tool result.

        Args:
            tool_name: Name of the tool that was executed
            result: Result dictionary from tool execution

        Returns:
            Brief summary string suitable for display next to the status

        Examples:
            >>> ToolDisplayFormatter.format_result("read_file", {"success": True, "line_count": 150})
            "(150 lines)"
            >>> ToolDisplayFormatter.format_result("execute_bash", {"success": True, "exit_code": 0})
            "(exit 0)"
            >>> ToolDisplayFormatter.format_result("read_file", {"success": False, "error": "File not found"})
            "(File not found)"
        """
        if result is None:
            return ""

        # Handle errors
        if not result.get("success", True):
            error = result.get("error", "Error")
            # Truncate long error messages
            if len(error) > 30:
                error = error[:27] + "..."
            return f"({error})"

        # Tool-specific result formatting
        if tool_name == "read_file":
            line_count = result.get("line_count", result.get("total_lines"))
            if line_count is not None:
                truncated = result.get("truncated", False)
                suffix = "+" if truncated else ""
                return f"({line_count}{suffix} lines)"
            return ""

        elif tool_name == "create_file":
            lines = result.get("lines_written", 0)
            return f"({lines} lines written)"

        elif tool_name == "edit_file":
            replaced = result.get("chars_replaced", 0)
            inserted = result.get("chars_inserted", 0)
            return f"(-{replaced}/+{inserted} chars)"

        elif tool_name == "execute_bash":
            exit_code = result.get("exit_code")
            if exit_code is not None:
                return f"(exit {exit_code})"
            return ""

        elif tool_name == "list_directory":
            file_count = result.get("file_count", 0)
            dir_count = result.get("directory_count", 0)
            return f"({file_count} files, {dir_count} dirs)"

        elif tool_name == "search_codebase":
            match_count = result.get("match_count", 0)
            return f"({match_count} matches)"

        elif tool_name == "recall_memory":
            count = result.get("count", 0)
            return f"({count} memories)"

        elif tool_name == "store_memory":
            return "(stored)"

        # Default: no summary for unknown tools
        return ""

    @classmethod
    def format_full(
        cls,
        tool_name: str,
        args: Optional[Dict[str, Any]],
        result: Optional[Dict[str, Any]],
        status: str = "complete"
    ) -> str:
        """
        Format a complete tool execution line for display.

        Combines the start text with result summary.

        Args:
            tool_name: Name of the tool
            args: Arguments passed to the tool
            result: Result from execution (None if still running)
            status: Current status ("running", "complete", "error")

        Returns:
            Complete formatted string for display

        Examples:
            >>> ToolDisplayFormatter.format_full("read_file", {"file_path": "main.py"}, {"success": True, "line_count": 50}, "complete")
            "Reading main.py (50 lines)"
        """
        start_text = cls.format_start(tool_name, args)

        if status == "running" or result is None:
            return start_text

        result_text = cls.format_result(tool_name, result)
        if result_text:
            return f"{start_text} {result_text}"
        return start_text

    @classmethod
    def _process_args_for_display(cls, args: Dict[str, Any]) -> Dict[str, str]:
        """
        Process arguments for display, truncating long values.

        Args:
            args: Raw arguments dictionary

        Returns:
            Dictionary with all values converted to truncated strings
        """
        display_args = {}
        for key, value in args.items():
            str_value = str(value)

            # For file paths, show just the filename if too long
            if key in ("file_path", "dir_path", "path"):
                if len(str_value) > cls.MAX_ARG_LENGTH:
                    # Try to show meaningful part of path
                    path = Path(str_value)
                    # Show last 2 components
                    parts = path.parts[-2:] if len(path.parts) > 1 else path.parts
                    str_value = "/".join(parts)
                    if len(str_value) > cls.MAX_ARG_LENGTH:
                        str_value = "..." + str_value[-(cls.MAX_ARG_LENGTH - 3):]
            elif len(str_value) > cls.MAX_ARG_LENGTH:
                str_value = str_value[:cls.MAX_ARG_LENGTH - 3] + "..."

            display_args[key] = str_value

        return display_args
