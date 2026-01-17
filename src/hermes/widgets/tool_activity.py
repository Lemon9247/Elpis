"""Tool activity widget for displaying tool execution status."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from textual.reactive import reactive
from textual.widgets import Static

from hermes.formatters.tool_formatter import ToolDisplayFormatter


@dataclass
class ToolExecution:
    """Represents a single tool execution."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"  # "running", "complete", "error"
    result: Optional[Dict[str, Any]] = None


class ToolActivity(Static):
    """Displays active and recent tool executions."""

    tools: reactive[list] = reactive(list, always_update=True)

    def __init__(self, *args, **kwargs):
        """Initialize with border title."""
        super().__init__(*args, **kwargs)
        self.border_title = "Tools"
        self._tool_list: list[ToolExecution] = []

    def add_tool_start(self, name: str, args: Optional[Dict[str, Any]] = None) -> None:
        """
        Record tool execution start.

        Args:
            name: Name of the tool being executed
            args: Arguments passed to the tool
        """
        self._tool_list.append(ToolExecution(name=name, args=args or {}, status="running"))
        self._render_tools()

    def update_tool_complete(self, name: str, result: Optional[Dict[str, Any]]) -> None:
        """
        Update tool status to complete.

        Args:
            name: Name of the tool
            result: Result dict from tool execution
        """
        # Find the running tool with this name
        for tool in reversed(self._tool_list):
            if tool.name == name and tool.status == "running":
                if result is None:
                    tool.status = "error"
                elif result.get("success", True):
                    tool.status = "complete"
                else:
                    tool.status = "error"
                tool.result = result
                break
        self._render_tools()

    def _render_tools(self) -> None:
        """Render tool list with status indicators."""
        lines = []
        # Show last 5 tools
        for tool in self._tool_list[-5:]:
            icon = {
                "running": "[yellow]...[/]",
                "complete": "[green]OK[/]",
                "error": "[red]!![/]",
            }.get(tool.status, "?")
            # Format display text using the formatter
            display_text = ToolDisplayFormatter.format_full(
                tool.name, tool.args, tool.result, tool.status
            )
            lines.append(f"{icon} {display_text}")

        self.update("\n".join(lines) or "[dim]No recent tools[/]")

    def render(self) -> str:
        """Default render."""
        if not self._tool_list:
            return "[dim]No recent tools[/]"

        lines = []
        for tool in self._tool_list[-5:]:
            icon = {
                "running": "[yellow]...[/]",
                "complete": "[green]OK[/]",
                "error": "[red]!![/]",
            }.get(tool.status, "?")
            # Format display text using the formatter
            display_text = ToolDisplayFormatter.format_full(
                tool.name, tool.args, tool.result, tool.status
            )
            lines.append(f"{icon} {display_text}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear tool history."""
        self._tool_list = []
        self._render_tools()
