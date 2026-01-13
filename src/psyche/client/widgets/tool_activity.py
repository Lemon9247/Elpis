"""Tool activity widget for displaying tool execution status."""

from dataclasses import dataclass, field
from typing import Optional

from textual.widgets import Static
from textual.reactive import reactive


@dataclass
class ToolExecution:
    """Represents a single tool execution."""

    name: str
    status: str = "running"  # "running", "complete", "error"
    result_preview: str = ""


class ToolActivity(Static):
    """Displays active and recent tool executions."""

    tools: reactive[list] = reactive(list, always_update=True)

    def __init__(self, *args, **kwargs):
        """Initialize with border title."""
        super().__init__(*args, **kwargs)
        self.border_title = "Tools"
        self._tool_list: list[ToolExecution] = []

    def add_tool_start(self, name: str) -> None:
        """
        Record tool execution start.

        Args:
            name: Name of the tool being executed
        """
        self._tool_list.append(ToolExecution(name=name, status="running"))
        self._render_tools()

    def update_tool_complete(self, name: str, result: Optional[dict]) -> None:
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
                tool.result_preview = str(result.get("result", ""))[:30] if result else ""
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
            lines.append(f"{icon} {tool.name}")

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
            lines.append(f"{icon} {tool.name}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear tool history."""
        self._tool_list = []
        self._render_tools()
