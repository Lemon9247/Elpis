"""Command registry for Psyche TUI application."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Command:
    """Definition of a user command."""

    name: str
    aliases: List[str]  # e.g., ["h", "?"] for help
    description: str
    shortcut: Optional[str]  # keyboard shortcut if any


# Registry of all available commands
COMMANDS: Dict[str, Command] = {
    "help": Command("help", ["h", "?"], "Show available commands", None),
    "quit": Command("quit", ["q", "exit"], "Exit the application", "Ctrl+Q"),
    "clear": Command("clear", ["c", "cls"], "Clear the chat history", "Ctrl+L"),
    "status": Command("status", ["s"], "Show server status", None),
    "thoughts": Command("thoughts", ["t"], "Toggle thought panel visibility", "Ctrl+T"),
    "emotion": Command("emotion", ["e"], "Show current emotional state", None),
}


def get_command(name: str) -> Optional[Command]:
    """
    Look up a command by name or alias.

    Args:
        name: Command name or alias (without leading slash)

    Returns:
        Command if found, None otherwise
    """
    # Direct lookup
    if name in COMMANDS:
        return COMMANDS[name]

    # Search by alias
    for cmd in COMMANDS.values():
        if name in cmd.aliases:
            return cmd

    return None


def format_help_text() -> str:
    """
    Format the help text for display.

    Returns:
        Formatted help text showing all commands
    """
    lines = ["Available commands:"]

    for cmd in COMMANDS.values():
        # Format aliases
        alias_str = ", ".join(f"/{a}" for a in cmd.aliases)
        aliases = f" ({alias_str})" if cmd.aliases else ""

        # Format shortcut
        shortcut = f" [{cmd.shortcut}]" if cmd.shortcut else ""

        lines.append(f"  /{cmd.name}{aliases}{shortcut} - {cmd.description}")

    return "\n".join(lines)


def format_shortcut_help() -> str:
    """
    Format keyboard shortcuts for display.

    Returns:
        Formatted shortcuts text
    """
    shortcuts = []
    for cmd in COMMANDS.values():
        if cmd.shortcut:
            shortcuts.append(f"{cmd.shortcut}: {cmd.description}")

    return "Shortcuts: " + ", ".join(shortcuts)


def format_startup_hint() -> str:
    """
    Return a short hint shown at startup.

    Returns:
        Short startup hint text
    """
    return "Type /help or /h for available commands"
