"""
Idle thinking handler for Psyche TUI.

This module handles autonomous "dreaming" when the user is idle.
It generates reflective thoughts and manages memory consolidation.

Methods that will move from server.py (Wave 2 TUI Agent will implement):
- _generate_idle_thought() (lines ~1104-1276) - Generate idle thoughts with tool use
- _get_reflection_prompt() (lines ~1278-1313) - Create reflection prompts
- _can_start_idle_thinking() (lines ~979-1001) - Check if can start
- _can_use_idle_tools() (lines ~1003-1035) - Rate limit idle tools
- _validate_idle_tool_call() (lines ~1078-1102) - Validate tool call
- _is_safe_idle_path() (lines ~1037-1076) - Path security checking
- _maybe_consolidate_memories() (lines ~1416-1477) - Memory consolidation

Wave 2 TUI Agent will implement these methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional, Set

if TYPE_CHECKING:
    from psyche.core.context_manager import ContextManager
    from psyche.core.memory_handler import MemoryHandler
    from psyche.mcp.client import ElpisClient, MnemosyneClient
    from psyche.tools.tool_engine import ToolEngine


# Safe tools that can be used during idle thinking (read-only operations)
# These tools cannot modify the system or leak sensitive data
SAFE_IDLE_TOOLS: FrozenSet[str] = frozenset({
    "read_file",
    "list_directory",
    "search_codebase",
    "recall_memory",
})

# Sensitive path patterns that should NEVER be accessed during idle reflection
# These protect credentials, keys, and other sensitive data
SENSITIVE_PATH_PATTERNS: FrozenSet[str] = frozenset({
    ".ssh",
    ".gnupg",
    ".gpg",
    ".aws",
    ".azure",
    ".gcloud",
    ".config/gh",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    ".pem",
    ".key",
    "credentials",
    "secrets",
    "tokens",
    ".env",
})


@dataclass
class IdleConfig:
    """Configuration for idle handler.

    Mirrors relevant settings from ServerConfig in server.py.
    """
    # Time to wait after user interaction before starting idle thinking
    post_interaction_delay: float = 60.0

    # Minimum time between idle tool uses (rate limiting)
    idle_tool_cooldown_seconds: float = 300.0

    # Time after startup before tools are allowed in idle mode
    startup_warmup_seconds: float = 120.0

    # Maximum tool iterations per idle thought
    max_idle_tool_iterations: int = 3

    # Truncate tool results to this size
    max_idle_result_chars: int = 8000

    # Temperature for reflection generation (lower = less hallucination)
    think_temperature: float = 0.7

    # Generation timeout in seconds
    generation_timeout: float = 120.0

    # Whether to allow tool use during idle reflection
    allow_idle_tools: bool = True

    # Whether to use emotional modulation
    emotional_modulation: bool = True

    # Workspace directory (for path validation)
    workspace_dir: str = "."

    # Memory consolidation settings
    enable_consolidation: bool = True
    consolidation_check_interval: float = 300.0  # Check every 5 minutes
    consolidation_importance_threshold: float = 0.6
    consolidation_similarity_threshold: float = 0.85


@dataclass
class ThoughtEvent:
    """Event representing an internal thought.

    Mirrors ThoughtEvent from server.py.
    """
    content: str
    thought_type: str  # "reflection", "planning", "memory", "idle"
    triggered_by: Optional[str] = None


class IdleHandler:
    """
    Handles idle thinking and memory consolidation.

    When the user is idle (no input for a configured period), this handler
    generates autonomous "reflections" - thoughts about the workspace,
    codebase, or conversation. These reflections can use read-only tools
    to explore the environment.

    Safety constraints:
    - Only read-only tools allowed (no bash, no writes)
    - Sensitive paths are blocked
    - Rate limiting prevents excessive tool use
    - Warmup period after startup before tools are enabled

    Memory consolidation:
    - Periodically checks if consolidation is needed
    - Promotes important short-term memories to long-term storage
    - Clusters similar memories together

    Responsibilities:
    - Generate reflective thoughts when user is idle
    - Manage safe tool usage during idle (read-only only)
    - Validate paths and tool calls for security
    - Trigger memory consolidation periodically
    - Respect rate limits and safety constraints
    - Stream thinking tokens to UI callback

    Dependencies (to be injected):
    - elpis_client: For LLM inference
    - tool_engine: For executing safe tools
    - context_manager: For getting conversation context
    - memory_handler: For consolidation operations
    - mnemosyne_client: For memory storage/retrieval
    """

    def __init__(
        self,
        elpis_client: ElpisClient,
        context_manager: ContextManager,
        tool_engine: Optional[ToolEngine] = None,
        memory_handler: Optional[MemoryHandler] = None,
        mnemosyne_client: Optional[MnemosyneClient] = None,
        config: Optional[IdleConfig] = None,
    ):
        """
        Initialize the idle handler.

        Args:
            elpis_client: Client for LLM inference
            context_manager: Manager for conversation context
            tool_engine: Optional engine for executing tools
            memory_handler: Optional handler for memory operations
            mnemosyne_client: Optional client for Mnemosyne memory server
            config: Configuration options
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    def can_start_thinking(self) -> bool:
        """
        Check if idle thinking can begin.

        Idle thinking is delayed after user interactions to avoid
        appearing to "continue speaking" without accepting input.

        Returns:
            True if enough time has passed since last user interaction,
            False otherwise
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    def can_use_tools(self) -> bool:
        """
        Check if tools can be used during idle (rate limiting).

        Tools are restricted during:
        1. Startup warmup period (first N seconds after server start)
        2. Cooldown period after last idle tool use

        Returns:
            True if idle tools can be used, False otherwise
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    async def generate_thought(
        self,
        on_token: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any], Optional[Dict[str, Any]]], None]] = None,
        on_thought: Optional[Callable[[ThoughtEvent], None]] = None,
    ) -> Optional[str]:
        """
        Generate an idle thought.

        This is the main entry point for idle thinking. It:
        1. Gets a reflection prompt
        2. Generates a response with streaming
        3. Optionally executes safe tools if needed
        4. Returns the final thought

        The process can be interrupted by user input.

        Args:
            on_token: Callback for streaming tokens (for thinking panel)
            on_tool_call: Callback for tool execution events
            on_thought: Callback for completed thought events

        Returns:
            Generated thought text, or None if interrupted/failed
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    def get_reflection_prompt(self) -> str:
        """
        Get a reflection prompt for idle thinking.

        Returns one of several prompts that encourage exploration
        and curiosity about the workspace/codebase.

        The prompt includes:
        - Instructions that this is private thinking (not shown to user)
        - Available tools for exploration
        - Guidance on using actual tools vs hallucinating

        Returns:
            A reflection prompt string
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    def validate_tool_call(self, tool_call: Dict[str, Any]) -> Optional[str]:
        """
        Validate that a tool call is safe for idle mode.

        Checks:
        1. Tool is in the safe list (read-only operations)
        2. Any path arguments are safe (not sensitive, within workspace)

        Args:
            tool_call: The parsed tool call dict

        Returns:
            Error message if invalid, None if valid
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    def is_safe_path(self, path: str) -> bool:
        """
        Check if a path is safe for idle tool access.

        A path is safe if:
        1. It doesn't match any sensitive patterns
        2. It doesn't use parent directory traversal (..)
        3. It's within the workspace directory

        Args:
            path: The path to check

        Returns:
            True if path is safe, False otherwise
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    async def maybe_consolidate(self) -> bool:
        """
        Check and run memory consolidation if needed.

        Called during idle periods to:
        1. Check if enough time has passed since last consolidation
        2. Ask Mnemosyne if consolidation is recommended
        3. Run consolidation if needed

        Returns:
            True if consolidation was run, False otherwise
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    def record_user_interaction(self) -> None:
        """
        Record that a user interaction occurred.

        Called when user submits input to reset the idle timer.
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    def record_tool_use(self) -> None:
        """
        Record that an idle tool was used.

        Called after successful idle tool execution to update rate limiting.
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    def interrupt(self) -> bool:
        """
        Request interruption of current idle thinking.

        Returns:
            True if interrupt was requested, False otherwise
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag after handling."""
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    @property
    def is_thinking(self) -> bool:
        """Check if currently generating idle thoughts."""
        raise NotImplementedError("Wave 2 TUI Agent will implement")
