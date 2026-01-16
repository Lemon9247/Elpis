"""
ReAct loop handler for Psyche TUI.

This module handles user input processing with the ReAct (Reasoning + Acting) loop.
It parses tool calls from LLM responses and executes them.

Methods that will move from server.py (Wave 2 TUI Agent will implement):
- _process_user_input() (lines ~663-843) - Main ReAct loop
- _parse_tool_call() (lines ~850-909) - Parse tool calls from response
- _execute_parsed_tool_call() (lines ~911-977) - Execute tool and handle result

Wave 2 TUI Agent will implement these methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from psyche.core.context_manager import ContextManager
    from psyche.core.memory_handler import MemoryHandler
    from psyche.mcp.client import ElpisClient
    from psyche.tools.tool_engine import ToolEngine


@dataclass
class ReactConfig:
    """Configuration for ReAct handler.

    Mirrors relevant settings from ServerConfig in server.py.
    """
    # Maximum iterations per user request (prevents infinite loops)
    max_tool_iterations: int = 10

    # Truncate tool results to avoid context overflow
    max_tool_result_chars: int = 16000

    # Generation timeout in seconds
    generation_timeout: float = 120.0

    # Whether to use emotional modulation during inference
    emotional_modulation: bool = True


@dataclass
class ToolCallResult:
    """Result of a tool call execution.

    Used to track results for importance scoring and context updates.
    """
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class ReactHandler:
    """
    Handles ReAct loop for user input processing.

    The ReAct (Reasoning + Acting) loop allows the LLM to:
    1. Think about the user's request
    2. Optionally call tools to gather information or perform actions
    3. Continue reasoning with tool results
    4. Repeat until a final response is ready

    Responsibilities:
    - Parse user input
    - Call PsycheCore/ElpisClient for inference
    - Parse tool calls from response (```tool_call blocks)
    - Execute tools via ToolEngine
    - Loop until no more tool calls or max iterations reached
    - Stream tokens to callbacks for real-time UI updates
    - Handle interrupts gracefully

    Dependencies (to be injected):
    - elpis_client: For LLM inference (generate_stream)
    - tool_engine: For executing tool calls
    - context_manager: For managing conversation context
    - memory_handler: For storing important exchanges

    Key methods from server.py that will move here:
    - _process_user_input(): Main entry point, orchestrates the loop
    - _parse_tool_call(): Extracts tool call JSON from LLM response
    - _execute_parsed_tool_call(): Executes a tool and formats result
    """

    def __init__(
        self,
        elpis_client: ElpisClient,
        tool_engine: ToolEngine,
        context_manager: ContextManager,
        memory_handler: Optional[MemoryHandler] = None,
        config: Optional[ReactConfig] = None,
    ):
        """
        Initialize the ReAct handler.

        Args:
            elpis_client: Client for LLM inference
            tool_engine: Engine for executing tools
            context_manager: Manager for conversation context
            memory_handler: Optional handler for memory operations
            config: Configuration options
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    async def process_input(
        self,
        text: str,
        on_token: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any], Optional[Dict[str, Any]]], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
        on_thought: Optional[Callable[[Any], None]] = None,
    ) -> str:
        """
        Process user input through ReAct loop.

        This is the main entry point. It:
        1. Retrieves relevant memories (via memory_handler)
        2. Adds user message to context
        3. Generates LLM response with streaming
        4. Parses for tool calls
        5. If tool call found: execute, add result to context, loop back to 3
        6. If no tool call: return final response

        The loop continues until:
        - LLM provides a response without tool calls (success)
        - Max iterations reached (error)
        - Interrupt requested (cancelled)
        - Connection lost (error)

        Args:
            text: User input text
            on_token: Callback for streaming tokens (for real-time display)
            on_tool_call: Callback for tool execution events
                         (name, args, result) where result=None indicates start
            on_response: Callback when response segment is complete
            on_thought: Callback for reasoning/thinking blocks

        Returns:
            Final response text

        Raises:
            RuntimeError: If connection is lost during processing
            asyncio.CancelledError: If processing is interrupted
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    def parse_tool_call(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse tool call from LLM response.

        Looks for patterns like:
        ```tool_call
        {"name": "tool_name", "arguments": {...}}
        ```

        Also handles cases where LLM outputs raw JSON without code block.

        Args:
            response_text: The LLM's response text

        Returns:
            Parsed tool call dict with 'name' and 'arguments' keys,
            or None if no valid tool call found
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    async def execute_tool(
        self,
        tool_call: Dict[str, Any],
        on_tool_call: Optional[Callable[[str, Dict[str, Any], Optional[Dict[str, Any]]], None]] = None,
    ) -> ToolCallResult:
        """
        Execute a parsed tool call.

        Handles:
        - Formatting call for tool engine
        - Executing with error handling
        - Truncating large results
        - Updating emotional state based on success/failure
        - Adding result to context

        Args:
            tool_call: Dict with 'name' and 'arguments' keys
            on_tool_call: Optional callback for tool events

        Returns:
            ToolCallResult with execution details
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    def interrupt(self) -> bool:
        """
        Request interruption of current processing.

        Sets an internal flag that is checked during:
        - Token generation
        - Before tool execution
        - Between ReAct iterations

        Returns:
            True if interrupt was requested (was in interruptible state),
            False otherwise
        """
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag after handling."""
        raise NotImplementedError("Wave 2 TUI Agent will implement")

    @property
    def is_processing(self) -> bool:
        """Check if currently processing user input."""
        raise NotImplementedError("Wave 2 TUI Agent will implement")
