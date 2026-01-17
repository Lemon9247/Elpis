"""
ReAct loop handler for Psyche.

This module handles user input processing with the ReAct (Reasoning + Acting) loop.
It parses tool calls from LLM responses and executes them.

This is core business logic, independent of the UI layer.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from loguru import logger

if TYPE_CHECKING:
    from psyche.mcp.client import ElpisClient
    from psyche.tools.tool_engine import ToolEngine

from psyche.memory.compaction import CompactionResult, ContextCompactor, Message, create_message


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

    # Whether reasoning mode is enabled (for parsing <reasoning> tags)
    reasoning_enabled: bool = True


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
    - Call ElpisClient for inference
    - Parse tool calls from response (```tool_call blocks)
    - Execute tools via ToolEngine
    - Loop until no more tool calls or max iterations reached
    - Stream tokens to callbacks for real-time UI updates
    - Handle interrupts gracefully

    Dependencies (to be injected):
    - elpis_client: For LLM inference (generate_stream)
    - tool_engine: For executing tool calls
    - compactor: For managing conversation context
    """

    def __init__(
        self,
        elpis_client: ElpisClient,
        tool_engine: ToolEngine,
        compactor: ContextCompactor,
        config: Optional[ReactConfig] = None,
        retrieve_memories_fn: Optional[Callable[[str], Any]] = None,
    ):
        """
        Initialize the ReAct handler.

        Args:
            elpis_client: Client for LLM inference
            tool_engine: Engine for executing tools
            compactor: Context compactor for managing messages
            config: Configuration options
            retrieve_memories_fn: Optional async function to retrieve relevant memories
        """
        self.client = elpis_client
        self.tool_engine = tool_engine
        self.compactor = compactor
        self.config = config or ReactConfig()
        self.retrieve_memories_fn = retrieve_memories_fn

        # Interrupt event for stopping generation
        self._interrupt_event = asyncio.Event()
        self._is_processing = False

        # Callback for compaction results
        self.on_compaction: Optional[Callable[[CompactionResult], Any]] = None

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
        1. Retrieves relevant memories (if available)
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
        self._is_processing = True
        self._interrupt_event.clear()

        try:
            # Verify connection before processing
            if not self.client.is_connected:
                logger.error("Elpis client disconnected, cannot process user input")
                if on_response:
                    on_response("[Error: Inference server disconnected]")
                return "[Error: Inference server disconnected]"

            logger.debug(f"Processing user input: {text[:50]}...")

            # Retrieve relevant memories if function is available
            if self.retrieve_memories_fn:
                try:
                    memory_context = await self.retrieve_memories_fn(text)
                    if memory_context:
                        memory_message = create_message(
                            "system",
                            f"[Relevant memories]\n{memory_context}"
                        )
                        compaction_result = self.compactor.add_message(memory_message)
                        if compaction_result and self.on_compaction:
                            await self.on_compaction(compaction_result)
                except Exception as e:
                    logger.warning(f"Failed to retrieve memories: {e}")

            # Add user message to context
            compaction_result = self.compactor.add_message(create_message("user", text))
            if compaction_result and self.on_compaction:
                await self.on_compaction(compaction_result)

            # Track tool results for importance scoring
            tool_results_collected: List[Dict[str, Any]] = []

            # ReAct loop - iterate until LLM provides final response without tools
            for iteration in range(self.config.max_tool_iterations):
                # Check for interrupt at the start of each iteration
                if self._interrupt_event.is_set():
                    self._interrupt_event.clear()
                    logger.info("ReAct loop interrupted by user before iteration")
                    if on_response:
                        on_response("[Interrupted]")
                    return "[Interrupted]"

                # Re-check connection before each iteration
                if not self.client.is_connected:
                    logger.error("Elpis client disconnected during processing")
                    if on_response:
                        on_response("[Error: Inference server disconnected]")
                    return "[Error: Inference server disconnected]"

                messages = self.compactor.get_api_messages()

                # Generate response with streaming (with timeout)
                response_tokens: List[str] = []
                interrupted = False

                try:
                    async with asyncio.timeout(self.config.generation_timeout):
                        async for token in self.client.generate_stream(
                            messages=messages,
                            emotional_modulation=self.config.emotional_modulation,
                        ):
                            # Check for interrupt request
                            if self._interrupt_event.is_set():
                                self._interrupt_event.clear()
                                logger.info("Generation interrupted by user")
                                interrupted = True
                                break

                            response_tokens.append(token)
                            # Stream token to UI callback
                            if on_token:
                                on_token(token)

                except asyncio.TimeoutError:
                    logger.error(f"Generation timed out after {self.config.generation_timeout}s")
                    if on_response:
                        partial = "".join(response_tokens)
                        on_response(partial + "\n\n[Generation timed out]")
                    return partial + "\n\n[Generation timed out]"
                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    if on_response:
                        on_response(f"[Error: {e}]")
                    return f"[Error: {e}]"

                # Handle interrupt - show partial response with marker
                if interrupted:
                    response_tokens.append("\n\n[Interrupted]")
                    response_text = "".join(response_tokens)

                    # Add partial response to context
                    compaction_result = self.compactor.add_message(create_message("assistant", response_text))
                    if compaction_result and self.on_compaction:
                        await self.on_compaction(compaction_result)

                    # Notify callback with interrupted response
                    if on_response:
                        on_response(response_text)
                    return response_text

                response_text = "".join(response_tokens)

                # Try to parse tool calls from the response
                tool_call = self.parse_tool_call(response_text)

                if tool_call:
                    # Found a tool call - execute it
                    logger.debug(f"Parsed tool call: {tool_call.get('name')}")

                    # Check for interrupt before tool execution
                    if self._interrupt_event.is_set():
                        self._interrupt_event.clear()
                        logger.info("Tool execution skipped due to interrupt")
                        if on_response:
                            on_response(response_text + "\n\n[Interrupted before tool execution]")
                        return response_text + "\n\n[Interrupted before tool execution]"

                    # Signal end of this generation (so UI can end stream before tool runs)
                    if on_response:
                        on_response(response_text)

                    # Add assistant's tool call to context
                    compaction_result = self.compactor.add_message(create_message("assistant", response_text))
                    if compaction_result and self.on_compaction:
                        await self.on_compaction(compaction_result)

                    # Execute the tool and collect result for importance scoring
                    tool_result = await self.execute_tool(tool_call, on_tool_call)
                    tool_results_collected.append({
                        "tool": tool_result.tool_name,
                        "success": tool_result.success,
                        "result": tool_result.result,
                    })

                    # Add tool result to context
                    result_str = json.dumps(tool_result.result, indent=2) if isinstance(tool_result.result, dict) else str(tool_result.result)
                    max_chars = self.config.max_tool_result_chars
                    if len(result_str) > max_chars:
                        result_str = result_str[:max_chars] + f"\n\n[... truncated, {len(result_str) - max_chars} chars omitted]"

                    compaction_result = self.compactor.add_message(create_message(
                        "user",
                        f"[Tool result for {tool_result.tool_name}]:\n{result_str}",
                    ))
                    if compaction_result and self.on_compaction:
                        await self.on_compaction(compaction_result)

                    # Continue loop to get next response
                    continue

                # No tool call - this is the final response
                # Parse for reasoning/thinking blocks if reasoning mode is enabled
                final_response = response_text
                if self.config.reasoning_enabled:
                    try:
                        from psyche.memory.reasoning import parse_reasoning
                        parsed = parse_reasoning(response_text)
                        if parsed.has_thinking and on_thought:
                            # Send reasoning to thought callback
                            from psyche.memory.server import ThoughtEvent
                            on_thought(ThoughtEvent(
                                content=parsed.thinking,
                                thought_type="reasoning",
                                triggered_by="response",
                            ))
                        # Use the cleaned response (without thinking tags)
                        final_response = parsed.response
                    except ImportError:
                        # reasoning module not available, use raw response
                        pass

                compaction_result = self.compactor.add_message(create_message("assistant", final_response))
                if compaction_result and self.on_compaction:
                    await self.on_compaction(compaction_result)

                # Notify callback (tokens already streamed, this signals completion)
                if on_response:
                    on_response(final_response)

                # Update emotional state based on interaction
                try:
                    await self._update_emotion_for_interaction(final_response)
                except Exception as e:
                    logger.debug(f"Failed to update emotion: {e}")

                return final_response

            # Max iterations reached
            logger.warning(f"Max tool iterations ({self.config.max_tool_iterations}) reached")
            if on_response:
                on_response("[Max tool iterations reached]")
            return "[Max tool iterations reached]"

        finally:
            self._is_processing = False

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
        # Look for tool_call code blocks
        pattern = r'```tool_call\s*\n?(.*?)\n?```'
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)

        if match:
            try:
                tool_json = match.group(1).strip()
                tool_call = json.loads(tool_json)

                # Validate structure
                if "name" in tool_call and "arguments" in tool_call:
                    return tool_call
                elif "name" in tool_call:
                    # Allow calls with no arguments
                    tool_call["arguments"] = {}
                    return tool_call
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {e}")

        # Also try to find JSON object with name/arguments at the start of response
        # (in case LLM doesn't use code block)
        if response_text.strip().startswith('{'):
            try:
                # Find the JSON object by counting braces
                brace_count = 0
                end_idx = 0
                for i, c in enumerate(response_text):
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break

                if end_idx > 0:
                    tool_json = response_text[:end_idx]
                    tool_call = json.loads(tool_json)
                    if "name" in tool_call:
                        if "arguments" not in tool_call:
                            tool_call["arguments"] = {}
                        return tool_call
            except json.JSONDecodeError:
                pass

        return None

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

        Args:
            tool_call: Dict with 'name' and 'arguments' keys
            on_tool_call: Optional callback for tool events

        Returns:
            ToolCallResult with execution details
        """
        tool_name = tool_call.get("name", "unknown")
        arguments = tool_call.get("arguments", {})

        logger.debug(f"Executing tool: {tool_name} with args: {arguments}")

        # Notify callback at start (result=None indicates start)
        if on_tool_call:
            on_tool_call(tool_name, arguments, None)

        try:
            # Convert to the format expected by tool engine
            formatted_call = {
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments,
                }
            }

            # Execute the tool
            result = await self.tool_engine.execute_tool_call(formatted_call)

            # Notify callback at end (with result)
            if on_tool_call:
                on_tool_call(tool_name, arguments, result)

            # Trigger emotion based on result
            success = result.get("success", True)
            try:
                if success:
                    await self.client.update_emotion("success", intensity=0.3)
                else:
                    await self.client.update_emotion("failure", intensity=0.5)
            except Exception as e:
                logger.debug(f"Failed to update emotion after tool: {e}")

            return ToolCallResult(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                success=success,
            )

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")

            # Notify callback with error
            error_result = {"success": False, "error": str(e)}
            if on_tool_call:
                on_tool_call(tool_name, arguments, error_result)

            # Trigger frustration emotion
            try:
                await self.client.update_emotion("frustration", intensity=0.5)
            except Exception:
                pass

            return ToolCallResult(
                tool_name=tool_name,
                arguments=arguments,
                result=error_result,
                success=False,
                error=str(e),
            )

    async def _update_emotion_for_interaction(self, content: str) -> None:
        """Update emotional state based on response text length."""
        content_length = len(content)

        if content_length > 500:
            await self.client.update_emotion("engagement", intensity=0.5)
        elif content_length < 50:
            await self.client.update_emotion("boredom", intensity=0.3)

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
        if self._is_processing:
            self._interrupt_event.set()
            logger.debug("Generation interrupt requested")
            return True
        return False

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag after handling."""
        self._interrupt_event.clear()

    @property
    def is_processing(self) -> bool:
        """Check if currently processing user input."""
        return self._is_processing
