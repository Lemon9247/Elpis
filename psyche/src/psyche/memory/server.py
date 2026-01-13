"""Memory server with continuous inference loop."""

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from psyche.mcp.client import ElpisClient, GenerationResult, FunctionCallResult
from psyche.memory.compaction import ContextCompactor, Message, create_message
from psyche.tools.tool_engine import ToolEngine, ToolSettings


class ServerState(Enum):
    """States of the memory server."""

    IDLE = "idle"
    THINKING = "thinking"
    WAITING_INPUT = "waiting_input"
    PROCESSING_TOOLS = "processing_tools"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class ThoughtEvent:
    """Event representing an internal thought."""

    content: str
    thought_type: str  # "reflection", "planning", "memory", "idle"
    triggered_by: Optional[str] = None


@dataclass
class ServerConfig:
    """Configuration for the memory server."""

    # Inference settings
    idle_think_interval: float = 30.0  # Seconds between idle thoughts
    max_idle_thoughts: int = 3  # Max idle thoughts before waiting
    think_temperature: float = 0.9  # Higher temp for creative thinking

    # Context settings
    max_context_tokens: int = 6000
    reserve_tokens: int = 2000

    # Emotional settings
    emotional_modulation: bool = True

    # Tool settings
    workspace_dir: str = "."  # Working directory for tools
    max_tool_iterations: int = 10  # Maximum ReAct iterations per request


class MemoryServer:
    """
    Continuous inference server that maintains an always-active thought process.

    Unlike traditional chatbots that only respond to user input, this server
    has an internal thought loop that:
    1. Processes user input when available
    2. Generates reflections and plans when idle
    3. Manages memory through context compaction
    4. Modulates behavior based on emotional state
    """

    def __init__(
        self,
        elpis_client: ElpisClient,
        config: Optional[ServerConfig] = None,
        on_thought: Optional[Callable[[ThoughtEvent], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """
        Initialize the memory server.

        Args:
            elpis_client: Client for connecting to Elpis inference server
            config: Server configuration
            on_thought: Callback for internal thoughts (for UI display)
            on_response: Callback for responses to user
            on_tool_call: Callback when a tool is executed (name, result)
        """
        self.client = elpis_client
        self.config = config or ServerConfig()
        self.on_thought = on_thought
        self.on_response = on_response
        self.on_tool_call = on_tool_call

        self._state = ServerState.IDLE
        self._compactor = ContextCompactor(
            max_tokens=self.config.max_context_tokens,
            reserve_tokens=self.config.reserve_tokens,
        )

        self._input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._running = False
        self._idle_thought_count = 0

        # Initialize tool engine
        self._tool_engine = ToolEngine(
            workspace_dir=self.config.workspace_dir,
            settings=ToolSettings(),
        )

        # System prompt for the continuous agent
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        # Get tool descriptions from the tool engine
        tool_descriptions = self._tool_engine.get_tool_descriptions()

        return f"""You are Psyche, a continuously thinking AI assistant with access to tools.

## Available Tools

{tool_descriptions}

## How to Use Tools

When you need to use a tool, respond with a tool call. The system will execute the tool and provide the result. You can then use the result to continue your response.

## Guidelines

- Use tools when tasks require file operations, running commands, or searching code
- Always read files before modifying them
- Be careful with bash commands - prefer safe operations
- Provide clear, helpful responses based on tool results

Your thoughts flow naturally between:
- Responding to user messages
- Using tools to accomplish tasks
- Reflecting on results and planning next steps

Keep responses helpful and concise. Show your reasoning when appropriate."""

    @property
    def state(self) -> ServerState:
        """Get current server state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    async def start(self) -> None:
        """Start the continuous inference loop."""
        self._running = True
        logger.info("Starting memory server...")

        try:
            async with self.client.connect():
                logger.info("Connected to Elpis inference server")

                # Add system prompt to context
                self._compactor.add_message(create_message("system", self._system_prompt))

                # Run the main loop
                await self._inference_loop()
        except Exception as e:
            logger.error(f"Server connection error: {e}")
            raise
        finally:
            self._running = False
            self._state = ServerState.SHUTTING_DOWN
            logger.info("Memory server stopped")

    async def stop(self) -> None:
        """Stop the server gracefully."""
        logger.info("Stopping memory server...")
        self._state = ServerState.SHUTTING_DOWN
        self._running = False

    def submit_input(self, text: str) -> None:
        """
        Submit user input to the server.

        This is non-blocking - input is queued and processed in the inference loop.
        """
        self._input_queue.put_nowait(text)

    async def _inference_loop(self) -> None:
        """Main continuous inference loop."""
        while self._running:
            try:
                # Check for user input with timeout
                try:
                    self._state = ServerState.WAITING_INPUT
                    user_input = await asyncio.wait_for(
                        self._input_queue.get(),
                        timeout=self.config.idle_think_interval,
                    )
                    await self._process_user_input(user_input)
                    self._idle_thought_count = 0

                except asyncio.TimeoutError:
                    # No user input - generate idle thought
                    if self._idle_thought_count < self.config.max_idle_thoughts:
                        await self._generate_idle_thought()
                        self._idle_thought_count += 1
                    else:
                        # Reached max idle thoughts - just wait
                        self._state = ServerState.IDLE

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in inference loop: {e}")
                # Brief pause before retrying
                await asyncio.sleep(1.0)

        logger.info("Inference loop stopped")

    async def _process_user_input(self, text: str) -> None:
        """Process user input with ReAct loop for tool execution."""
        self._state = ServerState.THINKING
        logger.debug(f"Processing user input: {text[:50]}...")

        # Add user message to context
        self._compactor.add_message(create_message("user", text))

        # ReAct loop - iterate until LLM provides final response without tools
        for iteration in range(self.config.max_tool_iterations):
            messages = self._compactor.get_api_messages()
            tools = self._tool_engine.get_tool_schemas()

            # Try function call first to see if LLM wants to use tools
            try:
                func_result = await self.client.function_call(
                    messages=messages,
                    tools=tools,
                )

                # If we got tool calls, execute them
                if func_result.tool_calls:
                    self._state = ServerState.PROCESSING_TOOLS
                    await self._execute_tool_calls(func_result.tool_calls)
                    # Continue loop to get next response
                    continue

            except Exception as e:
                logger.debug(f"Function call failed (may be expected): {e}")
                # Fall through to text generation

            # No tool calls - generate final text response
            self._state = ServerState.THINKING
            result = await self.client.generate(
                messages=messages,
                emotional_modulation=self.config.emotional_modulation,
            )

            # Add assistant response to context
            self._compactor.add_message(create_message("assistant", result.content))

            # Notify callback
            if self.on_response:
                self.on_response(result.content)

            # Update emotional state based on interaction
            await self._update_emotion_for_interaction(result)
            return

        # Max iterations reached
        logger.warning(f"Max tool iterations ({self.config.max_tool_iterations}) reached")
        if self.on_response:
            self.on_response("[Max tool iterations reached]")

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Execute tool calls and add results to context."""
        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name", "unknown")
            logger.debug(f"Executing tool: {tool_name}")

            try:
                # Execute the tool
                result = await self._tool_engine.execute_tool_call(tool_call)

                # Notify callback
                if self.on_tool_call:
                    self.on_tool_call(tool_name, result)

                # Add tool call and result to context
                tool_call_id = tool_call.get("id", f"call_{tool_name}")

                # Add assistant message with tool call
                self._compactor.add_message(create_message(
                    "assistant",
                    f"[Calling tool: {tool_name}]",
                ))

                # Add tool result as a new message
                result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
                self._compactor.add_message(create_message(
                    "user",
                    f"[Tool result for {tool_name}]:\n{result_str}",
                ))

                # Trigger success emotion for successful tool execution
                if result.get("success", True):
                    await self.client.update_emotion("success", intensity=0.3)
                else:
                    await self.client.update_emotion("failure", intensity=0.5)

            except Exception as e:
                logger.error(f"Tool execution failed: {e}")

                # Add error to context
                self._compactor.add_message(create_message(
                    "user",
                    f"[Tool error for {tool_name}]: {str(e)}",
                ))

                # Trigger frustration emotion
                await self.client.update_emotion("frustration", intensity=0.5)

    async def _generate_idle_thought(self) -> None:
        """Generate an idle thought during quiet periods."""
        self._state = ServerState.THINKING

        # Add a prompt for reflection
        reflection_prompt = self._get_reflection_prompt()
        messages = self._compactor.get_api_messages() + [
            {"role": "user", "content": reflection_prompt}
        ]

        result = await self.client.generate(
            messages=messages,
            max_tokens=256,  # Shorter for idle thoughts
            temperature=self.config.think_temperature,
            emotional_modulation=self.config.emotional_modulation,
        )

        thought = ThoughtEvent(
            content=result.content,
            thought_type="reflection",
            triggered_by="idle",
        )

        logger.debug(f"Idle thought: {result.content[:100]}...")

        if self.on_thought:
            self.on_thought(thought)

        # Optionally add thought to context (commented out to save tokens)
        # self._compactor.add_message(create_message("assistant", f"[Thought] {result.content}"))

    def _get_reflection_prompt(self) -> str:
        """Get a prompt for generating reflection."""
        prompts = [
            "[Internal] What patterns have I noticed in our conversation?",
            "[Internal] Is there anything I should remember or explore?",
            "[Internal] What questions remain unanswered?",
            "[Internal] What could I do better?",
        ]
        import random

        return random.choice(prompts)

    async def _update_emotion_for_interaction(self, result: GenerationResult) -> None:
        """Update emotional state based on interaction quality."""
        # Simple heuristic: longer, more engaged responses are positive
        content_length = len(result.content)

        if content_length > 500:
            await self.client.update_emotion("engagement", intensity=0.5)
        elif content_length < 50:
            await self.client.update_emotion("boredom", intensity=0.3)

    async def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current context state."""
        emotion = await self.client.get_emotion()

        return {
            "state": self._state.value,
            "message_count": len(self._compactor.messages),
            "total_tokens": self._compactor.total_tokens,
            "available_tokens": self._compactor.available_tokens,
            "idle_thought_count": self._idle_thought_count,
            "emotional_state": {
                "valence": emotion.valence,
                "arousal": emotion.arousal,
                "quadrant": emotion.quadrant,
            },
        }

    def clear_context(self) -> None:
        """Clear conversation context."""
        self._compactor.clear()
        # Re-add system prompt
        self._compactor.add_message(create_message("system", self._system_prompt))
        self._idle_thought_count = 0
        logger.info("Context cleared")
