"""Memory server with continuous inference loop."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from loguru import logger

from psyche.mcp.client import ElpisClient, GenerationResult, FunctionCallResult
from psyche.memory.compaction import ContextCompactor, Message, create_message
from psyche.tools.tool_engine import ToolEngine, ToolSettings


# Tools that are safe for autonomous idle reflection (read-only)
SAFE_IDLE_TOOLS: Set[str] = {"read_file", "list_directory", "search_codebase"}

# Sensitive paths that should never be accessed during idle reflection
SENSITIVE_PATH_PATTERNS: Set[str] = {
    ".ssh", ".gnupg", ".gpg", ".aws", ".azure", ".gcloud",
    ".config/gh", ".netrc", ".npmrc", ".pypirc",
    "id_rsa", "id_ed25519", "id_ecdsa", ".pem", ".key",
    "credentials", "secrets", "tokens", ".env",
}


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
    think_temperature: float = 0.7  # Moderate temp for reflection (lower = less hallucination)

    # Context settings
    max_context_tokens: int = 6000
    reserve_tokens: int = 2000

    # Emotional settings
    emotional_modulation: bool = True

    # Tool settings
    workspace_dir: str = "."  # Working directory for tools
    max_tool_iterations: int = 10  # Maximum ReAct iterations per request
    max_tool_result_chars: int = 8000  # Truncate tool results to avoid context overflow

    # Idle reflection settings
    allow_idle_tools: bool = True  # Allow read-only tools during reflection
    max_idle_tool_iterations: int = 3  # Max tool calls per reflection
    max_idle_result_chars: int = 4000  # Truncate tool results to this size


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

When you need to use a tool, respond with ONLY a JSON tool call in this exact format:
```tool_call
{{"name": "tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}
```

For example, to list files:
```tool_call
{{"name": "list_directory", "arguments": {{"dir_path": "."}}}}
```

To run a bash command:
```tool_call
{{"name": "execute_bash", "arguments": {{"command": "ls -la"}}}}
```

To read a file:
```tool_call
{{"name": "read_file", "arguments": {{"file_path": "example.txt"}}}}
```

IMPORTANT: When using a tool, respond with ONLY the tool_call block, nothing else. The system will execute the tool and show you the result. Then you can provide your final response.

## Guidelines

- Use tools when tasks require file operations, running commands, or searching code
- Always read files before modifying them
- Be careful with bash commands - prefer safe operations
- After receiving tool results, provide a helpful summary for the user

Keep responses helpful and concise."""

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

                except asyncio.TimeoutError:
                    # No user input - generate idle thought (no limit)
                    await self._generate_idle_thought()

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

            # Generate response
            self._state = ServerState.THINKING
            result = await self.client.generate(
                messages=messages,
                emotional_modulation=self.config.emotional_modulation,
            )

            response_text = result.content

            # Try to parse tool calls from the response
            tool_call = self._parse_tool_call(response_text)

            if tool_call:
                # Found a tool call - execute it
                self._state = ServerState.PROCESSING_TOOLS
                logger.debug(f"Parsed tool call: {tool_call.get('name')}")

                # Add assistant's tool call to context
                self._compactor.add_message(create_message("assistant", response_text))

                # Execute the tool
                await self._execute_parsed_tool_call(tool_call)

                # Continue loop to get next response
                continue

            # No tool call - this is the final response
            self._compactor.add_message(create_message("assistant", response_text))

            # Notify callback
            if self.on_response:
                self.on_response(response_text)

            # Update emotional state based on interaction
            await self._update_emotion_for_interaction(result)
            return

        # Max iterations reached
        logger.warning(f"Max tool iterations ({self.config.max_tool_iterations}) reached")
        if self.on_response:
            self.on_response("[Max tool iterations reached]")

    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse a tool call from the LLM's text response.

        Looks for patterns like:
        ```tool_call
        {"name": "tool_name", "arguments": {...}}
        ```

        Returns:
            Parsed tool call dict or None if no tool call found
        """
        import re

        # Look for tool_call code blocks
        pattern = r'```tool_call\s*\n?(.*?)\n?```'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

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
        if text.strip().startswith('{'):
            try:
                # Find the JSON object
                brace_count = 0
                end_idx = 0
                for i, c in enumerate(text):
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break

                if end_idx > 0:
                    tool_json = text[:end_idx]
                    tool_call = json.loads(tool_json)
                    if "name" in tool_call:
                        if "arguments" not in tool_call:
                            tool_call["arguments"] = {}
                        return tool_call
            except json.JSONDecodeError:
                pass

        return None

    async def _execute_parsed_tool_call(self, tool_call: Dict[str, Any]) -> None:
        """Execute a parsed tool call and add result to context."""
        tool_name = tool_call.get("name", "unknown")
        arguments = tool_call.get("arguments", {})

        logger.debug(f"Executing tool: {tool_name} with args: {arguments}")

        try:
            # Convert to the format expected by tool engine
            formatted_call = {
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments,
                }
            }

            # Execute the tool
            result = await self._tool_engine.execute_tool_call(formatted_call)

            # Notify callback
            if self.on_tool_call:
                self.on_tool_call(tool_name, result)

            # Add tool result to context (truncated to avoid context overflow)
            result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
            max_chars = self.config.max_tool_result_chars
            if len(result_str) > max_chars:
                result_str = result_str[:max_chars] + f"\n\n[... truncated, {len(result_str) - max_chars} chars omitted]"

            self._compactor.add_message(create_message(
                "user",
                f"[Tool result for {tool_name}]:\n{result_str}",
            ))

            # Trigger emotion based on result
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

    def _is_safe_idle_path(self, path: str) -> bool:
        """
        Check if a path is safe for idle reflection access.

        Paths must be within workspace and not match sensitive patterns.

        Args:
            path: The path to check

        Returns:
            True if path is safe, False otherwise
        """
        # Normalize the path
        path_lower = path.lower()

        # Check for sensitive patterns
        for pattern in SENSITIVE_PATH_PATTERNS:
            if pattern in path_lower:
                logger.warning(f"Blocked sensitive path in idle reflection: {path}")
                return False

        # Check for parent directory traversal
        if ".." in path:
            logger.warning(f"Blocked parent traversal in idle reflection: {path}")
            return False

        # Verify path is within workspace
        try:
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = (Path(self.config.workspace_dir) / path).resolve()
            else:
                resolved = resolved.resolve()

            workspace = Path(self.config.workspace_dir).resolve()
            resolved.relative_to(workspace)
            return True
        except ValueError:
            logger.warning(f"Path escapes workspace in idle reflection: {path}")
            return False

    def _validate_idle_tool_call(self, tool_call: Dict[str, Any]) -> Optional[str]:
        """
        Validate a tool call for idle reflection mode.

        Args:
            tool_call: The parsed tool call

        Returns:
            Error message if invalid, None if valid
        """
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        # Check if tool is in safe list
        if tool_name not in SAFE_IDLE_TOOLS:
            return f"Tool '{tool_name}' not allowed during idle reflection"

        # Check path arguments for safety
        path_args = ["file_path", "dir_path", "path"]
        for arg in path_args:
            if arg in arguments:
                if not self._is_safe_idle_path(arguments[arg]):
                    return f"Path '{arguments[arg]}' not allowed during idle reflection"

        return None

    async def _generate_idle_thought(self) -> None:
        """Generate an idle thought during quiet periods with optional tool use."""
        self._state = ServerState.THINKING

        # Add a prompt for reflection
        reflection_prompt = self._get_reflection_prompt()
        reflection_messages = self._compactor.get_api_messages() + [
            {"role": "user", "content": reflection_prompt}
        ]

        # ReAct loop for idle reflection (with restrictions)
        for iteration in range(self.config.max_idle_tool_iterations):
            result = await self.client.generate(
                messages=reflection_messages,
                max_tokens=512,
                temperature=self.config.think_temperature,
                emotional_modulation=self.config.emotional_modulation,
            )

            response_text = result.content

            # Check for tool calls
            tool_call = self._parse_tool_call(response_text)

            if tool_call and self.config.allow_idle_tools:
                # Validate the tool call for idle mode
                error = self._validate_idle_tool_call(tool_call)
                if error:
                    logger.debug(f"Idle tool call rejected: {error}")
                    # Add rejection to messages and continue
                    reflection_messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    reflection_messages.append({
                        "role": "user",
                        "content": f"[System] {error}. Continue your reflection without this tool."
                    })
                    continue

                # Execute the safe tool
                self._state = ServerState.PROCESSING_TOOLS
                logger.debug(f"Idle reflection using tool: {tool_call.get('name')}")

                reflection_messages.append({
                    "role": "assistant",
                    "content": response_text
                })

                try:
                    formatted_call = {
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call.get("arguments", {})),
                        }
                    }
                    tool_result = await self._tool_engine.execute_tool_call(formatted_call)

                    # Notify callback
                    if self.on_tool_call:
                        self.on_tool_call(tool_call["name"], tool_result)

                    # Truncate large results to avoid context overflow
                    result_str = json.dumps(tool_result, indent=2) if isinstance(tool_result, dict) else str(tool_result)
                    max_chars = self.config.max_idle_result_chars
                    if len(result_str) > max_chars:
                        result_str = result_str[:max_chars] + f"\n\n[... truncated, {len(result_str) - max_chars} chars omitted]"

                    reflection_messages.append({
                        "role": "user",
                        "content": f"[Tool result for {tool_call['name']}]:\n{result_str}"
                    })
                except Exception as e:
                    logger.error(f"Idle tool execution failed: {e}")
                    reflection_messages.append({
                        "role": "user",
                        "content": f"[Tool error]: {str(e)}"
                    })

                continue

            # No tool call or tools disabled - this is the final thought
            thought = ThoughtEvent(
                content=response_text,
                thought_type="reflection",
                triggered_by="idle",
            )

            logger.debug(f"Idle thought: {response_text[:100]}...")

            if self.on_thought:
                self.on_thought(thought)

            return

        # Max iterations reached
        logger.debug("Max idle tool iterations reached")

    def _get_reflection_prompt(self) -> str:
        """Get a prompt for generating internal reflection."""
        import random

        base_instruction = """[INTERNAL REFLECTION - NOT FOR USER]
This is your private thinking time. The user will not see this response.

IMPORTANT: Do NOT imagine or hallucinate file contents, directory listings,
or command outputs. If you want to explore, you MUST use the actual tools.

To use a tool, respond with ONLY:
```tool_call
{"name": "tool_name", "arguments": {...}}
```

Available tools: list_directory, read_file, search_codebase
(No bash or write access during reflection)

"""

        prompts = [
            base_instruction + "What exists in this workspace? Start by listing the directory.",

            base_instruction + "What would you like to understand about this codebase? Use tools to investigate.",

            base_instruction + "Reflect on the conversation so far. Is there something you could look up to help?",

            base_instruction + "Explore your surroundings. What files or patterns catch your interest?",
        ]

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
        logger.info("Context cleared")
