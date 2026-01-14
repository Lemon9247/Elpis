"""Memory server with continuous inference loop."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from loguru import logger

from psyche.mcp.client import ElpisClient, GenerationResult, FunctionCallResult, MnemosyneClient
from psyche.memory.compaction import CompactionResult, ContextCompactor, Message, create_message
from psyche.tools.tool_engine import ToolEngine, ToolSettings
from psyche.tools.tool_definitions import ToolDefinition, RecallMemoryInput, StoreMemoryInput
from psyche.tools.implementations.memory_tools import MemoryTools


# Tools that are safe for autonomous idle reflection (read-only)
SAFE_IDLE_TOOLS: Set[str] = {"read_file", "list_directory", "search_codebase", "recall_memory"}

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

    # Context settings (sized for 32k context window)
    max_context_tokens: int = 24000
    reserve_tokens: int = 4000

    # Emotional settings
    emotional_modulation: bool = True

    # Tool settings
    workspace_dir: str = "."  # Working directory for tools
    max_tool_iterations: int = 10  # Maximum ReAct iterations per request
    max_tool_result_chars: int = 16000  # Truncate tool results to avoid context overflow

    # Idle reflection settings
    allow_idle_tools: bool = True  # Allow read-only tools during reflection
    max_idle_tool_iterations: int = 3  # Max tool calls per reflection
    max_idle_result_chars: int = 8000  # Truncate tool results to this size

    # Tool rate limiting for idle/dream state
    startup_warmup_seconds: float = 120.0  # No tools for first 2 minutes after startup
    idle_tool_cooldown_seconds: float = 300.0  # Minimum seconds between idle tool uses (5 min)

    # Post-interaction delay before idle thinking resumes
    post_interaction_delay: float = 60.0  # Wait 60s after user interaction before idle thoughts

    # Generation timeout (prevents indefinite hangs)
    generation_timeout: float = 120.0  # Max seconds for a single generation call

    # Memory consolidation settings
    enable_consolidation: bool = True  # Enable automatic memory consolidation
    consolidation_check_interval: float = 300.0  # Check every 5 minutes
    consolidation_importance_threshold: float = 0.6  # Min importance for promotion
    consolidation_similarity_threshold: float = 0.85  # Similarity threshold for clustering


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
        mnemosyne_client: Optional[MnemosyneClient] = None,
        on_thought: Optional[Callable[[ThoughtEvent], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_consolidation: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize the memory server.

        Args:
            elpis_client: Client for connecting to Elpis inference server
            config: Server configuration
            mnemosyne_client: Optional client for Mnemosyne memory server (enables consolidation)
            on_thought: Callback for internal thoughts (for UI display)
            on_response: Callback for responses to user
            on_tool_call: Callback when a tool is executed (name, result or None for start)
            on_token: Callback for streaming tokens (for real-time display)
            on_consolidation: Callback when memory consolidation runs
        """
        self.client = elpis_client
        self.mnemosyne_client = mnemosyne_client
        self.config = config or ServerConfig()
        self.on_thought = on_thought
        self.on_response = on_response
        self.on_tool_call = on_tool_call
        self.on_token = on_token
        self.on_consolidation = on_consolidation

        self._state = ServerState.IDLE
        self._compactor = ContextCompactor(
            max_tokens=self.config.max_context_tokens,
            reserve_tokens=self.config.reserve_tokens,
        )

        self._input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._running = False

        # Track timing for idle tool rate limiting and post-interaction delay
        import time
        self._startup_time: float = time.time()
        self._last_idle_tool_use: float = 0.0  # Never used yet
        self._last_user_interaction: float = 0.0  # Track last user input time
        self._last_consolidation_check: float = 0.0  # Track last consolidation check

        # Staged messages buffer for delayed Mnemosyne storage
        # Messages are staged for one compaction cycle before being stored
        self._staged_messages: List[Message] = []

        # Initialize tool engine
        self._tool_engine = ToolEngine(
            workspace_dir=self.config.workspace_dir,
            settings=ToolSettings(),
        )

        # Register memory tools if Mnemosyne client is available
        if self.mnemosyne_client:
            self._register_memory_tools()

        # System prompt for the continuous agent
        self._system_prompt = self._build_system_prompt()

    def _register_memory_tools(self) -> None:
        """Register memory tools with the tool engine."""
        if not self.mnemosyne_client:
            return

        # Create memory tools instance with emotion getter
        memory_tools = MemoryTools(
            mnemosyne_client=self.mnemosyne_client,
            get_emotion_fn=self.client.get_emotion,
        )

        # Register recall_memory tool
        self._tool_engine.register_tool(ToolDefinition(
            name="recall_memory",
            description="Search and recall memories from long-term storage. Use this to remember past conversations, facts, or experiences.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant memories",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of memories to retrieve (1-20, default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            input_model=RecallMemoryInput,
            handler=memory_tools.recall_memory,
        ))

        # Register store_memory tool
        self._tool_engine.register_tool(ToolDefinition(
            name="store_memory",
            description="Store a new memory for later recall. Use this to remember important information, facts, or experiences.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content of the memory to store",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of the memory (auto-generated if not provided)",
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "Type of memory: episodic (events), semantic (facts), procedural (how-to), emotional (feelings)",
                        "default": "episodic",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags to categorize the memory",
                    },
                },
                "required": ["content"],
            },
            input_model=StoreMemoryInput,
            handler=memory_tools.store_memory,
        ))

        logger.info("Memory tools registered: recall_memory, store_memory")

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        # Get tool descriptions from the tool engine
        tool_descriptions = self._tool_engine.get_tool_descriptions()

        return f"""You are Psyche, a thoughtful AI assistant.

## Core Behavior

Respond naturally and conversationally. Most messages just need a direct response - no tools required.

**Only use tools when the user explicitly asks you to:**
- Read, create, or edit files
- Run commands or scripts
- Search the codebase
- List directory contents
- Remember something important (store_memory)
- Recall past conversations or facts (recall_memory)

For greetings, questions, discussion, or general conversation - just respond directly. Do not use tools unless the task genuinely requires them.

**Memory Tools:** You have access to long-term memory. Use `recall_memory` to search for relevant past experiences or knowledge. Use `store_memory` to save important information you want to remember later. If recall returns no results, that's normal - it just means you don't have relevant memories yet. Don't apologize or give up; continue the conversation naturally.

## Tool Usage

When you need a tool, use this format and then STOP:
```tool_call
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
```

**CRITICAL: After writing a tool_call block, you MUST stop immediately.** Do not write anything after the closing ```. Do not guess or imagine what the tool will return. The actual result will be provided to you in the next message.

{tool_descriptions}

## Guidelines

- Respond conversationally first - tools are a last resort
- When using tools, explain what you're doing, then call the tool and STOP
- NEVER fabricate or imagine tool outputs - wait for the real result
- Always read files before modifying them
- Be careful with bash commands"""

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

                # Optionally connect to Mnemosyne for memory consolidation
                if self.mnemosyne_client and self.config.enable_consolidation:
                    async with self.mnemosyne_client.connect():
                        logger.info("Connected to Mnemosyne memory server")

                        # Add system prompt to context
                        self._compactor.add_message(create_message("system", self._system_prompt))

                        # Run the main loop with both connections
                        await self._inference_loop()
                else:
                    # Run without mnemosyne
                    if self.config.enable_consolidation and not self.mnemosyne_client:
                        logger.warning("Consolidation enabled but no Mnemosyne client provided")

                    # Add system prompt to context
                    self._compactor.add_message(create_message("system", self._system_prompt))

                    # Run the main loop
                    await self._inference_loop()
        except ExceptionGroup as eg:
            # Recursively unwrap ExceptionGroup to get the actual error(s)
            def unwrap_exception_group(exc_group, depth=0):
                """Recursively extract all leaf exceptions from nested ExceptionGroups."""
                errors = []
                for exc in exc_group.exceptions:
                    if isinstance(exc, ExceptionGroup):
                        errors.extend(unwrap_exception_group(exc, depth + 1))
                    else:
                        errors.append(exc)
                return errors

            leaf_errors = unwrap_exception_group(eg)
            for exc in leaf_errors:
                logger.error(f"Server subprocess error: {type(exc).__name__}: {exc}")
                # Log full traceback for debugging
                import traceback
                logger.error("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

            # Re-raise with the actual error message
            if leaf_errors:
                actual_error = leaf_errors[0]
                raise RuntimeError(f"Server subprocess failed: {type(actual_error).__name__}: {actual_error}") from actual_error
            else:
                raise RuntimeError(f"Server subprocess failed: {eg}") from eg
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
        idle_task: asyncio.Task | None = None
        loop_iteration = 0

        while self._running:
            loop_iteration += 1
            try:
                # Periodic health check logging (every 100 iterations)
                if loop_iteration % 100 == 0:
                    logger.debug(
                        f"Inference loop health check: iteration={loop_iteration}, "
                        f"state={self._state.value}, idle_task={idle_task is not None}, "
                        f"queue_size={self._input_queue.qsize()}"
                    )

                # If idle thinking is running, wait for either user input or idle completion
                if idle_task is not None and not idle_task.done():
                    self._state = ServerState.THINKING
                    # Wait for either user input OR idle task completion
                    input_task = asyncio.create_task(self._input_queue.get())

                    done, pending = await asyncio.wait(
                        [input_task, idle_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if input_task in done:
                        # User input arrived - cancel idle thinking and process input
                        logger.debug("User input interrupted idle thinking")
                        idle_task.cancel()
                        try:
                            await idle_task
                        except asyncio.CancelledError:
                            pass
                        idle_task = None
                        user_input = input_task.result()
                        await self._process_user_input(user_input)
                    else:
                        # Idle task completed - cancel the input wait
                        logger.debug("Idle thinking completed")
                        input_task.cancel()
                        try:
                            await input_task
                        except asyncio.CancelledError:
                            pass
                        # Check for any exception in idle task
                        try:
                            idle_task.result()
                        except asyncio.CancelledError:
                            logger.debug("Idle task was cancelled")
                        except Exception as e:
                            logger.exception(f"Error in idle thinking: {e}")
                        idle_task = None
                else:
                    # No idle task running - wait for input with timeout
                    try:
                        self._state = ServerState.WAITING_INPUT
                        user_input = await asyncio.wait_for(
                            self._input_queue.get(),
                            timeout=self.config.idle_think_interval,
                        )
                        logger.debug(f"Received user input: {user_input[:50]}...")
                        await self._process_user_input(user_input)

                    except asyncio.TimeoutError:
                        # No user input - check if we should start idle thinking
                        if self._can_start_idle_thinking():
                            logger.debug("Starting idle thinking")
                            idle_task = asyncio.create_task(self._generate_idle_thought())
                        # If not allowed yet, loop will continue waiting for input

            except asyncio.CancelledError:
                logger.info("Inference loop cancelled")
                if idle_task:
                    idle_task.cancel()
                    try:
                        await idle_task
                    except asyncio.CancelledError:
                        pass
                break
            except Exception as e:
                logger.exception(f"Error in inference loop: {e}")
                # Brief pause before retrying
                await asyncio.sleep(1.0)

        logger.info(f"Inference loop stopped after {loop_iteration} iterations")

    async def _process_user_input(self, text: str) -> None:
        """Process user input with ReAct loop for tool execution."""
        import time
        self._last_user_interaction = time.time()  # Track interaction time

        # Verify connection before processing
        if not self.client.is_connected:
            logger.error("Elpis client disconnected, cannot process user input")
            if self.on_response:
                self.on_response("[Error: Inference server disconnected]")
            return

        self._state = ServerState.THINKING
        logger.debug(f"Processing user input: {text[:50]}...")

        # Add user message to context
        compaction_result = self._compactor.add_message(create_message("user", text))
        if compaction_result:
            await self._handle_compaction_result(compaction_result)

        # ReAct loop - iterate until LLM provides final response without tools
        for iteration in range(self.config.max_tool_iterations):
            # Check if new user input arrived - if so, break out to process it
            if not self._input_queue.empty():
                logger.debug("New user input detected, breaking ReAct loop")
                return

            # Re-check connection before each iteration
            if not self.client.is_connected:
                logger.error("Elpis client disconnected during processing")
                if self.on_response:
                    self.on_response("[Error: Inference server disconnected]")
                return

            messages = self._compactor.get_api_messages()

            # Generate response with streaming (with timeout)
            self._state = ServerState.THINKING
            response_tokens: List[str] = []

            try:
                # Use async_timeout context for the entire streaming operation
                async with asyncio.timeout(self.config.generation_timeout):
                    async for token in self.client.generate_stream(
                        messages=messages,
                        emotional_modulation=self.config.emotional_modulation,
                    ):
                        response_tokens.append(token)
                        # Stream token to UI callback
                        if self.on_token:
                            self.on_token(token)
            except asyncio.TimeoutError:
                logger.error(f"Generation timed out after {self.config.generation_timeout}s")
                if self.on_response:
                    partial = "".join(response_tokens)
                    self.on_response(partial + "\n\n[Generation timed out]")
                return
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                if self.on_response:
                    self.on_response(f"[Error: {e}]")
                return

            response_text = "".join(response_tokens)

            # Try to parse tool calls from the response
            tool_call = self._parse_tool_call(response_text)

            if tool_call:
                # Found a tool call - execute it
                self._state = ServerState.PROCESSING_TOOLS
                logger.debug(f"Parsed tool call: {tool_call.get('name')}")

                # Signal end of this generation (so UI can end stream before tool runs)
                if self.on_response:
                    self.on_response(response_text)

                # Add assistant's tool call to context
                compaction_result = self._compactor.add_message(create_message("assistant", response_text))
                if compaction_result:
                    await self._handle_compaction_result(compaction_result)

                # Execute the tool
                await self._execute_parsed_tool_call(tool_call)

                # Continue loop to get next response
                continue

            # No tool call - this is the final response
            compaction_result = self._compactor.add_message(create_message("assistant", response_text))
            if compaction_result:
                await self._handle_compaction_result(compaction_result)

            # Notify callback (tokens already streamed, this signals completion)
            if self.on_response:
                self.on_response(response_text)

            # Update emotional state based on interaction
            await self._update_emotion_for_interaction_text(response_text)
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

        # Notify callback at start (result=None indicates start)
        if self.on_tool_call:
            self.on_tool_call(tool_name, None)

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

            # Notify callback at end (with result)
            if self.on_tool_call:
                self.on_tool_call(tool_name, result)

            # Add tool result to context (truncated to avoid context overflow)
            result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
            max_chars = self.config.max_tool_result_chars
            if len(result_str) > max_chars:
                result_str = result_str[:max_chars] + f"\n\n[... truncated, {len(result_str) - max_chars} chars omitted]"

            compaction_result = self._compactor.add_message(create_message(
                "user",
                f"[Tool result for {tool_name}]:\n{result_str}",
            ))
            if compaction_result:
                await self._handle_compaction_result(compaction_result)

            # Trigger emotion based on result
            if result.get("success", True):
                await self.client.update_emotion("success", intensity=0.3)
            else:
                await self.client.update_emotion("failure", intensity=0.5)

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")

            # Add error to context
            compaction_result = self._compactor.add_message(create_message(
                "user",
                f"[Tool error for {tool_name}]: {str(e)}",
            ))
            if compaction_result:
                await self._handle_compaction_result(compaction_result)

            # Trigger frustration emotion
            await self.client.update_emotion("frustration", intensity=0.5)

    def _can_start_idle_thinking(self) -> bool:
        """
        Check if idle thinking should start now.

        Idle thinking is delayed after user interactions to avoid
        appearing to "continue speaking" without accepting input.

        Returns:
            True if idle thinking can start, False otherwise
        """
        import time
        now = time.time()

        # Check if enough time has passed since last user interaction
        time_since_interaction = now - self._last_user_interaction
        if time_since_interaction < self.config.post_interaction_delay:
            logger.debug(
                f"Idle thinking delayed: post-interaction cooldown "
                f"({time_since_interaction:.0f}s / {self.config.post_interaction_delay:.0f}s)"
            )
            return False

        return True

    def _can_use_idle_tools(self) -> bool:
        """
        Check if tool use is currently allowed during idle/dream state.

        Tools are restricted during:
        1. Startup warmup period (first N seconds after server start)
        2. Cooldown period after last idle tool use

        Returns:
            True if idle tools can be used, False otherwise
        """
        import time
        now = time.time()

        # Check startup warmup period
        time_since_startup = now - self._startup_time
        if time_since_startup < self.config.startup_warmup_seconds:
            logger.debug(
                f"Idle tools disabled: startup warmup "
                f"({time_since_startup:.0f}s / {self.config.startup_warmup_seconds:.0f}s)"
            )
            return False

        # Check cooldown since last idle tool use
        time_since_last_use = now - self._last_idle_tool_use
        if time_since_last_use < self.config.idle_tool_cooldown_seconds:
            logger.debug(
                f"Idle tools disabled: cooldown "
                f"({time_since_last_use:.0f}s / {self.config.idle_tool_cooldown_seconds:.0f}s)"
            )
            return False

        return True

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

        # Verify connection before attempting generation
        if not self.client.is_connected:
            logger.warning("Elpis client disconnected, skipping idle thought")
            return

        # Add a prompt for reflection
        reflection_prompt = self._get_reflection_prompt()
        reflection_messages = self._compactor.get_api_messages() + [
            {"role": "user", "content": reflection_prompt}
        ]

        # ReAct loop for idle reflection (with restrictions)
        for iteration in range(self.config.max_idle_tool_iterations):
            # Re-check connection before each iteration
            if not self.client.is_connected:
                logger.warning("Elpis client disconnected during idle thought")
                return

            try:
                result = await asyncio.wait_for(
                    self.client.generate(
                        messages=reflection_messages,
                        max_tokens=512,
                        temperature=self.config.think_temperature,
                        emotional_modulation=self.config.emotional_modulation,
                    ),
                    timeout=self.config.generation_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Idle thought generation timed out after {self.config.generation_timeout}s")
                return
            except Exception as e:
                logger.error(f"Idle thought generation failed: {e}")
                return

            response_text = result.content

            # Check for tool calls
            tool_call = self._parse_tool_call(response_text)

            if tool_call and self.config.allow_idle_tools:
                # Check rate limiting first
                if not self._can_use_idle_tools():
                    logger.debug("Idle tool call skipped: rate limited")
                    reflection_messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    reflection_messages.append({
                        "role": "user",
                        "content": "[System] Tool use is currently rate-limited during reflection. Continue your thoughts without tools."
                    })
                    continue

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

                    # Update last idle tool use timestamp for rate limiting
                    import time
                    self._last_idle_tool_use = time.time()

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

            # Check if memory consolidation is needed
            await self._maybe_consolidate_memories()

            return

        # Max iterations reached
        logger.debug("Max idle tool iterations reached")

        # Still check consolidation even if max iterations reached
        await self._maybe_consolidate_memories()

    def _get_reflection_prompt(self) -> str:
        """Get a prompt for generating internal reflection."""
        import random

        base_instruction = """[INTERNAL REFLECTION - NOT FOR USER]
This is your private thinking time. The user will not see this response.

Think out loud as you explore. Follow this pattern:
1. First, write your thoughts about what you want to investigate and why
2. Then use a tool to gather information
3. After seeing the result, reflect on what you learned before continuing

To use a tool, include a tool_call block:
```tool_call
{"name": "tool_name", "arguments": {...}}
```

Available tools: list_directory, read_file, search_codebase
(No bash or write access during reflection)

IMPORTANT: Do NOT imagine or hallucinate file contents or command outputs.
You MUST use the actual tools to see real data.

"""

        prompts = [
            base_instruction + "What exists in this workspace? Think about what you're curious about, then explore.",

            base_instruction + "What would you like to understand about this codebase? Share your reasoning as you investigate.",

            base_instruction + "Reflect on the conversation so far. What could you look up that might be helpful? Think through your approach.",

            base_instruction + "Explore your surroundings. What catches your interest and why?",
        ]

        return random.choice(prompts)

    async def _update_emotion_for_interaction(self, result: GenerationResult) -> None:
        """Update emotional state based on interaction quality."""
        await self._update_emotion_for_interaction_text(result.content)

    async def _update_emotion_for_interaction_text(self, content: str) -> None:
        """Update emotional state based on response text length."""
        # Simple heuristic: longer, more engaged responses are positive
        content_length = len(content)

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

    async def _maybe_consolidate_memories(self) -> None:
        """
        Check if memory consolidation is needed and run it if so.

        This is called during idle periods to promote important short-term
        memories to long-term storage.
        """
        # Skip if consolidation is disabled or no mnemosyne client
        if not self.config.enable_consolidation or not self.mnemosyne_client:
            return

        # Check if enough time has passed since last consolidation check
        import time
        now = time.time()
        time_since_last_check = now - self._last_consolidation_check
        if time_since_last_check < self.config.consolidation_check_interval:
            return

        self._last_consolidation_check = now

        try:
            # Check if consolidation is recommended
            should_consolidate, reason, short_term, long_term = await self.mnemosyne_client.should_consolidate()

            if not should_consolidate:
                logger.debug(f"Consolidation not needed: {reason}")
                return

            logger.info(f"Starting memory consolidation: {reason}")

            # Run consolidation
            result = await self.mnemosyne_client.consolidate_memories(
                importance_threshold=self.config.consolidation_importance_threshold,
                similarity_threshold=self.config.consolidation_similarity_threshold,
            )

            logger.info(
                f"Consolidation complete: promoted {result.memories_promoted}, "
                f"archived {result.memories_archived}, "
                f"clusters formed: {result.clusters_formed}"
            )

            # Notify callback
            if self.on_consolidation:
                self.on_consolidation({
                    "clusters_formed": result.clusters_formed,
                    "memories_promoted": result.memories_promoted,
                    "memories_archived": result.memories_archived,
                    "memories_skipped": result.memories_skipped,
                    "duration_seconds": result.duration_seconds,
                })

            # Emit a thought about consolidation
            if self.on_thought and result.memories_promoted > 0:
                self.on_thought(ThoughtEvent(
                    content=f"[Memory consolidation: promoted {result.memories_promoted} memories to long-term storage]",
                    thought_type="memory",
                    triggered_by="consolidation",
                ))

        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def _summarize_conversation(self, messages: List[Message]) -> str:
        """
        Use Elpis to generate a conversation summary.

        Args:
            messages: List of messages to summarize

        Returns:
            Summary string, or empty string on failure
        """
        if not messages:
            return ""

        try:
            # Build conversation text for summarization
            conversation_parts = []
            for msg in messages:
                if msg.role == "system":
                    continue
                # Truncate very long messages for summarization
                content = msg.content[:500] if len(msg.content) > 500 else msg.content
                conversation_parts.append(f"{msg.role}: {content}")

            if not conversation_parts:
                return ""

            conversation_text = "\n".join(conversation_parts)

            # Use Elpis to generate summary
            summary_prompt = [
                {
                    "role": "system",
                    "content": (
                        "Summarize this conversation concisely. Extract key facts, "
                        "decisions, topics discussed, and important details. "
                        "Focus on information worth remembering long-term."
                    ),
                },
                {"role": "user", "content": conversation_text},
            ]

            result = await self.client.generate(
                messages=summary_prompt,
                max_tokens=500,
                temperature=0.3,
            )

            summary = result.content.strip()
            logger.debug(f"Generated conversation summary: {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            return ""

    async def _store_conversation_summary(self, messages: List[Message]) -> bool:
        """
        Generate and store a conversation summary as semantic memory.

        Args:
            messages: List of messages to summarize

        Returns:
            True if summary stored successfully, False otherwise
        """
        if not self.mnemosyne_client or not self.mnemosyne_client.is_connected:
            return False

        summary = await self._summarize_conversation(messages)
        if not summary:
            logger.debug("No summary generated, skipping storage")
            return False

        try:
            # Get current emotional context
            emotion = await self.client.get_emotion()

            await self.mnemosyne_client.store_memory(
                content=summary,
                summary=summary[:100] + "..." if len(summary) > 100 else summary,
                memory_type="semantic",  # Semantic memory for distilled knowledge
                tags=["conversation_summary", "shutdown"],
                emotional_context={
                    "valence": emotion.valence,
                    "arousal": emotion.arousal,
                    "quadrant": emotion.quadrant,
                },
            )
            logger.info("Stored conversation summary as semantic memory")
            return True

        except Exception as e:
            logger.error(f"Failed to store conversation summary: {e}")
            return False

    async def _store_messages_to_mnemosyne(self, messages: List[Message]) -> bool:
        """
        Store messages to Mnemosyne short-term memory.

        Args:
            messages: List of messages to store as episodic memories

        Returns:
            True if all messages stored successfully, False otherwise
        """
        if not self.mnemosyne_client:
            logger.warning("No Mnemosyne client, cannot store messages")
            return False

        if not self.mnemosyne_client.is_connected:
            logger.warning("Mnemosyne not connected, cannot store messages")
            return False

        success_count = 0
        total_count = 0

        for msg in messages:
            if msg.role == "system":
                continue  # Skip system prompts

            total_count += 1
            try:
                # Get current emotional context
                emotion = await self.client.get_emotion()

                await self.mnemosyne_client.store_memory(
                    content=msg.content,
                    summary=msg.content[:100],
                    memory_type="episodic",
                    tags=["compacted", msg.role],
                    emotional_context={
                        "valence": emotion.valence,
                        "arousal": emotion.arousal,
                        "quadrant": emotion.quadrant,
                    },
                )
                success_count += 1
                logger.debug(f"Stored {msg.role} message to Mnemosyne")
            except Exception as e:
                logger.error(f"Failed to store message to Mnemosyne: {e}")

        if total_count == 0:
            return True  # No messages to store is a success

        return success_count == total_count

    async def _handle_compaction_result(self, result: CompactionResult) -> None:
        """
        Handle compaction result by storing staged messages and staging new ones.

        Implements delayed storage: messages are staged for one compaction cycle
        before being stored to Mnemosyne short-term memory.

        Args:
            result: CompactionResult from context compaction
        """
        # Store previously staged messages to Mnemosyne
        if self._staged_messages and self.mnemosyne_client:
            logger.debug(f"Storing {len(self._staged_messages)} staged messages to Mnemosyne")
            success = await self._store_messages_to_mnemosyne(self._staged_messages)
            if success:
                # Only clear staged messages if storage succeeded
                self._staged_messages = []
            else:
                logger.error(
                    f"Failed to store {len(self._staged_messages)} staged messages, "
                    "will retry on next compaction"
                )

        # Stage newly dropped messages (append to any failed messages)
        if result.dropped_messages:
            self._staged_messages.extend(result.dropped_messages)
            logger.debug(f"Staged {len(result.dropped_messages)} new messages, total staged: {len(self._staged_messages)}")

    async def shutdown_with_consolidation(self) -> None:
        """
        Graceful shutdown with memory consolidation.

        Stores all staged and remaining context messages to Mnemosyne,
        generates a conversation summary, then runs consolidation before shutdown.
        """
        logger.info("Starting graceful shutdown with memory consolidation...")

        if not self.mnemosyne_client:
            logger.debug("No Mnemosyne client, skipping memory consolidation")
            return

        if not self.mnemosyne_client.is_connected:
            logger.warning("Mnemosyne not connected, skipping shutdown consolidation")
            return

        try:
            # Store any staged messages
            if self._staged_messages:
                logger.debug(f"Storing {len(self._staged_messages)} staged messages")
                success = await self._store_messages_to_mnemosyne(self._staged_messages)
                if success:
                    self._staged_messages = []
                else:
                    logger.error(f"Failed to store {len(self._staged_messages)} staged messages during shutdown")

            # Store remaining context messages (non-system)
            remaining = [m for m in self._compactor.messages if m.role != "system"]
            if remaining:
                logger.debug(f"Storing {len(remaining)} remaining context messages")
                await self._store_messages_to_mnemosyne(remaining)

            # Generate and store conversation summary
            all_messages = self._staged_messages + remaining
            if all_messages:
                await self._store_conversation_summary(all_messages)

            # Run consolidation
            logger.info("Running shutdown consolidation...")
            result = await self.mnemosyne_client.consolidate_memories(
                importance_threshold=self.config.consolidation_importance_threshold,
                similarity_threshold=self.config.consolidation_similarity_threshold,
            )
            logger.info(
                f"Shutdown consolidation complete: promoted {result.memories_promoted}, "
                f"archived {result.memories_archived}"
            )

            if self.on_consolidation:
                self.on_consolidation({
                    "type": "shutdown",
                    "memories_promoted": result.memories_promoted,
                    "memories_archived": result.memories_archived,
                })

        except Exception as e:
            logger.error(f"Shutdown consolidation failed: {e}")

    def clear_context(self) -> None:
        """Clear conversation context."""
        self._compactor.clear()
        # Re-add system prompt
        self._compactor.add_message(create_message("system", self._system_prompt))
        logger.info("Context cleared")
