"""
Idle thinking handler for Psyche.

This module handles autonomous "dreaming" when the user is idle.
It generates reflective thoughts and manages memory consolidation.

This is core business logic, independent of the UI layer.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional, Set

from loguru import logger

if TYPE_CHECKING:
    from psyche.mcp.client import ElpisClient, MnemosyneClient
    from psyche.tools.tool_engine import ToolEngine

from psyche.memory.compaction import ContextCompactor
from psyche.shared.constants import (
    CONSOLIDATION_IMPORTANCE_THRESHOLD,
    CONSOLIDATION_SIMILARITY_THRESHOLD,
)


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
    consolidation_importance_threshold: float = CONSOLIDATION_IMPORTANCE_THRESHOLD
    consolidation_similarity_threshold: float = CONSOLIDATION_SIMILARITY_THRESHOLD


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
    - compactor: For getting conversation context
    - mnemosyne_client: For memory storage/retrieval
    """

    def __init__(
        self,
        elpis_client: ElpisClient,
        compactor: ContextCompactor,
        tool_engine: Optional[ToolEngine] = None,
        mnemosyne_client: Optional[MnemosyneClient] = None,
        config: Optional[IdleConfig] = None,
    ):
        """
        Initialize the idle handler.

        Args:
            elpis_client: Client for LLM inference
            compactor: Context compactor for getting conversation context
            tool_engine: Optional engine for executing tools
            mnemosyne_client: Optional client for Mnemosyne memory server
            config: Configuration options
        """
        self.client = elpis_client
        self.compactor = compactor
        self.tool_engine = tool_engine
        self.mnemosyne_client = mnemosyne_client
        self.config = config or IdleConfig()

        # Timing tracking
        self._startup_time: float = time.time()
        self._last_idle_tool_use: float = 0.0
        self._last_user_interaction: float = 0.0
        self._last_consolidation_check: float = 0.0

        # State tracking
        self._is_thinking = False
        self._interrupt_event = asyncio.Event()

        # Interaction counter for consolidation triggers
        self._interaction_count = 0

    def can_start_thinking(self) -> bool:
        """
        Check if idle thinking can begin.

        Idle thinking is delayed after user interactions to avoid
        appearing to "continue speaking" without accepting input.

        Returns:
            True if enough time has passed since last user interaction,
            False otherwise
        """
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

    def can_use_tools(self) -> bool:
        """
        Check if tools can be used during idle (rate limiting).

        Tools are restricted during:
        1. Startup warmup period (first N seconds after server start)
        2. Cooldown period after last idle tool use

        Returns:
            True if idle tools can be used, False otherwise
        """
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
        self._is_thinking = True
        self._interrupt_event.clear()

        try:
            # Verify connection before attempting generation
            if not self.client.is_connected:
                logger.warning("Elpis client disconnected, skipping idle thought")
                return None

            # Add a prompt for reflection
            reflection_prompt = self.get_reflection_prompt()
            reflection_messages = self.compactor.get_api_messages() + [
                {"role": "user", "content": reflection_prompt}
            ]

            # ReAct loop for idle reflection (with restrictions)
            for iteration in range(self.config.max_idle_tool_iterations):
                # Re-check connection before each iteration
                if not self.client.is_connected:
                    logger.warning("Elpis client disconnected during idle thought")
                    return None

                # Check for interrupt before starting generation
                if self._interrupt_event.is_set():
                    self._interrupt_event.clear()
                    logger.info("Idle thought interrupted before generation")
                    return None

                # Use streaming generation with interrupt support
                response_tokens: List[str] = []
                interrupted = False

                try:
                    async with asyncio.timeout(self.config.generation_timeout):
                        async for token in self.client.generate_stream(
                            messages=reflection_messages,
                            max_tokens=512,
                            temperature=self.config.think_temperature,
                            emotional_modulation=self.config.emotional_modulation,
                        ):
                            # Check for interrupt request
                            if self._interrupt_event.is_set():
                                self._interrupt_event.clear()
                                logger.info("Idle thought interrupted during generation")
                                interrupted = True
                                break

                            response_tokens.append(token)
                            # Stream token to thinking callback (for UI display)
                            if on_token:
                                on_token(token)

                except asyncio.TimeoutError:
                    logger.warning(f"Idle thought generation timed out after {self.config.generation_timeout}s")
                    return None
                except asyncio.CancelledError:
                    # Task was cancelled (e.g., user input arrived)
                    logger.debug("Idle thought generation cancelled")
                    raise
                except Exception as e:
                    error_msg = str(e) if str(e) else type(e).__name__
                    logger.error(f"Idle thought generation failed: {error_msg}")
                    if "Connection closed" in str(e) or "closed" in str(e).lower():
                        logger.warning("Server connection lost during idle thinking")
                    return None

                # If interrupted, stop processing
                if interrupted:
                    return None

                response_text = "".join(response_tokens)

                # Check for tool calls
                tool_call = self._parse_tool_call(response_text)

                if tool_call and self.config.allow_idle_tools and self.tool_engine:
                    # Check rate limiting first
                    if not self.can_use_tools():
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
                    error = self.validate_tool_call(tool_call)
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
                        tool_result = await self.tool_engine.execute_tool_call(formatted_call)

                        # Update last idle tool use timestamp for rate limiting
                        self.record_tool_use()

                        # Notify callback
                        if on_tool_call:
                            on_tool_call(tool_call["name"], tool_call.get("arguments", {}), tool_result)

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

                if on_thought:
                    on_thought(thought)

                # Check if memory consolidation is needed
                await self.maybe_consolidate()

                return response_text

            # Max iterations reached
            logger.debug("Max idle tool iterations reached")

            # Still check consolidation even if max iterations reached
            await self.maybe_consolidate()

            return None

        finally:
            self._is_thinking = False

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
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        # Check if tool is in safe list
        if tool_name not in SAFE_IDLE_TOOLS:
            return f"Tool '{tool_name}' not allowed during idle reflection"

        # Check path arguments for safety
        path_args = ["file_path", "dir_path", "path"]
        for arg in path_args:
            if arg in arguments:
                if not self.is_safe_path(arguments[arg]):
                    return f"Path '{arguments[arg]}' not allowed during idle reflection"

        return None

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
        # Skip if consolidation is disabled or no mnemosyne client
        if not self.config.enable_consolidation or not self.mnemosyne_client:
            return False

        # Check if enough time has passed since last consolidation check
        now = time.time()
        time_since_last_check = now - self._last_consolidation_check
        if time_since_last_check < self.config.consolidation_check_interval:
            return False

        self._last_consolidation_check = now

        try:
            # Check if Mnemosyne is connected
            if not self.mnemosyne_client.is_connected:
                logger.debug("Mnemosyne not connected, skipping consolidation check")
                return False

            # Check if consolidation is recommended
            should_consolidate, reason, short_term, long_term = await self.mnemosyne_client.should_consolidate()

            if not should_consolidate:
                logger.debug(f"Consolidation not needed: {reason}")
                return False

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

            return True

        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            return False

    def record_user_interaction(self) -> None:
        """
        Record that a user interaction occurred.

        Called when user submits input to reset the idle timer.
        """
        self._last_user_interaction = time.time()
        self._interaction_count += 1

    def record_tool_use(self) -> None:
        """
        Record that an idle tool was used.

        Called after successful idle tool execution to update rate limiting.
        """
        self._last_idle_tool_use = time.time()

    def interrupt(self) -> bool:
        """
        Request interruption of current idle thinking.

        Returns:
            True if interrupt was requested, False otherwise
        """
        if self._is_thinking:
            self._interrupt_event.set()
            logger.debug("Idle thought interrupt requested")
            return True
        return False

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag after handling."""
        self._interrupt_event.clear()

    @property
    def is_thinking(self) -> bool:
        """Check if currently generating idle thoughts."""
        return self._is_thinking
