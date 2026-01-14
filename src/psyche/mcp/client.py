"""MCP client for connecting to Elpis inference server."""

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class EmotionalState:
    """Representation of the inference server's emotional state."""

    valence: float = 0.0
    arousal: float = 0.0
    quadrant: str = "neutral"
    update_count: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionalState":
        """Create from dictionary returned by server."""
        return cls(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.0),
            quadrant=data.get("quadrant", "neutral"),
            update_count=data.get("update_count", 0),
        )


@dataclass
class GenerationResult:
    """Result from a generation request."""

    content: str
    emotional_state: EmotionalState
    modulated_params: Dict[str, float] = field(default_factory=dict)


@dataclass
class FunctionCallResult:
    """Result from a function call request."""

    tool_calls: List[Dict[str, Any]]
    emotional_state: EmotionalState


class ElpisClient:
    """
    MCP client for the Elpis inference server.

    Manages connection to the Elpis server and provides methods for:
    - Text generation with emotional modulation
    - Function/tool call generation
    - Emotional state management
    """

    def __init__(
        self,
        server_command: str = "elpis-server",
        server_args: Optional[List[str]] = None,
        quiet: bool = True,
    ):
        """
        Initialize the Elpis client.

        Args:
            server_command: Command to launch the Elpis server
            server_args: Additional arguments for the server command
            quiet: Suppress server stderr logging (set ELPIS_QUIET=1)
        """
        self.server_command = server_command
        self.server_args = server_args or []
        self.quiet = quiet
        self._session: Optional[ClientSession] = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to server."""
        return self._connected and self._session is not None

    @asynccontextmanager
    async def connect(self) -> AsyncIterator["ElpisClient"]:
        """
        Context manager for connecting to the Elpis server.

        Usage:
            async with client.connect() as connected_client:
                result = await connected_client.generate(messages)
        """
        # Set up environment for the server subprocess
        env = None
        if self.quiet:
            import os
            env = os.environ.copy()
            env["ELPIS_QUIET"] = "1"

        server_params = StdioServerParameters(
            command=self.server_command,
            args=self.server_args,
            env=env,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self._session = session
                self._connected = True

                # Initialize the session
                await session.initialize()
                logger.info("Connected to Elpis inference server")

                try:
                    yield self
                finally:
                    self._connected = False
                    self._session = None
                    logger.info("Disconnected from Elpis server")

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Elpis server. Use 'async with client.connect()'")

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the server and return parsed result."""
        self._ensure_connected()

        result = await self._session.call_tool(name, arguments)

        # Parse the JSON response
        if result.content and len(result.content) > 0:
            return json.loads(result.content[0].text)
        return {}

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        emotional_modulation: bool = True,
    ) -> GenerationResult:
        """
        Generate text completion with optional emotional modulation.

        Args:
            messages: Chat messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Override temperature (None = emotionally modulated)
            emotional_modulation: Whether to apply emotional parameter modulation

        Returns:
            GenerationResult with content and emotional state
        """
        arguments = {
            "messages": messages,
            "max_tokens": max_tokens,
            "emotional_modulation": emotional_modulation,
        }
        if temperature is not None:
            arguments["temperature"] = temperature

        result = await self._call_tool("generate", arguments)

        return GenerationResult(
            content=result.get("content", ""),
            emotional_state=EmotionalState.from_dict(result.get("emotional_state", {})),
            modulated_params=result.get("modulated_params", {}),
        )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        emotional_modulation: bool = True,
        poll_interval: float = 0.05,
    ) -> AsyncIterator[str]:
        """
        Generate text completion with streaming.

        Yields tokens as they are generated, enabling real-time display.

        Args:
            messages: Chat messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Override temperature (None = emotionally modulated)
            emotional_modulation: Whether to apply emotional parameter modulation
            poll_interval: Seconds between polling for new tokens

        Yields:
            Individual tokens as they become available
        """
        # Start the streaming generation
        start_args = {
            "messages": messages,
            "max_tokens": max_tokens,
            "emotional_modulation": emotional_modulation,
        }
        if temperature is not None:
            start_args["temperature"] = temperature

        start_result = await self._call_tool("generate_stream_start", start_args)

        if "error" in start_result:
            raise RuntimeError(f"Failed to start stream: {start_result['error']}")

        stream_id = start_result["stream_id"]

        # Poll for tokens until complete
        try:
            while True:
                read_result = await self._call_tool(
                    "generate_stream_read",
                    {"stream_id": stream_id}
                )

                if "error" in read_result:
                    raise RuntimeError(f"Stream error: {read_result['error']}")

                # Yield new tokens
                new_content = read_result.get("new_content", "")
                if new_content:
                    yield new_content

                # Check if complete
                if read_result.get("is_complete", False):
                    break

                # Wait before next poll
                await asyncio.sleep(poll_interval)

        except Exception:
            # Try to cancel the stream on error
            try:
                await self._call_tool("generate_stream_cancel", {"stream_id": stream_id})
            except Exception:
                pass
            raise

    async def function_call(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: Optional[float] = None,
    ) -> FunctionCallResult:
        """
        Generate function/tool calls.

        Args:
            messages: Chat messages in OpenAI format
            tools: Available tools in OpenAI format
            temperature: Override temperature

        Returns:
            FunctionCallResult with tool calls and emotional state
        """
        arguments = {
            "messages": messages,
            "tools": tools,
        }
        if temperature is not None:
            arguments["temperature"] = temperature

        result = await self._call_tool("function_call", arguments)

        return FunctionCallResult(
            tool_calls=result.get("tool_calls", []),
            emotional_state=EmotionalState.from_dict(result.get("emotional_state", {})),
        )

    async def update_emotion(
        self,
        event_type: str,
        intensity: float = 1.0,
        context: Optional[str] = None,
    ) -> EmotionalState:
        """
        Trigger an emotional event on the server.

        Args:
            event_type: Event category (success, failure, novelty, etc.)
            intensity: Event intensity multiplier (0.0 to 2.0)
            context: Optional description for logging

        Returns:
            Updated emotional state
        """
        arguments = {
            "event_type": event_type,
            "intensity": intensity,
        }
        if context:
            arguments["context"] = context

        result = await self._call_tool("update_emotion", arguments)
        return EmotionalState.from_dict(result)

    async def reset_emotion(self) -> EmotionalState:
        """
        Reset emotional state to baseline.

        Returns:
            Reset emotional state
        """
        result = await self._call_tool("reset_emotion", {})
        return EmotionalState.from_dict(result)

    async def get_emotion(self) -> EmotionalState:
        """
        Get current emotional state.

        Returns:
            Current emotional state
        """
        result = await self._call_tool("get_emotion", {})
        return EmotionalState.from_dict(result)

    async def read_resource(self, uri: str) -> str:
        """
        Read a resource from the server.

        Args:
            uri: Resource URI (e.g., "emotion://state")

        Returns:
            Resource content as string
        """
        self._ensure_connected()

        result = await self._session.read_resource(uri)
        if result.contents and len(result.contents) > 0:
            return result.contents[0].text
        return ""

    async def list_available_events(self) -> Dict[str, Any]:
        """
        Get available emotional event types.

        Returns:
            Dictionary of event types and their effects
        """
        content = await self.read_resource("emotion://events")
        return json.loads(content) if content else {}


@dataclass
class ConsolidationResult:
    """Result from memory consolidation."""

    clusters_formed: int = 0
    memories_promoted: int = 0
    memories_archived: int = 0
    memories_skipped: int = 0
    duration_seconds: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsolidationResult":
        """Create from dictionary returned by server."""
        return cls(
            clusters_formed=data.get("clusters_formed", 0),
            memories_promoted=data.get("memories_promoted", 0),
            memories_archived=data.get("memories_archived", 0),
            memories_skipped=data.get("memories_skipped", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
        )


class MnemosyneClient:
    """
    MCP client for the Mnemosyne memory server.

    Manages connection to the Mnemosyne server and provides methods for:
    - Memory storage and retrieval
    - Memory consolidation
    - Memory statistics
    """

    def __init__(
        self,
        server_command: str = "mnemosyne-server",
        server_args: Optional[List[str]] = None,
    ):
        """
        Initialize the Mnemosyne client.

        Args:
            server_command: Command to launch the Mnemosyne server
            server_args: Additional arguments for the server command
        """
        self.server_command = server_command
        self.server_args = server_args or []
        self._session: Optional[ClientSession] = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to server."""
        return self._connected and self._session is not None

    @asynccontextmanager
    async def connect(self) -> AsyncIterator["MnemosyneClient"]:
        """
        Context manager for connecting to the Mnemosyne server.

        Usage:
            async with client.connect() as connected_client:
                result = await connected_client.should_consolidate()
        """
        server_params = StdioServerParameters(
            command=self.server_command,
            args=self.server_args,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self._session = session
                self._connected = True

                # Initialize the session
                await session.initialize()
                logger.info("Connected to Mnemosyne memory server")

                try:
                    yield self
                finally:
                    self._connected = False
                    self._session = None
                    logger.info("Disconnected from Mnemosyne server")

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Mnemosyne server. Use 'async with client.connect()'")

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the server and return parsed result."""
        self._ensure_connected()

        result = await self._session.call_tool(name, arguments)

        # Parse the JSON response
        if result.content and len(result.content) > 0:
            return json.loads(result.content[0].text)
        return {}

    async def should_consolidate(self) -> tuple[bool, str, int, int]:
        """
        Check if memory consolidation is recommended.

        Returns:
            Tuple of (should_consolidate, reason, short_term_count, long_term_count)
        """
        result = await self._call_tool("should_consolidate", {})
        return (
            result.get("should_consolidate", False),
            result.get("reason", ""),
            result.get("short_term_count", 0),
            result.get("long_term_count", 0),
        )

    async def consolidate_memories(
        self,
        importance_threshold: float = 0.6,
        similarity_threshold: float = 0.85,
    ) -> ConsolidationResult:
        """
        Run memory consolidation cycle.

        Args:
            importance_threshold: Minimum importance for promotion (0.0 to 1.0)
            similarity_threshold: Similarity threshold for clustering (0.0 to 1.0)

        Returns:
            ConsolidationResult with statistics
        """
        result = await self._call_tool("consolidate_memories", {
            "importance_threshold": importance_threshold,
            "similarity_threshold": similarity_threshold,
        })
        return ConsolidationResult.from_dict(result)

    async def store_memory(
        self,
        content: str,
        summary: Optional[str] = None,
        memory_type: str = "episodic",
        tags: Optional[List[str]] = None,
        emotional_context: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Store a new memory.

        Args:
            content: Memory content
            summary: Brief summary
            memory_type: Type of memory (episodic, semantic, procedural, emotional)
            tags: Optional tags
            emotional_context: Optional emotional state {valence, arousal}

        Returns:
            Dict with memory_id and status
        """
        arguments: Dict[str, Any] = {
            "content": content,
            "memory_type": memory_type,
        }
        if summary:
            arguments["summary"] = summary
        if tags:
            arguments["tags"] = tags
        if emotional_context:
            arguments["emotional_context"] = emotional_context

        return await self._call_tool("store_memory", arguments)

    async def search_memories(
        self,
        query: str,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search memories by semantic similarity.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of matching memories
        """
        result = await self._call_tool("search_memories", {
            "query": query,
            "n_results": n_results,
        })
        return result.get("results", [])

    async def get_memory_stats(self) -> Dict[str, int]:
        """
        Get memory statistics.

        Returns:
            Dict with total_memories, short_term, long_term counts
        """
        return await self._call_tool("get_memory_stats", {})
