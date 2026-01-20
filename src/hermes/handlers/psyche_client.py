"""
PsycheClient - Abstract interface for connecting to Psyche Core.

Provides HTTP client for connecting to a remote Psyche server.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Dict, List, Optional


class PsycheClient(ABC):
    """
    Abstract interface for Psyche Core connections.

    This abstraction allows the TUI (and other clients) to connect to
    PsycheCore either:
    - Locally (in-process) - for single-user desktop use
    - Remotely (HTTP/MCP) - for future multi-client server mode

    All methods are async to support both local and network operations.
    """

    @abstractmethod
    async def add_user_message(self, content: str) -> Optional[str]:
        """
        Add user message and retrieve relevant memories.

        Args:
            content: The user's message

        Returns:
            Formatted memory context string, or None if no memories found
        """
        ...

    @abstractmethod
    async def add_assistant_message(
        self,
        content: str,
        user_message: str = "",
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add assistant message with optional tool results.

        Args:
            content: The assistant's response
            user_message: The original user message (for importance scoring)
            tool_results: Any tool results from this exchange
        """
        ...

    @abstractmethod
    def add_tool_result(self, tool_name: str, result: str) -> None:
        """
        Add a tool result to context.

        Args:
            tool_name: Name of the tool that was executed
            result: The tool result (formatted as string)
        """
        ...

    @abstractmethod
    async def generate(
        self,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response.

        Args:
            max_tokens: Maximum tokens to generate
            temperature: Override temperature (None = emotionally modulated)

        Returns:
            Dict with:
            - content: str - The response text
            - thinking: str - Extracted reasoning (if any)
            - has_thinking: bool - Whether reasoning was found
            - emotional_state: EmotionalState - Current emotion
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response token by token.

        Args:
            max_tokens: Maximum tokens to generate
            temperature: Override temperature
            on_token: Optional callback for each token

        Yields:
            Individual tokens as they become available
        """
        ...

    @abstractmethod
    async def retrieve_memories(self, query: str, n: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve memories by query.

        Args:
            query: Search query
            n: Number of memories to retrieve

        Returns:
            List of memory dictionaries
        """
        ...

    @abstractmethod
    async def store_memory(
        self,
        content: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Store a memory.

        Args:
            content: Memory content to store
            importance: Importance score (0.0 to 1.0)
            tags: Optional tags for categorization

        Returns:
            True if stored successfully
        """
        ...

    @abstractmethod
    async def get_emotion(self) -> Dict[str, Any]:
        """
        Get current emotional state.

        Returns:
            Dict with valence, arousal, quadrant
        """
        ...

    @abstractmethod
    async def update_emotion(
        self,
        event_type: str,
        intensity: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Update emotional state.

        Args:
            event_type: Type of emotional event
            intensity: Event intensity multiplier

        Returns:
            Updated emotional state dict
        """
        ...

    @abstractmethod
    def set_reasoning_mode(self, enabled: bool) -> None:
        """
        Toggle reasoning mode.

        Args:
            enabled: Whether to enable reasoning mode
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Graceful shutdown with memory consolidation."""
        ...

    @abstractmethod
    def clear_context(self) -> None:
        """Clear working memory context."""
        ...

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the core with system prompt."""
        ...

    @abstractmethod
    def set_tool_descriptions(self, descriptions: str) -> None:
        """
        Set tool descriptions for the system prompt.

        Args:
            descriptions: Formatted tool description string
        """
        ...

    @abstractmethod
    def get_api_messages(self) -> List[Dict[str, str]]:
        """Get current messages formatted for API calls."""
        ...

    @property
    @abstractmethod
    def reasoning_enabled(self) -> bool:
        """Check if reasoning mode is enabled."""
        ...

    @property
    @abstractmethod
    def context_summary(self) -> Dict[str, Any]:
        """
        Get context summary.

        Returns:
            Dict with message_count, total_tokens, available_tokens, etc.
        """
        ...

    @property
    @abstractmethod
    def is_mnemosyne_available(self) -> bool:
        """Check if Mnemosyne memory server is available."""
        ...


class RemotePsycheClient(PsycheClient):
    """
    HTTP connection to remote Psyche Server.

    Connects to a Psyche server via OpenAI-compatible HTTP API.
    This enables multi-client mode where multiple TUI instances
    can share the same Psyche substrate.

    The server maintains conversation state - this client just
    sends requests and receives responses.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8741"):
        """
        Initialize the remote client.

        Args:
            base_url: Base URL of the Psyche server
        """
        self.base_url = base_url.rstrip("/")
        self._session: Optional[Any] = None  # aiohttp.ClientSession
        self._connected = False

        # Local state mirrors
        self._tool_descriptions: str = ""
        self._reasoning_enabled: bool = True
        self._messages: List[Dict[str, str]] = []  # Local message history
        self._tools: List[Dict[str, Any]] = []  # Tool definitions in OpenAI format

        # Cached status (refreshed on demand)
        self._cached_status: Dict[str, Any] = {}
        self._mnemosyne_available: bool = False

        # Tool call state from last response
        self._last_tool_calls: Optional[List[Dict[str, Any]]] = None
        self._last_finish_reason: Optional[str] = None

    async def connect(self) -> None:
        """Establish connection to the server."""
        import aiohttp

        self._session = aiohttp.ClientSession()

        # Verify connection with health check
        try:
            async with self._session.get(f"{self.base_url}/health") as resp:
                if resp.status != 200:
                    raise ConnectionError(f"Server returned {resp.status}")
                data = await resp.json()
                self._connected = True

                # Get initial status
                await self._refresh_status()

        except aiohttp.ClientError as e:
            await self._session.close()
            self._session = None
            raise ConnectionError(f"Cannot connect to Psyche server: {e}")

    async def disconnect(self) -> None:
        """Close connection to the server."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    async def _ensure_connected(self) -> None:
        """Ensure we have an active connection."""
        if not self._session:
            await self.connect()

    async def _refresh_status(self) -> None:
        """Refresh cached status from server."""
        # Use a simple request to check status
        # In future, could add a /status endpoint
        pass

    async def _chat_request(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
    ) -> Any:
        """Send chat completion request to server."""
        await self._ensure_connected()

        payload = {
            "model": "psyche",
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
        }

        if temperature is not None:
            payload["temperature"] = temperature

        if self._tools:
            payload["tools"] = self._tools

        return await self._session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )

    async def add_user_message(self, content: str) -> Optional[str]:
        """Add user message. Returns memory context if retrieved."""
        self._messages.append({"role": "user", "content": content})
        # Memory retrieval happens server-side, we don't get direct feedback
        return None

    async def add_assistant_message(
        self,
        content: str,
        user_message: str = "",
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add assistant message to local history."""
        self._messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_name: str, result: str) -> None:
        """Add a tool result to conversation."""
        self._messages.append({
            "role": "tool",
            "name": tool_name,
            "content": result,
        })

    async def generate(
        self,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate a response from the server."""
        async with await self._chat_request(
            self._messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"Server error: {error}")

            data = await resp.json()

            # Extract response
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")

            # Add to local history
            self._messages.append({"role": "assistant", "content": content})

            return {
                "content": content,
                "thinking": "",
                "has_thinking": False,
                "tool_calls": message.get("tool_calls"),
            }

    async def generate_stream(
        self,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> AsyncIterator[str]:
        """Stream a response token by token."""
        import json

        # Reset tool call state
        self._last_tool_calls = None
        self._last_finish_reason = None

        async with await self._chat_request(
            self._messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"Server error: {error}")

            full_content = ""

            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta", {})

                    # Check for content token
                    token = delta.get("content", "")
                    if token:
                        full_content += token
                        if on_token:
                            on_token(token)
                        yield token

                    # Check for finish reason and tool_calls
                    finish_reason = choice.get("finish_reason")
                    if finish_reason:
                        self._last_finish_reason = finish_reason
                        # Tool calls come in the delta on the finish chunk
                        if "tool_calls" in delta:
                            self._last_tool_calls = delta["tool_calls"]

                except json.JSONDecodeError:
                    continue

            # Add complete response to history (only if no tool calls)
            # When there are tool calls, Hermes handles adding to history
            if full_content and not self._last_tool_calls:
                self._messages.append({"role": "assistant", "content": full_content})

    def get_pending_tool_calls(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get tool_calls from the last streamed response, if any.

        Returns:
            List of tool call dictionaries if finish_reason was "tool_calls",
            None otherwise.
        """
        if self._last_finish_reason == "tool_calls" and self._last_tool_calls:
            return self._last_tool_calls
        return None

    async def retrieve_memories(self, query: str, n: int = 3) -> List[Dict[str, Any]]:
        """Retrieve memories via server."""
        # For now, return empty - would need MCP endpoint
        return []

    async def store_memory(
        self,
        content: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Store a memory via server."""
        # For now, return False - would need MCP endpoint
        return False

    async def get_emotion(self) -> Dict[str, Any]:
        """Get emotional state from server."""
        await self._ensure_connected()
        try:
            async with self._session.get(f"{self.base_url}/v1/psyche/emotion") as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"valence": 0.0, "arousal": 0.0, "quadrant": "neutral"}
        except Exception:
            return {"valence": 0.0, "arousal": 0.0, "quadrant": "neutral"}

    async def update_emotion(
        self,
        event_type: str,
        intensity: float = 1.0,
    ) -> Dict[str, Any]:
        """Update emotional state on server."""
        await self._ensure_connected()
        try:
            async with self._session.post(
                f"{self.base_url}/v1/psyche/emotion",
                json={"event_type": event_type, "intensity": intensity},
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"valence": 0.0, "arousal": 0.0, "quadrant": "neutral"}
        except Exception:
            return {"valence": 0.0, "arousal": 0.0, "quadrant": "neutral"}

    def set_reasoning_mode(self, enabled: bool) -> None:
        """Toggle reasoning mode (local flag, server controls actual behavior)."""
        self._reasoning_enabled = enabled

    async def shutdown(self) -> None:
        """Disconnect from server."""
        await self.disconnect()

    def clear_context(self) -> None:
        """Clear local message history."""
        self._messages = []

    def initialize(self) -> None:
        """Initialize is a no-op for remote client (server is already initialized)."""
        pass

    def set_tool_descriptions(self, descriptions: str) -> None:
        """Store tool descriptions for sending with requests."""
        self._tool_descriptions = descriptions

    def set_tools(self, tools: List[Dict[str, Any]]) -> None:
        """
        Set tool definitions in OpenAI format.

        Args:
            tools: List of tool definitions in OpenAI format
        """
        self._tools = tools

    def get_api_messages(self) -> List[Dict[str, str]]:
        """Get local message history."""
        return list(self._messages)

    @property
    def reasoning_enabled(self) -> bool:
        """Check if reasoning is enabled."""
        return self._reasoning_enabled

    @property
    def context_summary(self) -> Dict[str, Any]:
        """Get context summary (local approximation)."""
        return {
            "message_count": len(self._messages),
            "remote": True,
            "server_url": self.base_url,
        }

    @property
    def is_mnemosyne_available(self) -> bool:
        """Check if Mnemosyne is available (cached from server)."""
        return self._mnemosyne_available

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._connected
