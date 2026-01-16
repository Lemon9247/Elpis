"""
PsycheClient - Abstract interface for connecting to Psyche Core.

Supports both local (in-process) and remote (HTTP/MCP) connections.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from psyche.core.server import PsycheCore


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


class LocalPsycheClient(PsycheClient):
    """
    Direct in-process connection to PsycheCore.

    This is the default client for single-user desktop use.
    It delegates all operations directly to a local PsycheCore instance.
    """

    def __init__(self, core: PsycheCore):
        """
        Initialize the local client.

        Args:
            core: The PsycheCore instance to delegate to
        """
        self._core = core

    @property
    def core(self) -> PsycheCore:
        """Access the underlying PsycheCore instance."""
        return self._core

    async def add_user_message(self, content: str) -> Optional[str]:
        """Add user message, returns memory context if retrieved."""
        return await self._core.add_user_message(content)

    async def add_assistant_message(
        self,
        content: str,
        user_message: str = "",
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add assistant message with optional tool results."""
        await self._core.add_assistant_message(content, user_message, tool_results)

    def add_tool_result(self, tool_name: str, result: str) -> None:
        """Add a tool result to context."""
        self._core.add_tool_result(tool_name, result)

    async def generate(
        self,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate a response."""
        return await self._core.generate(max_tokens, temperature)

    async def generate_stream(
        self,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> AsyncIterator[str]:
        """Stream a response token by token."""
        async for token in self._core.generate_stream(max_tokens, temperature, on_token):
            yield token

    async def retrieve_memories(self, query: str, n: int = 3) -> List[Dict[str, Any]]:
        """Retrieve memories."""
        return await self._core.retrieve_memories(query, n)

    async def store_memory(
        self,
        content: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Store a memory."""
        return await self._core.store_memory(content, importance, tags)

    async def get_emotion(self) -> Dict[str, Any]:
        """Get emotional state."""
        return await self._core.get_emotion()

    async def update_emotion(
        self,
        event_type: str,
        intensity: float = 1.0,
    ) -> Dict[str, Any]:
        """Update emotional state."""
        return await self._core.update_emotion(event_type, intensity)

    def set_reasoning_mode(self, enabled: bool) -> None:
        """Toggle reasoning mode."""
        self._core.set_reasoning_mode(enabled)

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        await self._core.shutdown()

    def clear_context(self) -> None:
        """Clear working memory context."""
        self._core.clear_context()

    def initialize(self) -> None:
        """Initialize the core with system prompt."""
        self._core.initialize()

    def set_tool_descriptions(self, descriptions: str) -> None:
        """Set tool descriptions for the system prompt."""
        self._core.set_tool_descriptions(descriptions)

    def get_api_messages(self) -> List[Dict[str, str]]:
        """Get current messages formatted for API calls."""
        return self._core.get_api_messages()

    @property
    def reasoning_enabled(self) -> bool:
        """Check if reasoning is enabled."""
        return self._core.reasoning_enabled

    @property
    def context_summary(self) -> Dict[str, Any]:
        """Get context summary."""
        return self._core.context_summary

    @property
    def is_mnemosyne_available(self) -> bool:
        """Check if Mnemosyne is available."""
        return self._core.is_mnemosyne_available


class RemotePsycheClient(PsycheClient):
    """
    HTTP connection to remote Psyche Server.

    Stub for Phase 5 - raises NotImplementedError for now.
    This will enable multi-client server mode where multiple
    TUI instances can connect to a shared Psyche server.
    """

    def __init__(self, base_url: str):
        """
        Initialize the remote client.

        Args:
            base_url: Base URL of the Psyche server (e.g., "http://localhost:8080")
        """
        self.base_url = base_url

    async def add_user_message(self, content: str) -> Optional[str]:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    async def add_assistant_message(
        self,
        content: str,
        user_message: str = "",
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    def add_tool_result(self, tool_name: str, result: str) -> None:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    async def generate(
        self,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    async def generate_stream(
        self,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> AsyncIterator[str]:
        raise NotImplementedError("Remote client will be implemented in Phase 5")
        # Make this a generator (required for async generator type)
        if False:
            yield ""

    async def retrieve_memories(self, query: str, n: int = 3) -> List[Dict[str, Any]]:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    async def store_memory(
        self,
        content: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> bool:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    async def get_emotion(self) -> Dict[str, Any]:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    async def update_emotion(
        self,
        event_type: str,
        intensity: float = 1.0,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    def set_reasoning_mode(self, enabled: bool) -> None:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    async def shutdown(self) -> None:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    def clear_context(self) -> None:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    def initialize(self) -> None:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    def set_tool_descriptions(self, descriptions: str) -> None:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    def get_api_messages(self) -> List[Dict[str, str]]:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    @property
    def reasoning_enabled(self) -> bool:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    @property
    def context_summary(self) -> Dict[str, Any]:
        raise NotImplementedError("Remote client will be implemented in Phase 5")

    @property
    def is_mnemosyne_available(self) -> bool:
        raise NotImplementedError("Remote client will be implemented in Phase 5")
