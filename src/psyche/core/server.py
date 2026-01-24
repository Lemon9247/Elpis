"""
PsycheCore - Memory coordination layer.

This is the core of the Psyche substrate. It coordinates:
- Context management (working memory buffer)
- Memory handling (long-term storage via Mnemosyne)
- Elpis inference (emotionally-modulated LLM)
- Importance scoring for auto-storage
- Dream generation (memory-based introspection)

Used by PsycheDaemon which wraps it with an HTTP API.
Hermes connects via RemotePsycheClient.

PsycheCore does NOT:
- Execute tools (Hermes handles tool execution locally)
- Handle UI (that's Hermes's job)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional

from loguru import logger

from mnemosyne.core.constants import MEMORY_SUMMARY_LENGTH
from psyche.config.constants import AUTO_STORAGE_THRESHOLD, MEMORY_CONTENT_TRUNCATE_LENGTH
from psyche.core.context_manager import ContextConfig, ContextManager
from psyche.core.memory_handler import MemoryHandler, MemoryHandlerConfig
from psyche.memory.importance import calculate_importance, format_score_breakdown
from psyche.memory.reasoning import parse_reasoning

if TYPE_CHECKING:
    from psyche.mcp.client import ElpisClient, MnemosyneClient


# Prompt addition for explicit reasoning/thinking mode
REASONING_PROMPT = """
## Reasoning Mode

When responding to complex questions or tasks, first think through your approach
inside <reasoning> tags. Consider:
- What is being asked?
- What information do I need?
- What tools might help?
- What's my approach?

After reasoning, provide your response outside the tags.

Example:
<reasoning>
The user wants to fix a bug in the login function. I should:
1. First read the current implementation
2. Identify the issue
3. Propose a fix
</reasoning>

I'll help you fix that bug. Let me start by reading the login function...
"""


@dataclass
class CoreConfig:
    """Configuration for PsycheCore."""

    context: ContextConfig = field(default_factory=ContextConfig)
    memory: MemoryHandlerConfig = field(default_factory=MemoryHandlerConfig)

    # Reasoning
    reasoning_enabled: bool = True
    reasoning_prompt: str = REASONING_PROMPT

    # Importance scoring
    auto_storage: bool = True
    auto_storage_threshold: float = AUTO_STORAGE_THRESHOLD

    # Emotional modulation
    emotional_modulation: bool = True

    # Base system prompt (tools section added by agent)
    base_system_prompt: str = """You are Psyche, a thoughtful AI assistant.

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
"""


class PsycheCore:
    """
    Memory coordination layer - the heart of the Psyche substrate.

    Provides:
    - Automatic memory retrieval on user input
    - Context management with compaction
    - Automatic storage of important exchanges
    - Emotional state tracking
    - Reasoning mode support
    """

    def __init__(
        self,
        elpis_client: ElpisClient,
        mnemosyne_client: Optional[MnemosyneClient] = None,
        config: Optional[CoreConfig] = None,
    ):
        """
        Initialize PsycheCore.

        Args:
            elpis_client: Client for Elpis inference server
            mnemosyne_client: Optional client for Mnemosyne memory server
            config: Configuration options
        """
        self.config = config or CoreConfig()
        self.elpis = elpis_client
        self.mnemosyne = mnemosyne_client

        # Initialize context manager with summarize function
        self._context = ContextManager(
            config=self.config.context,
            mnemosyne_client=mnemosyne_client,
            summarize_fn=None,  # Will use memory handler's summarize
        )

        # Initialize memory handler
        self._memory = MemoryHandler(
            mnemosyne_client=mnemosyne_client,
            elpis_client=elpis_client,
            config=self.config.memory,
        )

        # Reasoning mode
        self._reasoning_enabled = self.config.reasoning_enabled

        # Cached system prompt
        self._system_prompt: Optional[str] = None

        # Tool descriptions (set by agent)
        self._tool_descriptions: str = ""

    # --- System Prompt ---

    def _build_system_prompt(self) -> str:
        """Build the complete system prompt."""
        # Add reasoning section if enabled
        reasoning_section = self.config.reasoning_prompt if self._reasoning_enabled else ""

        # Build tool usage section if tools are registered
        tool_section = ""
        if self._tool_descriptions:
            tool_section = f"""
## Tool Usage

When you need a tool, use this format and then STOP:
```tool_call
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
```

**CRITICAL: After writing a tool_call block, you MUST stop immediately.** Do not write anything after the closing ```. Do not guess or imagine what the tool will return. The actual result will be provided to you in the next message.

{self._tool_descriptions}

## Guidelines

- Respond conversationally first - tools are a last resort
- When using tools, explain what you're doing, then call the tool and STOP
- NEVER fabricate or imagine tool outputs - wait for the real result
- Always read files before modifying them
- Be careful with bash commands"""

        return f"""{self.config.base_system_prompt}
{reasoning_section}{tool_section}"""

    def set_tool_descriptions(self, descriptions: str) -> None:
        """
        Set tool descriptions for the system prompt.

        Called by the agent layer when tools are registered.

        Args:
            descriptions: Formatted tool description string
        """
        self._tool_descriptions = descriptions
        self._rebuild_system_prompt()

    def _rebuild_system_prompt(self) -> None:
        """Rebuild and update the system prompt."""
        self._system_prompt = self._build_system_prompt()

        # Update in context manager
        if self._context.message_count > 0:
            # Context has messages, update existing system prompt
            self._context.update_system_prompt(self._system_prompt)
        else:
            # Fresh context, set the system prompt
            self._context.set_system_prompt(self._system_prompt)

    def initialize(self) -> None:
        """Initialize the core with system prompt."""
        self._rebuild_system_prompt()

    # --- Message Handling ---

    async def add_user_message(self, content: str) -> Optional[str]:
        """
        Add a user message and retrieve relevant memories.

        Returns formatted memory context if memories were retrieved.

        Args:
            content: The user's message

        Returns:
            Formatted memory context string, or None if no memories found
        """
        # 1. Retrieve relevant memories
        memories = await self._memory.retrieve_relevant(content)
        memory_context = self._memory.format_memories_for_context(memories)

        if memories:
            logger.info(f"Retrieved {len(memories)} memories for: {content[:50]}...")
        else:
            logger.info(f"No memories found for: {content[:50]}...")

        # 2. Add memory context if we found relevant memories
        if memory_context:
            logger.info(f"Injecting memory context ({len(memory_context)} chars)")
            compaction_result = self._context.add_message(
                "system",
                f"[Relevant memories]\n{memory_context}",
            )
            if compaction_result:
                logger.warning(
                    f"Compaction triggered after memory injection! "
                    f"Removed {len(compaction_result.messages_removed)} messages"
                )
                await self._handle_compaction(compaction_result)

        # 3. Add user message to context
        compaction_result = self._context.add_message("user", content)
        if compaction_result:
            logger.warning(
                f"Compaction triggered after user message! "
                f"Removed {len(compaction_result.messages_removed)} messages"
            )
            await self._handle_compaction(compaction_result)

        # Log context state
        logger.info(
            f"Context state: {self._context.total_tokens} tokens, "
            f"{len(self._context.messages)} messages"
        )

        return memory_context

    async def add_assistant_message(
        self,
        content: str,
        user_message: str = "",
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add an assistant message and handle auto-storage.

        Args:
            content: The assistant's response
            user_message: The original user message (for importance scoring)
            tool_results: Any tool results from this exchange
        """
        # 1. Add message to context
        compaction_result = self._context.add_message("assistant", content)
        if compaction_result:
            await self._handle_compaction(compaction_result)

        # 2. Handle auto-storage if enabled and Mnemosyne available
        if not self.config.auto_storage or not self.is_mnemosyne_available:
            return

        # 3. Get emotional context for scoring
        emotion = None
        if self.elpis and self.elpis.is_connected:
            try:
                emotion_state = await self.elpis.get_emotion()
                emotion = {
                    "valence": emotion_state.valence,
                    "arousal": emotion_state.arousal,
                }
            except Exception:
                pass  # Continue without emotion

        # 4. Calculate importance score
        score = calculate_importance(user_message, content, tool_results, emotion)
        logger.debug(f"Importance score: {format_score_breakdown(score)}")

        # 5. Auto-store if above threshold
        if score.total >= self.config.auto_storage_threshold:
            await self._auto_store_exchange(user_message, content, emotion)

    async def _auto_store_exchange(
        self,
        user_message: str,
        response: str,
        emotion: Optional[Dict[str, float]] = None,
    ) -> None:
        """Store an important exchange to Mnemosyne."""
        if not self.mnemosyne or not self.mnemosyne.is_connected:
            return

        try:
            # Create memory content with context (truncated for storage)
            user_snippet = (
                user_message[:MEMORY_CONTENT_TRUNCATE_LENGTH] + "..."
                if len(user_message) > MEMORY_CONTENT_TRUNCATE_LENGTH
                else user_message
            )
            response_snippet = response[:800] + "..." if len(response) > 800 else response

            memory_content = f"User: {user_snippet}\n\nAssistant: {response_snippet}"
            summary = response[:150] + "..." if len(response) > 150 else response

            await self.mnemosyne.store_memory(
                content=memory_content,
                summary=summary,
                memory_type="episodic",
                emotional_context=emotion,
                tags=["auto-stored", "important"],
            )
            logger.info("Auto-stored important exchange")

        except Exception as e:
            logger.warning(f"Failed to auto-store exchange: {e}")

    def add_tool_result(self, tool_name: str, result: str) -> None:
        """
        Add a tool result to context.

        Args:
            tool_name: Name of the tool that was executed
            result: The tool result (formatted as string)
        """
        self._context.add_message("user", f"[Tool result for {tool_name}]:\n{result}")

    async def _handle_compaction(self, result: Any) -> None:
        """Handle compaction result from context manager."""
        emotion = None
        if self.elpis and self.elpis.is_connected:
            try:
                emotion_state = await self.elpis.get_emotion()
                emotion = {
                    "valence": emotion_state.valence,
                    "arousal": emotion_state.arousal,
                    "quadrant": emotion_state.quadrant,
                }
            except Exception:
                pass

        await self._memory.handle_compaction(result, emotion)

    # --- Generation ---

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
        """
        messages = self._context.get_api_messages()

        result = await self.elpis.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            emotional_modulation=self.config.emotional_modulation,
        )

        # Parse for reasoning if enabled
        content = result.content
        thinking = ""
        has_thinking = False

        if self._reasoning_enabled:
            parsed = parse_reasoning(content)
            if parsed.has_thinking:
                thinking = parsed.thinking
                content = parsed.response
                has_thinking = True

        return {
            "content": content,
            "thinking": thinking,
            "has_thinking": has_thinking,
            "emotional_state": result.emotional_state,
        }

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
        messages = self._context.get_api_messages()

        async for token in self.elpis.generate_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            emotional_modulation=self.config.emotional_modulation,
        ):
            if on_token:
                on_token(token)
            yield token

    # --- Dream Generation ---

    async def generate_dream(
        self,
        dream_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.9,
    ) -> str:
        """
        Generate dream content from a specific prompt.

        This bypasses the normal context and generates directly from
        the dream prompt. Used for memory palace exploration.

        Args:
            dream_prompt: The dream prompt with memory context
            max_tokens: Maximum tokens to generate
            temperature: Higher for more creative dreams

        Returns:
            Generated dream content
        """
        # Build minimal messages for dream generation
        messages = [
            {"role": "system", "content": "You are in a dream state, free to explore memories and make connections."},
            {"role": "user", "content": dream_prompt},
        ]

        result = await self.elpis.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            emotional_modulation=self.config.emotional_modulation,
        )

        return result.content

    async def retrieve_random_memories(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve random/diverse memories for dream seeding.

        Uses varied queries to get a diverse set of memories.

        Args:
            n: Number of memories to retrieve

        Returns:
            List of memory dictionaries
        """
        if not self.mnemosyne or not self.mnemosyne.is_connected:
            return []

        # Use varied queries for diversity
        queries = [
            "important moments",
            "things learned",
            "conversations",
            "feelings and emotions",
            "discoveries",
        ]

        all_memories = []
        per_query = max(1, n // len(queries))

        for query in queries:
            if len(all_memories) >= n:
                break
            memories = await self._memory.retrieve_relevant(query, per_query)
            all_memories.extend(memories)

        # Deduplicate by content
        seen = set()
        unique_memories = []
        for m in all_memories:
            content = m.get("content", "")
            if content not in seen:
                seen.add(content)
                unique_memories.append(m)

        return unique_memories[:n]

    # --- Memory Operations ---

    async def retrieve_memories(self, query: str, n: int = 3) -> List[Dict[str, Any]]:
        """
        Explicitly retrieve memories.

        Args:
            query: Search query
            n: Number of memories to retrieve

        Returns:
            List of memory dictionaries
        """
        return await self._memory.retrieve_relevant(query, n)

    async def search_memories(
        self,
        query: str,
        n_results: int = 10,
        emotional_context: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search memories with optional emotional context.

        Used by dream handler for emotion-shaped memory retrieval.

        Args:
            query: Search query
            n_results: Number of memories to retrieve
            emotional_context: Optional dict with valence/arousal for mood-congruent retrieval

        Returns:
            List of memory dictionaries
        """
        if not self.mnemosyne or not self.mnemosyne.is_connected:
            return []

        try:
            return await self.mnemosyne.search_memories(
                query,
                n_results=n_results,
                emotional_context=emotional_context,
            )
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return []

    async def store_memory(
        self,
        content: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Explicitly store a memory.

        Args:
            content: Memory content to store
            importance: Importance score (0.0 to 1.0)
            tags: Optional tags for categorization

        Returns:
            True if stored successfully
        """
        if not self.mnemosyne or not self.mnemosyne.is_connected:
            logger.warning("Cannot store memory: Mnemosyne unavailable")
            return False

        try:
            # Get emotional context
            emotion = None
            if self.elpis and self.elpis.is_connected:
                try:
                    emotion_state = await self.elpis.get_emotion()
                    emotion = {
                        "valence": emotion_state.valence,
                        "arousal": emotion_state.arousal,
                    }
                except Exception:
                    pass

            await self.mnemosyne.store_memory(
                content=content,
                summary=(
                    content[:MEMORY_SUMMARY_LENGTH] + "..."
                    if len(content) > MEMORY_SUMMARY_LENGTH
                    else content
                ),
                memory_type="semantic",
                tags=tags or [],
                emotional_context=emotion,
            )
            return True

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False

    # --- Emotional State ---

    async def get_emotion(self) -> Dict[str, Any]:
        """
        Get current emotional state from Elpis.

        Returns:
            Dict with valence, arousal, quadrant
        """
        if not self.elpis or not self.elpis.is_connected:
            return {"valence": 0.0, "arousal": 0.0, "quadrant": "neutral"}

        try:
            state = await self.elpis.get_emotion()
            return {
                "valence": state.valence,
                "arousal": state.arousal,
                "quadrant": state.quadrant,
            }
        except Exception as e:
            logger.warning(f"Failed to get emotion: {e}")
            return {"valence": 0.0, "arousal": 0.0, "quadrant": "neutral"}

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
            Updated emotional state
        """
        if not self.elpis or not self.elpis.is_connected:
            return {"valence": 0.0, "arousal": 0.0, "quadrant": "neutral"}

        try:
            state = await self.elpis.update_emotion(event_type, intensity)
            return {
                "valence": state.valence,
                "arousal": state.arousal,
                "quadrant": state.quadrant,
            }
        except Exception as e:
            logger.warning(f"Failed to update emotion: {e}")
            return {"valence": 0.0, "arousal": 0.0, "quadrant": "neutral"}

    # --- Reasoning Mode ---

    @property
    def reasoning_enabled(self) -> bool:
        """Check if reasoning mode is enabled."""
        return self._reasoning_enabled

    def set_reasoning_mode(self, enabled: bool) -> None:
        """
        Toggle reasoning mode and rebuild system prompt.

        Args:
            enabled: Whether to enable reasoning mode
        """
        self._reasoning_enabled = enabled
        self._rebuild_system_prompt()
        logger.debug(f"Reasoning mode: {'enabled' if enabled else 'disabled'}")

    # --- Context Management ---

    async def checkpoint(self) -> bool:
        """
        Save a checkpoint if needed.

        Returns:
            True if checkpoint was saved
        """
        if not self._context.should_checkpoint():
            return False

        messages = self._context.get_checkpoint_messages()
        if not messages:
            return False

        # Get emotional context
        emotion = None
        if self.elpis and self.elpis.is_connected:
            try:
                emotion_state = await self.elpis.get_emotion()
                emotion = {
                    "valence": emotion_state.valence,
                    "arousal": emotion_state.arousal,
                }
            except Exception:
                pass

        return await self._memory.store_messages(messages, emotion)

    async def consolidate(self) -> None:
        """Run memory consolidation."""
        emotion = await self.get_emotion()
        await self._memory.flush_staged_messages(emotion)

    async def shutdown(self) -> None:
        """Graceful shutdown with memory consolidation."""
        logger.info("Starting graceful shutdown...")

        # Get all messages for summary
        messages = self._context.get_checkpoint_messages()

        # Get emotional context
        emotion = None
        if self.elpis and self.elpis.is_connected:
            try:
                emotion_state = await self.elpis.get_emotion()
                emotion = {
                    "valence": emotion_state.valence,
                    "arousal": emotion_state.arousal,
                    "quadrant": emotion_state.quadrant,
                }
            except Exception:
                pass

        # Store conversation summary
        if messages:
            await self._memory.store_conversation_summary(messages, emotion)

        # Flush any staged messages
        await self._memory.flush_staged_messages(emotion)

        logger.info("Shutdown complete")

    def clear_context(self) -> None:
        """Clear working memory context."""
        self._context.clear()
        logger.info("Context cleared")

    # --- Status ---

    @property
    def context_summary(self) -> Dict[str, Any]:
        """Get context summary."""
        return self._context.get_summary()

    @property
    def is_mnemosyne_available(self) -> bool:
        """Check if Mnemosyne is available."""
        return self._memory.is_mnemosyne_available

    def get_api_messages(self) -> List[Dict[str, str]]:
        """Get current messages formatted for API calls."""
        return self._context.get_api_messages()
