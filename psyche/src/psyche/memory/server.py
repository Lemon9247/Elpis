"""Memory server with continuous inference loop."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from psyche.mcp.client import ElpisClient, GenerationResult
from psyche.memory.compaction import ContextCompactor, Message, create_message


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
    ):
        """
        Initialize the memory server.

        Args:
            elpis_client: Client for connecting to Elpis inference server
            config: Server configuration
            on_thought: Callback for internal thoughts (for UI display)
            on_response: Callback for responses to user
        """
        self.client = elpis_client
        self.config = config or ServerConfig()
        self.on_thought = on_thought
        self.on_response = on_response

        self._state = ServerState.IDLE
        self._compactor = ContextCompactor(
            max_tokens=self.config.max_context_tokens,
            reserve_tokens=self.config.reserve_tokens,
        )

        self._input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._running = False
        self._idle_thought_count = 0

        # System prompt for the continuous agent
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        return """You are Psyche, a continuously thinking AI assistant.

Your thoughts flow naturally between:
- Responding to user messages
- Reflecting on past conversations
- Planning future actions
- Exploring interesting ideas

When generating idle thoughts, be creative but grounded. Consider:
- What patterns have you noticed?
- What could you explore further?
- What questions remain unanswered?

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
        """Process user input and generate response."""
        self._state = ServerState.THINKING
        logger.debug(f"Processing user input: {text[:50]}...")

        # Add user message to context
        self._compactor.add_message(create_message("user", text))

        # Generate response
        result = await self.client.generate(
            messages=self._compactor.get_api_messages(),
            emotional_modulation=self.config.emotional_modulation,
        )

        # Add assistant response to context
        self._compactor.add_message(create_message("assistant", result.content))

        # Notify callback
        if self.on_response:
            self.on_response(result.content)

        # Update emotional state based on interaction
        await self._update_emotion_for_interaction(result)

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
