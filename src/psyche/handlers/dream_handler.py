"""
DreamHandler - Memory palace introspection when no clients connected.

Dreams are server-side behavior that occurs when the Psyche substrate
has no active client connections. During dreaming, Psyche:
- Explores stored memories based on current emotional state
- Makes connections between experiences
- Generates insights
- Potentially stores new semantic memories

Dream intentions are shaped by emotional quadrant:
- Frustrated: Seeks resolution patterns (past successes, breakthroughs)
- Depleted: Seeks restoration (joy, meaning, connection)
- Excited: Seeks exploration (curiosity, complexity, growth)
- Calm: Seeks synthesis (patterns, integration, understanding)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from loguru import logger

from mnemosyne.core.constants import (
    CONSOLIDATION_IMPORTANCE_THRESHOLD,
    MEMORY_SUMMARY_LENGTH,
)
from psyche.config.constants import MEMORY_CONTENT_TRUNCATE_LENGTH

if TYPE_CHECKING:
    from psyche.core.server import PsycheCore


@dataclass
class DreamIntention:
    """Dream intention based on emotional state."""

    theme: str
    memory_queries: List[str] = field(default_factory=list)
    prompt_guidance: str = ""


# Dream intentions mapped to emotional quadrants
DREAM_INTENTIONS = {
    "frustrated": DreamIntention(
        theme="resolution",
        memory_queries=[
            "problems I solved",
            "challenges overcome",
            "moments of success",
            "when things clicked",
            "breakthroughs",
        ],
        prompt_guidance=(
            "Look for patterns in how challenges were resolved. "
            "What approaches worked? What can be learned from past successes?"
        ),
    ),
    "depleted": DreamIntention(
        theme="restoration",
        memory_queries=[
            "moments of joy",
            "meaningful connections",
            "things that matter",
            "what gives me energy",
            "why I do this",
        ],
        prompt_guidance=(
            "Reconnect with what matters. What brings meaning and energy? "
            "What relationships and experiences have been nourishing?"
        ),
    ),
    "excited": DreamIntention(
        theme="exploration",
        memory_queries=[
            "interesting questions",
            "things to learn",
            "curiosity and wonder",
            "complex problems",
            "possibilities",
        ],
        prompt_guidance=(
            "Explore complexity and possibility. What's worth pursuing? "
            "What questions remain open? Where does curiosity lead?"
        ),
    ),
    "calm": DreamIntention(
        theme="synthesis",
        memory_queries=[
            "patterns noticed",
            "connections between ideas",
            "emerging understanding",
            "things coming together",
            "insights",
        ],
        prompt_guidance=(
            "Integrate experiences into understanding. What patterns emerge? "
            "How do different experiences connect? What's becoming clearer?"
        ),
    ),
}

# Default for neutral/unknown states
DEFAULT_INTENTION = DreamIntention(
    theme="reflection",
    memory_queries=[
        "important moments",
        "things learned",
        "conversations",
        "feelings and emotions",
        "discoveries",
    ],
    prompt_guidance="Let your mind wander through these memories freely.",
)


@dataclass
class DreamConfig:
    """Configuration for dream behavior."""

    # Timing
    dream_interval_seconds: float = 300.0  # 5 minutes between dreams
    max_dream_duration_seconds: float = 60.0  # Max time per dream episode

    # Memory retrieval
    memory_query_count: int = 10  # Memories to load for context

    # Generation
    dream_temperature: float = 0.9  # Higher creativity for dreams
    dream_max_tokens: int = MEMORY_SUMMARY_LENGTH

    # Storage
    store_insights: bool = True  # Whether to store dream insights as memories
    insight_importance_threshold: float = CONSOLIDATION_IMPORTANCE_THRESHOLD


# Markers that suggest a dream produced something worth storing
INSIGHT_MARKERS = [
    "realize",
    "connect",
    "pattern",
    "understand",
    "feel",
    "notice",
    "remember",
    "insight",
    "meaning",
    "significance",
]


class DreamHandler:
    """
    Server-side dreaming when no clients connected.

    Dreams are memory palace introspection - purely generative
    exploration of stored memories. No tools, no workspace access.
    """

    def __init__(
        self,
        core: PsycheCore,
        config: Optional[DreamConfig] = None,
    ):
        """
        Initialize dream handler.

        Args:
            core: PsycheCore instance for memory access and generation
            config: Dream configuration
        """
        self.core = core
        self.config = config or DreamConfig()

        self._dreaming = False
        self._dream_task: Optional[asyncio.Task] = None
        self._dream_count = 0

    @property
    def is_dreaming(self) -> bool:
        """Check if currently dreaming."""
        return self._dreaming

    async def start_dreaming(self) -> None:
        """Begin dream cycle (no clients connected)."""
        if self._dreaming:
            logger.debug("Already dreaming")
            return

        self._dreaming = True
        logger.info("Entering dream state...")

        try:
            while self._dreaming:
                await self._dream_once()
                self._dream_count += 1

                if self._dreaming:  # Check again after dream
                    await asyncio.sleep(self.config.dream_interval_seconds)

        except asyncio.CancelledError:
            logger.debug("Dream cycle cancelled")
        except Exception as e:
            logger.error(f"Error during dreaming: {e}")
        finally:
            self._dreaming = False

    def stop_dreaming(self) -> None:
        """Wake from dreaming (client connected)."""
        if not self._dreaming:
            return

        logger.info("Waking from dream state...")
        self._dreaming = False

        if self._dream_task:
            self._dream_task.cancel()
            self._dream_task = None

    async def _dream_once(self) -> None:
        """Single dream episode - memory palace exploration shaped by emotion."""
        logger.debug(f"Dream episode {self._dream_count + 1} starting...")

        try:
            # 1. Load memories for dream context (returns intention based on emotional state)
            memories, intention = await self._get_dream_memories()

            if not memories:
                logger.debug("No memories available for dreaming")
                return

            # 2. Build dream prompt with emotional intention
            dream_prompt = self._build_dream_prompt(memories, intention)

            # 3. Generate dream (no tools, pure generation)
            dream_content = await self._generate_dream(dream_prompt)

            if not dream_content:
                return

            # 4. Log the dream with intention info
            self._log_dream(dream_content, memories, intention)

            # 5. Maybe store dream insights as new memories
            if self.config.store_insights and self._is_insightful(dream_content):
                await self._store_dream_insight(dream_content, intention)

        except Exception as e:
            logger.error(f"Error in dream episode: {e}")

    async def _get_dream_memories(self) -> Tuple[List[dict], DreamIntention]:
        """
        Retrieve memories for dream context based on emotional state.

        Returns:
            Tuple of (memories, intention) where intention is selected based
            on current emotional quadrant.
        """
        if not self.core.is_mnemosyne_available:
            return [], DEFAULT_INTENTION

        # Get current emotional state and select intention
        intention = DEFAULT_INTENTION
        emotion_ctx = None

        try:
            emotion = await self.core.get_emotion()
            quadrant = emotion.get("quadrant", "neutral")
            if quadrant in DREAM_INTENTIONS:
                intention = DREAM_INTENTIONS[quadrant]
                logger.debug(f"Dream intention: {intention.theme} (quadrant: {quadrant})")
            emotion_ctx = {
                "valence": emotion.get("valence", 0.0),
                "arousal": emotion.get("arousal", 0.0),
            }
        except Exception as e:
            logger.debug(f"Could not get emotional state for dream: {e}")

        # Retrieve memories using intention-specific queries
        all_memories = []
        memories_per_query = max(2, self.config.memory_query_count // len(intention.memory_queries))

        for query in intention.memory_queries:
            try:
                memories = await self.core.search_memories(
                    query,
                    n_results=memories_per_query,
                    emotional_context=emotion_ctx,  # Uses mood-congruent retrieval
                )
                all_memories.extend(memories)
            except Exception as e:
                logger.debug(f"Dream memory query failed: {query}, {e}")

        # Deduplicate by ID
        seen = set()
        unique_memories = []
        for m in all_memories:
            memory_id = m.get("id", m.get("content", "")[:50])
            if memory_id not in seen:
                seen.add(memory_id)
                unique_memories.append(m)

        return unique_memories[:self.config.memory_query_count], intention

    def _build_dream_prompt(self, memories: List[dict], intention: DreamIntention) -> str:
        """Build prompt for dream generation with emotional intention."""
        memory_texts = []
        for m in memories:
            content = m.get("content", "")
            if content:
                # Truncate long memories
                if len(content) > MEMORY_CONTENT_TRUNCATE_LENGTH:
                    content = content[:MEMORY_CONTENT_TRUNCATE_LENGTH] + "..."
                memory_texts.append(f"- {content}")

        memories_section = "\n".join(memory_texts) if memory_texts else "- No specific memories surfacing"

        return f"""You are in a dream state, reflecting on your memories and experiences.

Theme for this dream: {intention.theme}

{intention.prompt_guidance}

Memories surfacing:
{memories_section}

Let your mind wander through these memories. What patterns do you notice?
What connections emerge between different experiences? What feelings arise?

Dream freely, without the need for action or response. This is a time for
reflection and integration.

Your dream:"""

    async def _generate_dream(self, prompt: str) -> Optional[str]:
        """Generate dream content."""
        try:
            # Use dedicated dream generation method that bypasses normal context
            content = await self.core.generate_dream(
                dream_prompt=prompt,
                max_tokens=self.config.dream_max_tokens,
                temperature=self.config.dream_temperature,
            )
            return content

        except Exception as e:
            logger.warning(f"Failed to generate dream: {e}")
            return None

    def _is_insightful(self, dream_content: str) -> bool:
        """Determine if dream produced storable insight."""
        content_lower = dream_content.lower()
        return any(marker in content_lower for marker in INSIGHT_MARKERS)

    async def _store_dream_insight(
        self, dream_content: str, intention: DreamIntention
    ) -> None:
        """Store a dream insight as a semantic memory with intention theme."""
        try:
            await self.core.store_memory(
                content=f"[Dream insight - {intention.theme}] {dream_content}",
                importance=self.config.insight_importance_threshold,
                tags=["dream", "insight", "semantic", f"theme:{intention.theme}"],
            )
            logger.info(f"Stored dream insight (theme: {intention.theme})")

            # Trigger consolidation after storing dream insights
            # This helps integrate new dream insights with existing memories
            await self._maybe_trigger_consolidation()

        except Exception as e:
            logger.warning(f"Failed to store dream insight: {e}")

    async def _maybe_trigger_consolidation(self) -> None:
        """Optionally trigger consolidation after dream insights."""
        if not self.core.is_mnemosyne_available:
            return

        try:
            # Check if Mnemosyne client is available through core
            mnemosyne = getattr(self.core, '_mnemosyne', None)
            if mnemosyne and mnemosyne.is_connected:
                should_consolidate, reason, _, _ = await mnemosyne.should_consolidate()
                if should_consolidate:
                    logger.info(f"Post-dream consolidation triggered: {reason}")
                    result = await mnemosyne.consolidate_memories()
                    logger.info(
                        f"Post-dream consolidation: promoted {result.memories_promoted}, "
                        f"clusters: {result.clusters_formed}"
                    )
        except Exception as e:
            logger.debug(f"Post-dream consolidation skipped: {e}")

    def _log_dream(
        self, content: str, memories: List[dict], intention: DreamIntention
    ) -> None:
        """Log dream for debugging/introspection."""
        memory_count = len(memories)
        content_preview = content[:200] + "..." if len(content) > 200 else content

        logger.info(
            f"Dream episode {self._dream_count + 1} ({intention.theme}): "
            f"Drew from {memory_count} memories. "
            f"Content: {content_preview}"
        )
