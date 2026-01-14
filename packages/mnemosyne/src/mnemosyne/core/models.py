"""Core memory data structures."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class MemoryType(Enum):
    """Classification of memory types."""
    EPISODIC = "episodic"      # Specific events/conversations
    SEMANTIC = "semantic"       # General knowledge/facts
    PROCEDURAL = "procedural"   # How to do things
    EMOTIONAL = "emotional"     # Emotional associations


class MemoryStatus(Enum):
    """Memory lifecycle status."""
    SHORT_TERM = "short_term"
    CONSOLIDATING = "consolidating"
    LONG_TERM = "long_term"
    ARCHIVED = "archived"


@dataclass
class EmotionalContext:
    """Emotional state at time of memory encoding."""
    
    valence: float  # -1 to 1
    arousal: float  # -1 to 1
    quadrant: str   # excited, frustrated, calm, depleted

    @property
    def salience(self) -> float:
        """
        Compute emotional salience score.
        
        Higher arousal = more salient.
        Returns value between 0.5 and 1.0.
        """
        arousal_factor = (abs(self.arousal) + 1) / 2  # 0.5 to 1.0
        valence_factor = (abs(self.valence) + 1) / 2  # 0.5 to 1.0
        
        # Arousal matters more for salience
        return (arousal_factor * 0.7) + (valence_factor * 0.3)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "quadrant": self.quadrant,
            "salience": self.salience,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionalContext":
        """Create from dictionary."""
        return cls(
            valence=data["valence"],
            arousal=data["arousal"],
            quadrant=data["quadrant"],
        )


@dataclass
class Memory:
    """A single memory unit."""

    # Core content
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    summary: str = ""

    # Classification
    memory_type: MemoryType = MemoryType.EPISODIC
    status: MemoryStatus = MemoryStatus.SHORT_TERM

    # Temporal info
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # Emotional context
    emotional_context: Optional[EmotionalContext] = None

    # Relationships
    related_memory_ids: List[str] = field(default_factory=list)
    source_memory_ids: List[str] = field(default_factory=list)

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Importance score
    importance_score: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "memory_type": self.memory_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "emotional_context": self.emotional_context.to_dict() if self.emotional_context else None,
            "related_memory_ids": self.related_memory_ids,
            "source_memory_ids": self.source_memory_ids,
            "tags": self.tags,
            "metadata": self.metadata,
            "importance_score": self.importance_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Reconstruct from dictionary."""
        emotional_ctx = None
        if data.get("emotional_context"):
            emotional_ctx = EmotionalContext.from_dict(data["emotional_context"])

        return cls(
            id=data["id"],
            content=data["content"],
            summary=data.get("summary", ""),
            memory_type=MemoryType(data["memory_type"]),
            status=MemoryStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
            emotional_context=emotional_ctx,
            related_memory_ids=data.get("related_memory_ids", []),
            source_memory_ids=data.get("source_memory_ids", []),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            importance_score=data.get("importance_score", 0.5),
        )

    def mark_accessed(self) -> None:
        """Update access timestamp and count."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def compute_importance(self) -> float:
        """
        Compute importance score based on multiple factors.
        
        Returns value between 0 and 1.
        """
        # Emotional salience
        salience = self.emotional_context.salience if self.emotional_context else 0.5
        
        # Recency (decays over time)
        age_days = (datetime.now() - self.created_at).days
        recency = max(0.0, 1.0 - (age_days / 365))  # Decay over a year
        
        # Access frequency
        access_factor = min(1.0, self.access_count / 10)  # Caps at 10 accesses
        
        # Weighted combination
        importance = (
            salience * 0.4 +
            recency * 0.3 +
            access_factor * 0.3
        )
        
        return importance
