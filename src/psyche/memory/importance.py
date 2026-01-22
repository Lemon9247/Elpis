"""Heuristic importance scoring for automatic memory storage."""

from dataclasses import dataclass
from typing import Any

from psyche.config.constants import AUTO_STORAGE_THRESHOLD


@dataclass
class ImportanceScore:
    """Breakdown of importance score factors."""

    total: float
    length_score: float
    code_score: float
    tool_score: float
    error_score: float
    explicit_score: float
    emotion_score: float


# Phrases that indicate the user wants something remembered
EXPLICIT_MEMORY_PHRASES = [
    "remember",
    "important",
    "note that",
    "don't forget",
    "keep in mind",
    "make a note",
    "save this",
    "store this",
]


def calculate_importance(
    message: str,
    response: str,
    tool_results: list[dict[str, Any]] | None = None,
    emotion: dict[str, float] | None = None,
) -> ImportanceScore:
    """
    Calculate importance score (0.0 to 1.0) for an exchange.

    This heuristic scoring system evaluates how "important" a user-assistant
    exchange is, helping decide which exchanges should be auto-stored to
    long-term memory.

    Factors:
    - Response length (longer = more effort, likely more substantive)
    - Contains code blocks (likely a solution or implementation)
    - Tool execution occurred (concrete actions were taken)
    - Error messages present (learn from mistakes)
    - User said "remember this" (explicit request)
    - Emotional intensity (significant moments)

    Args:
        message: The user's input message
        response: The assistant's response
        tool_results: List of tool execution results (if any)
        emotion: Current emotional state dict with valence/arousal keys

    Returns:
        ImportanceScore with total score and breakdown by factor
    """
    scores = {
        "length_score": 0.0,
        "code_score": 0.0,
        "tool_score": 0.0,
        "error_score": 0.0,
        "explicit_score": 0.0,
        "emotion_score": 0.0,
    }

    # Length-based scoring (longer responses indicate more effort/substance)
    response_len = len(response)
    if response_len > 1000:
        scores["length_score"] = 0.35
    elif response_len > 500:
        scores["length_score"] = 0.25
    elif response_len > 200:
        scores["length_score"] = 0.15

    # Code blocks (likely a solution or implementation)
    # Count code blocks for weighted scoring
    code_block_count = response.count("```")
    if code_block_count >= 4:  # 2+ code blocks
        scores["code_score"] = 0.35
    elif code_block_count >= 2:  # 1 code block
        scores["code_score"] = 0.25

    # Tool execution (concrete actions were taken)
    if tool_results:
        # More tools = more important action
        tool_count = len(tool_results)
        if tool_count >= 3:
            scores["tool_score"] = 0.3
        elif tool_count >= 1:
            scores["tool_score"] = 0.2

        # Check for errors in tool results (learning from mistakes is important)
        error_indicators = ["error", "failed", "exception", "traceback"]
        for result in tool_results:
            result_str = str(result).lower()
            if any(indicator in result_str for indicator in error_indicators):
                scores["error_score"] = 0.2
                break

    # Explicit user request to remember
    message_lower = message.lower()
    if any(phrase in message_lower for phrase in EXPLICIT_MEMORY_PHRASES):
        scores["explicit_score"] = 0.4  # High weight for explicit requests

    # Emotional intensity from Elpis
    if emotion:
        valence = abs(emotion.get("valence", 0))
        arousal = abs(emotion.get("arousal", 0))

        # High emotional intensity suggests a significant moment
        if valence > 0.7 or arousal > 0.7:
            scores["emotion_score"] = 0.25
        elif valence > 0.5 or arousal > 0.5:
            scores["emotion_score"] = 0.15

    # Calculate total, capped at 1.0
    total = min(1.0, sum(scores.values()))

    return ImportanceScore(total=total, **scores)


def is_worth_storing(
    score: ImportanceScore,
    threshold: float = AUTO_STORAGE_THRESHOLD,
) -> bool:
    """
    Determine if an exchange should be auto-stored based on importance score.

    Args:
        score: The calculated ImportanceScore
        threshold: Minimum score required for storage (default from constants)

    Returns:
        True if the exchange should be stored, False otherwise
    """
    return score.total >= threshold


def format_score_breakdown(score: ImportanceScore) -> str:
    """
    Format importance score breakdown as a human-readable string.

    Useful for debugging and logging.

    Args:
        score: The ImportanceScore to format

    Returns:
        Formatted string showing score breakdown
    """
    parts = []

    if score.length_score > 0:
        parts.append(f"length={score.length_score:.2f}")
    if score.code_score > 0:
        parts.append(f"code={score.code_score:.2f}")
    if score.tool_score > 0:
        parts.append(f"tools={score.tool_score:.2f}")
    if score.error_score > 0:
        parts.append(f"errors={score.error_score:.2f}")
    if score.explicit_score > 0:
        parts.append(f"explicit={score.explicit_score:.2f}")
    if score.emotion_score > 0:
        parts.append(f"emotion={score.emotion_score:.2f}")

    breakdown = ", ".join(parts) if parts else "none"
    return f"total={score.total:.2f} ({breakdown})"
