"""Parser for extracting reasoning from model responses."""

import re
from dataclasses import dataclass


@dataclass
class ParsedResponse:
    """Response with extracted reasoning.

    Attributes:
        thinking: Content extracted from <reasoning> or <thinking> tags
        response: Content outside tags (the user-facing response)
        has_thinking: Whether any reasoning content was found
    """
    thinking: str
    response: str
    has_thinking: bool


# Pattern to match <reasoning>...</reasoning> blocks (case-insensitive, multiline)
# Also supports legacy <thinking> tags for backwards compatibility
REASONING_PATTERN = re.compile(
    r"<(?:reasoning|thinking)>(.*?)</(?:reasoning|thinking)>",
    re.DOTALL | re.IGNORECASE
)

# Legacy alias for backwards compatibility
THINKING_PATTERN = REASONING_PATTERN


def parse_reasoning(text: str) -> ParsedResponse:
    """
    Extract reasoning from model response.

    Parses text that may contain <reasoning>...</reasoning> tags (or legacy
    <thinking> tags), separating the internal reasoning from the user-facing response.

    Args:
        text: Raw model response that may contain <reasoning> tags

    Returns:
        ParsedResponse with separated reasoning and response content

    Examples:
        >>> result = parse_reasoning("<reasoning>Let me think...</reasoning>Hello!")
        >>> result.thinking
        'Let me think...'
        >>> result.response
        'Hello!'
        >>> result.has_thinking
        True

        >>> result = parse_reasoning("Just a plain response")
        >>> result.thinking
        ''
        >>> result.response
        'Just a plain response'
        >>> result.has_thinking
        False
    """
    # Find all thinking blocks
    matches = THINKING_PATTERN.findall(text)

    if matches:
        # Join all thinking content (in case there are multiple blocks)
        thinking = "\n\n".join(match.strip() for match in matches)
        # Remove all thinking blocks from response
        response = THINKING_PATTERN.sub("", text).strip()
        return ParsedResponse(
            thinking=thinking,
            response=response,
            has_thinking=True,
        )

    return ParsedResponse(
        thinking="",
        response=text,
        has_thinking=False,
    )


def extract_thinking_blocks(text: str) -> list[str]:
    """
    Extract all thinking blocks from text.

    Args:
        text: Text potentially containing thinking blocks

    Returns:
        List of thinking content strings (empty list if none found)
    """
    return [match.strip() for match in THINKING_PATTERN.findall(text)]
