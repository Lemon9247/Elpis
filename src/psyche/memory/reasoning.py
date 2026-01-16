"""Parser for extracting reasoning from model responses."""

import re
from dataclasses import dataclass


@dataclass
class ParsedResponse:
    """Response with extracted reasoning.

    Attributes:
        thinking: Content extracted from <thinking> tags
        response: Content outside tags (the user-facing response)
        has_thinking: Whether any <thinking> content was found
    """
    thinking: str
    response: str
    has_thinking: bool


# Pattern to match <thinking>...</thinking> blocks (case-insensitive, multiline)
THINKING_PATTERN = re.compile(
    r"<thinking>(.*?)</thinking>",
    re.DOTALL | re.IGNORECASE
)


def parse_reasoning(text: str) -> ParsedResponse:
    """
    Extract reasoning from model response.

    Parses text that may contain <thinking>...</thinking> tags,
    separating the internal reasoning from the user-facing response.

    Args:
        text: Raw model response that may contain <thinking> tags

    Returns:
        ParsedResponse with separated thinking and response content

    Examples:
        >>> result = parse_reasoning("<thinking>Let me think...</thinking>Hello!")
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
