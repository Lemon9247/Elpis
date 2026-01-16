"""Unit tests for reasoning parser module."""

import pytest

from psyche.memory.reasoning import (
    ParsedResponse,
    parse_reasoning,
    extract_thinking_blocks,
    THINKING_PATTERN,
)


class TestParsedResponse:
    """Tests for ParsedResponse dataclass."""

    def test_parsed_response_creation(self):
        """Create ParsedResponse with all fields."""
        response = ParsedResponse(
            thinking="My reasoning here",
            response="The actual response",
            has_thinking=True,
        )
        assert response.thinking == "My reasoning here"
        assert response.response == "The actual response"
        assert response.has_thinking is True

    def test_parsed_response_no_thinking(self):
        """Create ParsedResponse without thinking."""
        response = ParsedResponse(
            thinking="",
            response="Just a response",
            has_thinking=False,
        )
        assert response.thinking == ""
        assert response.response == "Just a response"
        assert response.has_thinking is False


class TestParseReasoning:
    """Tests for parse_reasoning function."""

    def test_parse_with_thinking_tags(self):
        """Parse response with thinking tags."""
        text = """<thinking>
Let me analyze this problem.
1. First, I need to understand what's being asked
2. Then I can provide a solution
</thinking>

Here's my answer to your question."""

        result = parse_reasoning(text)

        assert result.has_thinking is True
        assert "Let me analyze this problem" in result.thinking
        assert "First, I need to understand" in result.thinking
        assert "Here's my answer" in result.response
        assert "<thinking>" not in result.response
        assert "</thinking>" not in result.response

    def test_parse_without_thinking_tags(self):
        """Parse response without any thinking tags."""
        text = "This is just a plain response with no thinking tags."

        result = parse_reasoning(text)

        assert result.has_thinking is False
        assert result.thinking == ""
        assert result.response == text

    def test_parse_empty_string(self):
        """Parse empty string."""
        result = parse_reasoning("")

        assert result.has_thinking is False
        assert result.thinking == ""
        assert result.response == ""

    def test_parse_case_insensitive(self):
        """Parse thinking tags with different cases."""
        text = """<THINKING>
This is my reasoning.
</THINKING>

Response here."""

        result = parse_reasoning(text)

        assert result.has_thinking is True
        assert "This is my reasoning" in result.thinking
        assert "Response here" in result.response

    def test_parse_mixed_case(self):
        """Parse thinking tags with mixed case."""
        text = """<Thinking>
Some thoughts here.
</thinking>

The answer is 42."""

        result = parse_reasoning(text)

        assert result.has_thinking is True
        assert "Some thoughts here" in result.thinking

    def test_parse_multiple_thinking_blocks(self):
        """Parse response with multiple thinking blocks."""
        text = """<thinking>
First thought process.
</thinking>

Some intermediate text.

<thinking>
Second thought process.
</thinking>

Final response."""

        result = parse_reasoning(text)

        assert result.has_thinking is True
        # Both thinking blocks should be captured
        assert "First thought process" in result.thinking
        assert "Second thought process" in result.thinking
        # The response should have neither thinking block
        assert "<thinking>" not in result.response
        assert "intermediate text" in result.response
        assert "Final response" in result.response

    def test_parse_thinking_at_start(self):
        """Parse with thinking block at the very start."""
        text = "<thinking>Quick thought</thinking>Hello!"

        result = parse_reasoning(text)

        assert result.has_thinking is True
        assert result.thinking == "Quick thought"
        assert result.response == "Hello!"

    def test_parse_thinking_at_end(self):
        """Parse with thinking block at the very end."""
        text = "Here's my answer.\n<thinking>Afterthought</thinking>"

        result = parse_reasoning(text)

        assert result.has_thinking is True
        assert result.thinking == "Afterthought"
        assert result.response.strip() == "Here's my answer."

    def test_parse_multiline_thinking(self):
        """Parse thinking with multiple lines."""
        text = """<thinking>
Line 1
Line 2
Line 3
</thinking>

Response."""

        result = parse_reasoning(text)

        assert result.has_thinking is True
        assert "Line 1" in result.thinking
        assert "Line 2" in result.thinking
        assert "Line 3" in result.thinking

    def test_parse_thinking_with_special_chars(self):
        """Parse thinking containing special characters."""
        text = """<thinking>
What if x > 5 && y < 10?
Let's check: `code_block()`
</thinking>

The result is computed."""

        result = parse_reasoning(text)

        assert result.has_thinking is True
        assert "x > 5 && y < 10" in result.thinking
        assert "`code_block()`" in result.thinking

    def test_parse_empty_thinking_tags(self):
        """Parse empty thinking tags."""
        text = "<thinking></thinking>Some response."

        result = parse_reasoning(text)

        assert result.has_thinking is True
        assert result.thinking == ""
        assert result.response == "Some response."

    def test_parse_whitespace_only_thinking(self):
        """Parse thinking tags with only whitespace."""
        text = "<thinking>   \n\t  </thinking>Response."

        result = parse_reasoning(text)

        assert result.has_thinking is True
        assert result.thinking == ""  # stripped
        assert result.response == "Response."

    def test_parse_preserves_response_formatting(self):
        """Ensure response formatting is preserved."""
        text = """<thinking>Thought</thinking>

# Header

- List item 1
- List item 2

```python
code_block()
```"""

        result = parse_reasoning(text)

        assert result.has_thinking is True
        assert "# Header" in result.response
        assert "- List item 1" in result.response
        assert "```python" in result.response


class TestExtractThinkingBlocks:
    """Tests for extract_thinking_blocks function."""

    def test_extract_single_block(self):
        """Extract single thinking block."""
        text = "<thinking>Just one thought</thinking>Response"

        blocks = extract_thinking_blocks(text)

        assert len(blocks) == 1
        assert blocks[0] == "Just one thought"

    def test_extract_multiple_blocks(self):
        """Extract multiple thinking blocks."""
        text = """<thinking>First</thinking>
Middle text
<thinking>Second</thinking>
End text"""

        blocks = extract_thinking_blocks(text)

        assert len(blocks) == 2
        assert blocks[0] == "First"
        assert blocks[1] == "Second"

    def test_extract_no_blocks(self):
        """Return empty list when no thinking blocks found."""
        text = "No thinking tags here."

        blocks = extract_thinking_blocks(text)

        assert len(blocks) == 0
        assert blocks == []

    def test_extract_empty_string(self):
        """Return empty list for empty input."""
        blocks = extract_thinking_blocks("")

        assert len(blocks) == 0


class TestThinkingPattern:
    """Tests for the THINKING_PATTERN regex."""

    def test_pattern_matches_basic(self):
        """Pattern matches basic thinking tags."""
        text = "<thinking>content</thinking>"
        match = THINKING_PATTERN.search(text)

        assert match is not None
        assert match.group(1) == "content"

    def test_pattern_matches_multiline(self):
        """Pattern matches multiline content."""
        text = "<thinking>\nline1\nline2\n</thinking>"
        match = THINKING_PATTERN.search(text)

        assert match is not None
        assert "line1" in match.group(1)
        assert "line2" in match.group(1)

    def test_pattern_is_non_greedy(self):
        """Pattern is non-greedy for multiple blocks."""
        text = "<thinking>first</thinking>middle<thinking>second</thinking>"
        matches = THINKING_PATTERN.findall(text)

        assert len(matches) == 2
        assert matches[0] == "first"
        assert matches[1] == "second"
