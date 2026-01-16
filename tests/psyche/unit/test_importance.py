"""Tests for importance scoring module."""

import pytest

from psyche.memory.importance import (
    ImportanceScore,
    calculate_importance,
    is_worth_storing,
    format_score_breakdown,
    EXPLICIT_MEMORY_PHRASES,
)


class TestCalculateImportance:
    """Test calculate_importance function."""

    def test_empty_response_low_score(self):
        """Empty or very short responses should score low."""
        score = calculate_importance(
            message="Hello",
            response="Hi!",
        )
        assert score.total < 0.3
        assert score.length_score == 0.0

    def test_short_response_low_score(self):
        """Short responses without other factors should score low."""
        score = calculate_importance(
            message="What's the weather?",
            response="It's sunny today.",
        )
        assert score.total < 0.3
        assert score.length_score == 0.0
        assert score.code_score == 0.0
        assert score.tool_score == 0.0

    def test_long_response_medium_score(self):
        """Longer responses should get length score."""
        # 200+ chars
        score = calculate_importance(
            message="Explain something",
            response="A" * 250,
        )
        assert score.length_score == 0.15

        # 500+ chars
        score = calculate_importance(
            message="Explain something",
            response="A" * 600,
        )
        assert score.length_score == 0.25

        # 1000+ chars
        score = calculate_importance(
            message="Explain something in detail",
            response="A" * 1200,
        )
        assert score.length_score == 0.35

    def test_code_blocks_high_score(self):
        """Responses with code blocks should score higher."""
        # Single code block
        score = calculate_importance(
            message="Show me how to do X",
            response="Here's the code:\n```python\ndef foo(): pass\n```",
        )
        assert score.code_score == 0.25

        # Multiple code blocks
        score = calculate_importance(
            message="Show me examples",
            response="First example:\n```python\nx = 1\n```\nSecond:\n```python\ny = 2\n```",
        )
        assert score.code_score == 0.35

    def test_tool_execution_scores(self):
        """Tool execution should increase score."""
        # Single tool
        score = calculate_importance(
            message="Read the file",
            response="I read the file.",
            tool_results=[{"success": True, "content": "file contents"}],
        )
        assert score.tool_score == 0.2

        # Multiple tools
        score = calculate_importance(
            message="Read multiple files",
            response="I read the files.",
            tool_results=[
                {"success": True, "content": "file1"},
                {"success": True, "content": "file2"},
                {"success": True, "content": "file3"},
            ],
        )
        assert score.tool_score == 0.3

    def test_error_in_tool_results_scores(self):
        """Errors in tool results should add error score."""
        score = calculate_importance(
            message="Try something",
            response="There was an error.",
            tool_results=[{"success": False, "error": "File not found"}],
        )
        assert score.tool_score == 0.2
        assert score.error_score == 0.2

        # "Failed" indicator
        score = calculate_importance(
            message="Try something",
            response="The operation failed.",
            tool_results=[{"status": "failed", "message": "Connection timeout"}],
        )
        assert score.error_score == 0.2

    def test_explicit_remember_request(self):
        """Explicit memory requests should score very high."""
        for phrase in EXPLICIT_MEMORY_PHRASES:
            score = calculate_importance(
                message=f"Please {phrase} this information",
                response="I'll remember that.",
            )
            assert score.explicit_score == 0.4, f"Failed for phrase: {phrase}"

    def test_remember_case_insensitive(self):
        """Explicit phrases should be case-insensitive."""
        score = calculate_importance(
            message="REMEMBER THIS!",
            response="OK",
        )
        assert score.explicit_score == 0.4

        score = calculate_importance(
            message="Don't Forget this detail",
            response="Got it",
        )
        assert score.explicit_score == 0.4

    def test_emotional_intensity_low(self):
        """Low emotional intensity should not add score."""
        score = calculate_importance(
            message="Hello",
            response="Hi",
            emotion={"valence": 0.2, "arousal": 0.1},
        )
        assert score.emotion_score == 0.0

    def test_emotional_intensity_medium(self):
        """Medium emotional intensity adds score."""
        score = calculate_importance(
            message="This is exciting!",
            response="I agree!",
            emotion={"valence": 0.6, "arousal": 0.4},
        )
        assert score.emotion_score == 0.15

    def test_emotional_intensity_high(self):
        """High emotional intensity adds higher score."""
        score = calculate_importance(
            message="This is amazing!",
            response="Absolutely!",
            emotion={"valence": 0.8, "arousal": 0.9},
        )
        assert score.emotion_score == 0.25

    def test_negative_emotion_counts(self):
        """Negative valence with high magnitude should also count."""
        score = calculate_importance(
            message="This is frustrating",
            response="I understand",
            emotion={"valence": -0.7, "arousal": 0.6},
        )
        # abs(-0.7) > 0.5 and arousal 0.6 > 0.5, both hit medium threshold
        assert score.emotion_score == 0.15

    def test_score_capped_at_one(self):
        """Total score should never exceed 1.0."""
        # Create conditions that would sum to more than 1.0
        score = calculate_importance(
            message="Please remember this important code",  # explicit: 0.4
            response="Here's the code:\n```python\ndef foo(): pass\n```" + "A" * 1000,  # code: 0.25, length: 0.35
            tool_results=[
                {"success": True},
                {"success": False, "error": "test"},  # tool: 0.2, error: 0.2
            ],
            emotion={"valence": 0.9, "arousal": 0.9},  # emotion: 0.25
        )
        # All components would sum to: 0.4 + 0.35 + 0.35 + 0.2 + 0.2 + 0.25 = 1.75
        # But total should be capped at 1.0
        assert score.total == 1.0

    def test_high_importance_exchange_with_code_and_tools(self):
        """Exchange with code blocks and tool use should score high."""
        score = calculate_importance(
            message="Create a function to parse JSON",
            response="""Here's a function to parse JSON:

```python
import json

def parse_json(data: str) -> dict:
    return json.loads(data)
```

I've tested it and it works correctly.""",
            tool_results=[{"success": True, "output": "Test passed"}],
        )
        # code_score: 0.25, tool_score: 0.2 (response ~150 chars, no length score)
        assert score.total >= 0.4
        assert score.code_score == 0.25
        assert score.tool_score == 0.2

    def test_none_emotion_handled(self):
        """None emotion should be handled gracefully."""
        score = calculate_importance(
            message="Hello",
            response="Hi",
            emotion=None,
        )
        assert score.emotion_score == 0.0

    def test_empty_tool_results_handled(self):
        """Empty tool_results list should be handled."""
        score = calculate_importance(
            message="Hello",
            response="Hi",
            tool_results=[],
        )
        assert score.tool_score == 0.0


class TestIsWorthStoring:
    """Test is_worth_storing function."""

    def test_above_threshold_is_stored(self):
        """Scores above threshold should be worth storing."""
        score = ImportanceScore(
            total=0.7,
            length_score=0.3,
            code_score=0.25,
            tool_score=0.15,
            error_score=0.0,
            explicit_score=0.0,
            emotion_score=0.0,
        )
        assert is_worth_storing(score, threshold=0.6) is True

    def test_below_threshold_not_stored(self):
        """Scores below threshold should not be stored."""
        score = ImportanceScore(
            total=0.4,
            length_score=0.15,
            code_score=0.25,
            tool_score=0.0,
            error_score=0.0,
            explicit_score=0.0,
            emotion_score=0.0,
        )
        assert is_worth_storing(score, threshold=0.6) is False

    def test_at_threshold_is_stored(self):
        """Scores exactly at threshold should be stored."""
        score = ImportanceScore(
            total=0.6,
            length_score=0.3,
            code_score=0.3,
            tool_score=0.0,
            error_score=0.0,
            explicit_score=0.0,
            emotion_score=0.0,
        )
        assert is_worth_storing(score, threshold=0.6) is True

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        score = ImportanceScore(
            total=0.5,
            length_score=0.25,
            code_score=0.25,
            tool_score=0.0,
            error_score=0.0,
            explicit_score=0.0,
            emotion_score=0.0,
        )
        assert is_worth_storing(score, threshold=0.4) is True
        assert is_worth_storing(score, threshold=0.6) is False


class TestFormatScoreBreakdown:
    """Test format_score_breakdown function."""

    def test_format_empty_score(self):
        """Formatting score with no factors shows 'none'."""
        score = ImportanceScore(
            total=0.0,
            length_score=0.0,
            code_score=0.0,
            tool_score=0.0,
            error_score=0.0,
            explicit_score=0.0,
            emotion_score=0.0,
        )
        result = format_score_breakdown(score)
        assert "total=0.00" in result
        assert "(none)" in result

    def test_format_with_factors(self):
        """Formatting score shows only non-zero factors."""
        score = ImportanceScore(
            total=0.65,
            length_score=0.25,
            code_score=0.25,
            tool_score=0.15,
            error_score=0.0,
            explicit_score=0.0,
            emotion_score=0.0,
        )
        result = format_score_breakdown(score)
        assert "total=0.65" in result
        assert "length=0.25" in result
        assert "code=0.25" in result
        assert "tools=0.15" in result
        assert "errors" not in result  # 0.0 should not appear
        assert "explicit" not in result
        assert "emotion" not in result

    def test_format_all_factors(self):
        """Formatting score with all factors."""
        score = ImportanceScore(
            total=1.0,
            length_score=0.15,
            code_score=0.25,
            tool_score=0.2,
            error_score=0.1,
            explicit_score=0.2,
            emotion_score=0.1,
        )
        result = format_score_breakdown(score)
        assert "length=0.15" in result
        assert "code=0.25" in result
        assert "tools=0.20" in result
        assert "errors=0.10" in result
        assert "explicit=0.20" in result
        assert "emotion=0.10" in result


class TestImportanceScoreDataclass:
    """Test ImportanceScore dataclass."""

    def test_dataclass_creation(self):
        """ImportanceScore can be created with all fields."""
        score = ImportanceScore(
            total=0.75,
            length_score=0.25,
            code_score=0.25,
            tool_score=0.15,
            error_score=0.1,
            explicit_score=0.0,
            emotion_score=0.0,
        )
        assert score.total == 0.75
        assert score.length_score == 0.25
        assert score.code_score == 0.25
        assert score.tool_score == 0.15
        assert score.error_score == 0.1
        assert score.explicit_score == 0.0
        assert score.emotion_score == 0.0

    def test_dataclass_equality(self):
        """ImportanceScore instances with same values are equal."""
        score1 = ImportanceScore(
            total=0.5,
            length_score=0.25,
            code_score=0.25,
            tool_score=0.0,
            error_score=0.0,
            explicit_score=0.0,
            emotion_score=0.0,
        )
        score2 = ImportanceScore(
            total=0.5,
            length_score=0.25,
            code_score=0.25,
            tool_score=0.0,
            error_score=0.0,
            explicit_score=0.0,
            emotion_score=0.0,
        )
        assert score1 == score2


class TestRealWorldScenarios:
    """Test realistic exchange scenarios."""

    def test_simple_greeting_low_importance(self):
        """Simple greetings should have low importance."""
        score = calculate_importance(
            message="Hi there!",
            response="Hello! How can I help you today?",
        )
        assert score.total < 0.3
        assert not is_worth_storing(score)

    def test_question_answer_medium_importance(self):
        """Simple Q&A without code should be medium importance."""
        score = calculate_importance(
            message="What is a decorator in Python?",
            response="A decorator in Python is a function that modifies the behavior of another function. "
                     "It allows you to wrap another function to extend its behavior without permanently "
                     "modifying it. Decorators are commonly used for logging, authentication, and caching.",
        )
        # Length > 200 chars = 0.15
        assert score.total < 0.6
        assert not is_worth_storing(score)

    def test_code_solution_high_importance(self):
        """Code solutions should have high importance."""
        score = calculate_importance(
            message="Write a function to calculate factorial",
            response="""Here's a factorial function:

```python
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

This uses recursion to calculate the factorial. For n=5, it returns 120.""",
        )
        # code_score: 0.25, length > 200: 0.15
        assert score.total >= 0.4
        assert score.code_score == 0.25

    def test_file_modification_high_importance(self):
        """File modifications with tools should have high importance."""
        score = calculate_importance(
            message="Update the config file to add a new setting",
            response="I've updated the config file with the new setting.",
            tool_results=[
                {"success": True, "tool": "read_file", "content": "old content"},
                {"success": True, "tool": "write_file", "message": "File written"},
            ],
        )
        # tool_score for 2 tools: 0.2
        assert score.tool_score == 0.2
        assert score.total >= 0.2

    def test_debugging_session_high_importance(self):
        """Debugging with errors should have high importance."""
        score = calculate_importance(
            message="Why is my test failing?",
            response="""The test is failing because of a type error. Here's the fix:

```python
def process(data: str) -> str:
    return data.strip()
```

The issue was that `data` was being passed as `None` in some cases.""",
            tool_results=[
                {"success": False, "error": "AssertionError: expected 'hello' but got None"},
            ],
        )
        # code_score: 0.25, tool_score: 0.2, error_score: 0.2, length > 200: 0.15
        assert score.total >= 0.6
        assert is_worth_storing(score)

    def test_explicit_remember_triggers_storage(self):
        """Explicit 'remember' request should always trigger storage."""
        score = calculate_importance(
            message="Remember: the API key is stored in .env file",
            response="I'll remember that the API key is in the .env file.",
        )
        # explicit_score: 0.4
        assert score.explicit_score == 0.4
        assert score.total >= 0.4

    def test_emotional_breakthrough_moment(self):
        """High emotion moments should be preserved."""
        score = calculate_importance(
            message="Finally got it working!",
            response="That's great news! Congratulations on fixing the issue!",
            emotion={"valence": 0.9, "arousal": 0.8},
        )
        # emotion_score: 0.25
        assert score.emotion_score == 0.25
        assert score.total >= 0.25
