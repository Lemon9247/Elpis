# Session 11: Importance Scoring Implementation Report

**Agent**: Importance Agent
**Session**: 11 (B2.4)
**Date**: 2026-01-16
**Branch**: `phase3/memory-reasoning`
**Status**: Complete

## Summary

Implemented heuristic importance scoring for automatic memory storage. This enables Psyche to automatically identify and store significant exchanges to long-term memory without explicit user requests.

## Tasks Completed

### 11.1 Importance Scoring Module

Created `/home/lemoneater/Projects/Elpis/src/psyche/memory/importance.py` with:

- **`ImportanceScore` dataclass**: Stores total score and breakdown by factor
- **`calculate_importance()` function**: Evaluates exchange importance based on:
  - **Response length** (0.15-0.35): Longer responses indicate more substantive content
  - **Code blocks** (0.25-0.35): Presence of code suggests solutions or implementations
  - **Tool execution** (0.2-0.3): Concrete actions were taken
  - **Error messages** (0.2): Learning from mistakes is valuable
  - **Explicit requests** (0.4): User phrases like "remember", "important", "don't forget"
  - **Emotional intensity** (0.15-0.25): High valence/arousal moments are significant
- **`is_worth_storing()` function**: Threshold check helper
- **`format_score_breakdown()` function**: Debug/logging helper

### 11.2 Configuration Settings

Added to `ServerConfig` in `/home/lemoneater/Projects/Elpis/src/psyche/memory/server.py`:

```python
# Auto-storage settings (importance-based automatic memory storage)
auto_storage: bool = True  # Enable automatic storage of important exchanges
auto_storage_threshold: float = 0.6  # Min importance score to trigger auto-storage
```

### 11.3 Response Flow Integration

Added `_after_response()` method to `MemoryServer` that:
1. Checks if auto-storage is enabled
2. Verifies Mnemosyne connection is available
3. Gets current emotional state from Elpis
4. Calculates importance score for the exchange
5. Stores exchanges above threshold to Mnemosyne with:
   - Truncated content (user: 300 chars, response: 800 chars)
   - Generated summary (150 chars)
   - Memory type: "episodic"
   - Tags: ["auto-stored", "important"]
   - Emotional context

Also modified:
- `_execute_parsed_tool_call()`: Now returns tool result for collection
- `_process_user_input()`: Collects tool results and calls `_after_response()` after final response

### 11.4 Unit Tests

Created `/home/lemoneater/Projects/Elpis/tests/psyche/unit/test_importance.py` with 32 tests covering:

- **TestCalculateImportance** (16 tests):
  - Empty/short responses score low
  - Long responses get length bonus
  - Code blocks detected and scored
  - Tool execution scoring
  - Error detection in tool results
  - Explicit memory phrases (case-insensitive)
  - Emotional intensity thresholds
  - Score capping at 1.0
  - Edge cases (None emotion, empty tool results)

- **TestIsWorthStoring** (4 tests):
  - Threshold comparisons
  - Custom threshold support

- **TestFormatScoreBreakdown** (3 tests):
  - Empty score formatting
  - Non-zero factor display

- **TestImportanceScoreDataclass** (2 tests):
  - Creation and equality

- **TestRealWorldScenarios** (7 tests):
  - Simple greetings (low importance)
  - Q&A without code (medium)
  - Code solutions (high)
  - File modifications with tools
  - Debugging sessions with errors
  - Explicit "remember" requests
  - Emotional breakthrough moments

## Files Modified

| File | Change |
|------|--------|
| `src/psyche/memory/importance.py` | **NEW** - Importance scoring module |
| `src/psyche/memory/server.py` | Added auto_storage config, `_after_response()` method |
| `tests/psyche/unit/test_importance.py` | **NEW** - 32 unit tests |

## Testing Results

```
tests/psyche/unit/test_importance.py: 32 passed
tests/psyche/unit/test_server_parsing.py: 8 passed
```

All tests pass. The importance scoring module has 100% coverage.

## Design Decisions

1. **Score Weights**: Chose weights that make explicit requests (0.4) and code solutions (0.35) rank highest, while simple length (0.15-0.35) provides a baseline.

2. **Threshold Default**: Set at 0.6 to require at least 2-3 factors before auto-storing, preventing noise from simple exchanges.

3. **Content Truncation**: Truncated stored content to reasonable sizes (user: 300, response: 800 chars) to avoid bloating memory storage while preserving context.

4. **Graceful Degradation**: Method handles missing Mnemosyne connection, missing Elpis emotional state, and storage failures gracefully without disrupting the main response flow.

5. **Tool Result Collection**: Modified the existing tool execution flow to return and collect results for importance scoring, maintaining backward compatibility.

## Integration Notes

- The `_after_response()` method is called at the end of `_process_user_input()` after the final response
- Auto-storage runs asynchronously but doesn't block the user experience
- Storage failures are logged as warnings, not errors
- The feature can be disabled via `ServerConfig.auto_storage = False`

## Coordination

No conflicts with other agents:
- Session 10 (Summarization): Works on summary prompt, no overlap
- Session 12 (Reasoning): Works on thinking tags and `/thinking` command, no overlap

This completes the B2.4 task for heuristic importance scoring.
