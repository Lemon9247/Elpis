# Session 12: Reasoning Workflow Implementation Report

**Agent**: Reasoning Agent (Session 12 / D1)
**Date**: 2026-01-16
**Branch**: `phase3/memory-reasoning`
**Status**: Complete

## Summary

Successfully implemented the reasoning/thinking workflow with `<thinking>` tag parsing and ThoughtPanel integration. The system now supports an optional reasoning mode where the model is prompted to show its thinking process, which is then extracted and displayed in the ThoughtPanel.

## Changes Made

### 1. New File: `src/psyche/memory/reasoning.py`

Created a new module for parsing reasoning from model responses:

- **`ParsedResponse` dataclass**: Contains `thinking`, `response`, and `has_thinking` fields
- **`parse_reasoning()` function**: Extracts content from `<thinking>` tags and returns cleaned response
- **`extract_thinking_blocks()` helper**: Returns list of all thinking blocks found
- **`THINKING_PATTERN` regex**: Case-insensitive, multiline pattern for matching tags

Key features:
- Supports multiple `<thinking>` blocks (joined with newlines)
- Case-insensitive tag matching
- Non-greedy regex to handle multiple blocks correctly
- Preserves response formatting when removing thinking tags

### 2. Updated: `src/psyche/memory/server.py`

Added reasoning mode support:

- **`REASONING_PROMPT` constant**: Instructions for the model to use `<thinking>` tags
- **`_reasoning_enabled` instance variable**: Tracks current mode state
- **`set_reasoning_mode()` method**: Enables/disables reasoning and updates system prompt
- **`reasoning_enabled` property**: Read-only access to current state
- **Updated `_build_system_prompt()`**: Includes reasoning prompt when enabled
- **Updated response handling**: Parses `<thinking>` tags and routes to ThoughtPanel

System prompt update mechanism:
- When reasoning mode changes, the system prompt is rebuilt
- The compactor's system message is updated by directly replacing the first system message
- Token count difference is tracked to maintain accurate context size

### 3. Updated: `src/psyche/client/widgets/thought_panel.py`

Added new thought type:
- Added `"reasoning": "green"` to `type_colors` dictionary
- Reasoning thoughts now display in green color in the panel

### 4. Updated: `src/psyche/client/commands.py`

Added `/thinking` command:
- Name: `thinking`
- Aliases: `r`, `reason`
- Description: "Toggle reasoning display (on/off)"
- Shortcut: `Ctrl+R`

### 5. Updated: `src/psyche/client/app.py`

Added keybinding and handlers:
- Added `Binding("ctrl+r", "toggle_reasoning", "Reasoning", show=False)` to BINDINGS
- Added command handler for `/thinking` supporting:
  - `/thinking on` or `/thinking true` - enable
  - `/thinking off` or `/thinking false` - disable
  - `/thinking` - toggle current state
- Added `action_toggle_reasoning()` method for keybinding

### 6. New Test File: `tests/psyche/unit/test_reasoning.py`

Created comprehensive unit tests (22 tests total):

- **TestParsedResponse** (2 tests): Dataclass creation
- **TestParseReasoning** (13 tests):
  - Basic parsing with thinking tags
  - No thinking tags
  - Empty string
  - Case insensitivity
  - Multiple thinking blocks
  - Position variations (start, end)
  - Multiline content
  - Special characters
  - Empty/whitespace-only tags
  - Formatting preservation
- **TestExtractThinkingBlocks** (4 tests): Helper function
- **TestThinkingPattern** (3 tests): Regex pattern behavior

## Testing

All tests pass:

```
tests/psyche/unit/test_reasoning.py - 22 tests PASSED
tests/psyche/unit/test_server_parsing.py - 8 tests PASSED
```

All modified files compile without syntax errors.

## Usage

### Enable Reasoning Mode

```
/thinking on
# or
/r on
# or
Ctrl+R (toggle)
```

### How It Works

1. When reasoning mode is enabled, the system prompt includes instructions to use `<thinking>` tags
2. The model's response is parsed after generation completes
3. Content inside `<thinking>` tags is extracted and sent to ThoughtPanel
4. The cleaned response (without thinking tags) is shown to the user
5. Reasoning appears in the ThoughtPanel with green color

### Example Model Response

Input (from model):
```
<thinking>
The user wants to fix a bug in the login function. I should:
1. First read the current implementation
2. Identify the issue
3. Propose a fix
</thinking>

I'll help you fix that bug. Let me start by reading the login function...
```

Output:
- ThoughtPanel shows (in green): "The user wants to fix a bug..."
- Chat shows: "I'll help you fix that bug. Let me start by reading the login function..."

## Files Modified

| File | Changes |
|------|---------|
| `src/psyche/memory/reasoning.py` | New - reasoning parser module |
| `src/psyche/memory/server.py` | Added REASONING_PROMPT, set_reasoning_mode(), response parsing |
| `src/psyche/client/widgets/thought_panel.py` | Added "reasoning" type color |
| `src/psyche/client/commands.py` | Added /thinking command |
| `src/psyche/client/app.py` | Added Ctrl+R binding and handler |
| `tests/psyche/unit/test_reasoning.py` | New - 22 unit tests |

## Coordination Notes

This implementation is independent of Sessions 10 (Summarization) and 11 (Importance Scoring). The only shared file is `server.py`, where I modified:
- Added `REASONING_PROMPT` constant (near line 48-70)
- Added `_reasoning_enabled` instance variable in `__init__`
- Added `set_reasoning_mode()` method and `reasoning_enabled` property
- Modified `_build_system_prompt()` to include reasoning section
- Added reasoning parsing in response handling (lines ~806-820)

Note: Session 11 (Importance Agent) also added `auto_storage` settings to `ServerConfig` and `_after_response()` method. These changes are compatible with reasoning workflow.

## Remaining for Integration Testing (Session 13)

- Verify reasoning mode toggle persists during session
- Test interaction between reasoning mode and memory auto-storage
- Confirm reasoning content is NOT stored as memory (only actual responses)
- Test edge cases: long reasoning blocks, interrupted generation
