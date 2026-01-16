# Text-Based Tool Call Parsing Implementation

**Date:** 2026-01-13
**Commit:** 644d27c

## Problem

When using Psyche with the Elpis inference server, the model could see the available tools in the system prompt but failed to actually use them. The root cause was that llama-cpp-python's native function calling API was not working reliably with the local model.

## Solution

Switched from native function calling to text-based tool parsing:

1. **Updated system prompt** in `MemoryServer._build_system_prompt()` with:
   - Explicit tool call format using ````tool_call` code blocks
   - Concrete examples for common tools (list_directory, execute_bash, read_file)
   - Clear instruction to respond with ONLY the tool_call block when using tools

2. **Added `_parse_tool_call()` method** that:
   - Searches for ````tool_call` code blocks in LLM responses
   - Falls back to checking for raw JSON at start of response
   - Handles edge cases (missing arguments, extra newlines, case insensitive)

3. **Added `_execute_parsed_tool_call()` method** that:
   - Converts parsed dict to ToolEngine format
   - Executes the tool and adds result to context
   - Updates emotional state based on success/failure

4. **Cleaned up**:
   - Removed unused `_execute_tool_calls()` method
   - Updated ReAct loop in `_process_user_input()` to use new parsing

## Testing

- Added 8 unit tests in `psyche/tests/unit/test_server_parsing.py`
- All 171 Psyche tests pass

## Files Changed

- `psyche/src/psyche/memory/server.py` - Core implementation
- `psyche/tests/unit/test_server_parsing.py` - New test file
