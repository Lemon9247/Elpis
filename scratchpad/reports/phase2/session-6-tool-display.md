# Session 6: Tool Display Enhancement (C1)

**Date:** 2026-01-16
**Branch:** phase2/ux-improvements
**Session Type:** Feature Implementation

## Summary

Implemented enhanced tool activity display in the Psyche TUI. Previously, tool activity only showed the tool name with a status icon (e.g., `[OK] read_file`). Now the display shows human-readable descriptions with arguments and result summaries (e.g., `[OK] Reading src/main.py (150 lines)`).

## Changes Made

### 1. Created New Formatter Module

**File:** `src/psyche/client/formatters/tool_formatter.py`

Created `ToolDisplayFormatter` class with three class methods:

- `format_start(tool_name, args)` - Formats the tool invocation description
  - Uses templates for known tools: `read_file` -> "Reading {file_path}"
  - Falls back to title-cased tool name for unknown tools

- `format_result(tool_name, result)` - Formats a brief result summary
  - Tool-specific summaries: line counts, exit codes, match counts, etc.
  - Error handling with truncation for long error messages

- `format_full(tool_name, args, result, status)` - Combines start + result

**Supported tools:**
- `read_file` - "Reading path/to/file" + "(N lines)"
- `create_file` - "Creating path/to/file" + "(N lines written)"
- `edit_file` - "Editing path/to/file" + "(-X/+Y chars)"
- `execute_bash` - "$ command" + "(exit N)"
- `list_directory` - "Listing path" + "(N files, M dirs)"
- `search_codebase` - "Searching: pattern" + "(N matches)"
- `recall_memory` - "Recalling: query" + "(N memories)"
- `store_memory` - "Storing memory" + "(stored)"

### 2. Modified Server Callback Signature

**File:** `src/psyche/memory/server.py`

Updated `on_tool_call` callback to include arguments:
- Old: `Callable[[str, Optional[Dict[str, Any]]], None]` (name, result)
- New: `Callable[[str, Dict[str, Any], Optional[Dict[str, Any]]], None]` (name, args, result)

Changes made:
- Line 126: Updated type hint
- Line 139: Updated docstring
- Line 803-804: Pass `arguments` at tool start
- Line 819-820: Pass `arguments` at tool completion
- Line 1081-1082: Pass arguments in idle tool call notification

### 3. Updated Tool Activity Widget

**File:** `src/psyche/client/widgets/tool_activity.py`

Updated `ToolExecution` dataclass:
- Added `args: Dict[str, Any]` field (default empty dict)
- Changed `result_preview: str` to `result: Optional[Dict[str, Any]]`

Updated methods:
- `add_tool_start(name, args)` - Now accepts and stores args
- `update_tool_complete(name, result)` - Stores full result dict
- `_render_tools()` - Uses `ToolDisplayFormatter.format_full()`
- `render()` - Uses `ToolDisplayFormatter.format_full()`

### 4. Updated App Callback Handler

**File:** `src/psyche/client/app.py`

Updated `_on_tool_call` method signature to accept `args` parameter and pass it to the widget.

## Testing

All 205 existing tests pass. Manual verification confirmed:

```
Testing format_start:
  read_file: Reading src/main.py
  execute_bash: $ ls -la
  list_directory: Listing .
  search_codebase: Searching: def foo
  unknown_tool: Unknown Tool

Testing format_result:
  read_file success: (150 lines)
  read_file truncated: (150+ lines)
  execute_bash: (exit 0)
  list_directory: (5 files, 2 dirs)
  error: (File not found: test.py)

Testing format_full:
  running: Reading src/main.py
  complete: Reading src/main.py (150 lines)
```

## Files Created/Modified

| File | Action |
|------|--------|
| `src/psyche/client/formatters/__init__.py` | Created |
| `src/psyche/client/formatters/tool_formatter.py` | Created |
| `src/psyche/memory/server.py` | Modified |
| `src/psyche/client/widgets/tool_activity.py` | Modified |
| `src/psyche/client/app.py` | Modified |

## Testing Checklist

- [x] `read_file` shows "Reading path/to/file.py"
- [x] Completion shows "(150 lines)" or similar summary
- [x] Errors show formatted error message (truncated if >30 chars)
- [x] Unknown tools fall back gracefully to title-cased name

## Notes

- The formatter handles long paths by showing the last 2 path components
- Argument values are truncated to 40 characters max
- Error messages are truncated to 30 characters to fit the display
- The truncation indicator for read_file results uses "+" suffix (e.g., "150+ lines") when truncated
