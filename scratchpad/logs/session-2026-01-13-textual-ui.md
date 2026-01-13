# Session Log: Psyche UI Overhaul with Textual

**Date:** 2026-01-13
**Branch:** `psyche-ui-improvements`
**Focus:** Full-stack streaming + Textual TUI

## Summary

Implemented a complete overhaul of the Psyche user interface:
1. Full-stack streaming from LLM through MCP to UI
2. Modern Textual TUI with split-pane layout
3. Real-time token display, tool activity, emotional state

## Changes Made

### Phase 1: Streaming Infrastructure

#### 1.1 LlamaInference Streaming
**File:** `src/elpis/llm/inference.py`
- Added `chat_completion_stream()` async method
- Uses `stream=True` in llama-cpp-python
- Bridges sync iterator to async via queue in thread

#### 1.2 MCP Server Streaming Tools
**File:** `src/elpis/server.py`
- Added `StreamState` dataclass for tracking active streams
- New tools:
  - `generate_stream_start` - Starts background generation, returns stream_id
  - `generate_stream_read` - Polls for new tokens
  - `generate_stream_cancel` - Cancels active stream
- Test updated to expect 8 tools (was 5)

#### 1.3 ElpisClient Streaming
**File:** `src/psyche/mcp/client.py`
- Added `generate_stream()` method
- Polls server with configurable interval (default 50ms)
- Yields tokens as async iterator

#### 1.4 MemoryServer Callbacks
**File:** `src/psyche/memory/server.py`
- Added `on_token` callback for streaming tokens
- Modified `_process_user_input` to use streaming
- Modified `on_tool_call` to fire at start (None) and end (result)
- Added `_update_emotion_for_interaction_text()` helper

### Phase 2: Textual TUI

#### 2.1 New Widgets
**Directory:** `src/psyche/client/widgets/`
- `chat_view.py` - ChatView with streaming support
- `sidebar.py` - EmotionalStateDisplay + StatusDisplay
- `tool_activity.py` - ToolActivity with running/complete/error states
- `user_input.py` - UserInput with command detection
- `thought_panel.py` - Collapsible ThoughtPanel

#### 2.2 Main Application
**File:** `src/psyche/client/app.py`
- PsycheApp with:
  - Header, Footer, Sidebar, ChatView, ThoughtPanel, UserInput
  - All callbacks wired (on_token, on_thought, on_response, on_tool_call)
  - Periodic emotional state updates
  - Command handling (/help, /status, /clear, /emotion, /thoughts, /quit)
  - Keyboard shortcuts (Ctrl+C, Ctrl+L, Ctrl+T)

#### 2.3 Stylesheet
**File:** `src/psyche/client/app.tcss`
- Split-pane layout (sidebar 25 chars, content fills rest)
- Colored borders, streaming visual feedback
- Tool status colors (running=yellow, complete=green, error=red)

#### 2.4 CLI Update
**File:** `src/psyche/cli.py`
- Replaced PsycheREPL with PsycheApp
- Simplified arguments (removed show_thoughts, show_emotion - now in-app commands)

### Dependencies
**File:** `pyproject.toml`
- Added `textual>=0.47.0`
- Added textual to mypy ignore list

## Test Results

All 267 tests pass after changes.

## Layout Preview

```
+-------------------------------------------------------------------+
|  Header: Psyche - Continuous Inference Agent                      |
+-------------------+-----------------------------------------------+
|  Sidebar          |  ChatView (scrollable)                        |
|  +--------------+ |  +------------------------------------------+ |
|  | Emotional    | |  | You: Hello                               | |
|  | State        | |  | Assistant: Hi there! [streaming...]      | |
|  | v=0.2 a=0.5  | |  +------------------------------------------+ |
|  +--------------+ |                                               |
|  | Status       | |  ThoughtPanel (collapsible)                  |
|  | idle, 5 msgs | |  +------------------------------------------+ |
|  +--------------+ |  | [reflection] Considering context...      | |
|  | Tool         | |  +------------------------------------------+ |
|  | Activity     | |  UserInput                                    |
|  | [read_file]  | |  +------------------------------------------+ |
|  +--------------+ |  | > Type your message...                   | |
+-------------------+-----------------------------------------------+
|  Footer: /help | /status | /clear | /quit                         |
+-------------------------------------------------------------------+
```

## Commits

1. `acb1534` - Add streaming support and Textual TUI for Psyche

## Branch Status

Pushed to `origin/psyche-ui-improvements`, ready for PR or further work.

## Next Steps

- Test the full flow with actual LLM
- Fine-tune CSS for different terminal sizes
- Consider adding syntax highlighting for code responses
- Add more keybindings/commands as needed
