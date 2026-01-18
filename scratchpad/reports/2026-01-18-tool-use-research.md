# Tool Use Research: Current State vs Modern Patterns

**Date:** 2026-01-18
**Topic:** Investigation of Psyche tool use implementation

## Summary

Psyche uses **text-based tool parsing** (looking for ```tool_call blocks in LLM output) because Elpis uses local inference (llama.cpp/transformers) which doesn't support structured tool calling like OpenAI/Anthropic APIs.

Remote mode (`hermes --server`) has an **incomplete tool execution loop** - the server returns tool_calls but Hermes never executes them.

---

## Modern LLM Tool Use (Structured)

### OpenAI Format
```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather in a location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string", "description": "City and state"}
      },
      "required": ["location"]
    }
  }
}
```

### Anthropic Claude Format
```json
{
  "name": "get_weather",
  "description": "Get current weather in a location",
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "City and state"}
    },
    "required": ["location"]
  }
}
```

Key differences:
- Anthropic uses `input_schema` not `parameters`
- Both emphasize **extremely detailed descriptions** (3-4+ sentences)

### Response Formats

**OpenAI:**
```json
{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "I'll help you..."},
    {
      "type": "tool_calls",
      "tool_calls": [{
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"San Francisco\"}"
        }
      }]
    }
  ]
}
```

**Claude:**
```json
{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "I'll help you..."},
    {
      "type": "tool_use",
      "id": "toolu_01A09q90qw90lq917835lq9",
      "name": "get_weather",
      "input": {"location": "San Francisco, CA"}
    }
  ]
}
```

### Tool Results Format

Both require tool results in a single user message:
```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "call_123",
      "content": "San Francisco: 68°F, partly cloudy"
    }
  ]
}
```

Critical rules:
- Tool result blocks MUST come FIRST in content array
- Include `"is_error": true` if tool execution failed
- All results in one message (encourages parallel tool use)

---

## Current Psyche Implementation

### Tool Definition Flow
```
ToolEngine._register_tools()
├── Creates implementation instances (FileTools, BashTool, SearchTool)
└── Registers with ToolDefinition (name, description, parameters, handler)

Tools registered:
- read_file, create_file, edit_file (FileTools)
- execute_bash (BashTool)
- search_codebase (SearchTool)
- list_directory (FileTools)

Memory tools registered separately in hermes/cli.py:
- recall_memory, store_memory (MemoryTools)
```

### System Prompt Format
Tools are described as markdown in the system prompt:
```
### read_file
Read contents of a file from the workspace
Parameters:
  - file_path (string): Path to file (relative to workspace or absolute)
  - max_lines (integer): Maximum number of lines to read (default: 2000)
```

LLM is instructed to output:
```
\`\`\`tool_call
{"name": "tool_name", "arguments": {...}}
\`\`\`
```

### Parsing (ReactHandler.parse_tool_call)
- Regex: `r"```tool_call\s*\n?(.*?)\n?```"`
- Validates JSON has "name" key
- Fallback: Parse raw JSON if response starts with `{`

### Execution Loop (Local Mode)
```
1. Generate response with streaming
2. Parse for tool_call block
3. If found:
   - Execute via ToolEngine
   - Add result to context as user message
   - Loop back to step 1
4. If not found:
   - Return final response
```

---

## Issues Identified

### Issue 1: Remote Mode Tool Loop Incomplete
**Problem:** In remote mode, Hermes streams the response but never:
- Checks for tool_calls in the finish chunk
- Executes tools locally
- Sends results back to server

**Location:**
- `hermes/app.py:394-418` - `_process_via_client()` just streams tokens
- `psyche/handlers/psyche_client.py:515-561` - `generate_stream()` ignores finish chunk

### Issue 2: Dual Tool Definition Formats
**Problem:** Tools defined twice:
- Pydantic `ToolInput` models (for validation)
- JSON Schema dicts in `ToolDefinition.parameters` (for API/docs)

**Impact:** Changes require updates in both places.

### Issue 3: Text-Based Parsing is Fragile
**Problem:** Regex parsing of LLM output can fail if:
- Extra whitespace/formatting
- Model continues after tool_call block
- JSON syntax errors

**Location:** `ReactHandler.parse_tool_call()` lines 358-421

### Issue 4: Memory Tools Not in Base Registration
**Problem:** Memory tools only registered in hermes/cli.py, not in ToolEngine base.

**Impact:** Memory tools unavailable in HTTP server mode.

### Issue 5: Tool Descriptions are Markdown, Not Structured
**Problem:** Tool descriptions embedded in system prompt are human-readable, not JSON schema.

**Impact:** API schema and system prompt can diverge.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `psyche/tools/tool_engine.py` | Tool registration, execution, description generation |
| `psyche/tools/tool_definitions.py` | Pydantic input models for validation |
| `psyche/tools/implementations/` | Actual tool implementations |
| `psyche/handlers/react_handler.py` | ReAct loop with tool parsing |
| `psyche/core/server.py` | System prompt building |
| `psyche/server/http.py` | HTTP server tool_call parsing |
| `hermes/app.py` | TUI app with tool callbacks |
| `hermes/cli.py` | App setup, memory tool registration |

---

## Best Practices from Research

1. **Descriptions matter most** - Detailed tool descriptions improve accuracy more than schema
2. **Use structured tool calling** - When API supports it (not text parsing)
3. **Format results correctly** - All tool results in single user message
4. **Enable parallelism** - Modern models naturally make parallel tool calls
5. **Check stop_reason** - Handle `tool_use`, `end_turn`, `max_tokens`
6. **Error handling** - Return `is_error: true` for failed tools

---

## Recommendations

### Short-term: Fix Remote Mode
1. Update `generate_stream()` to capture tool_calls from finish chunk
2. Update `_process_via_client()` to implement tool execution loop
3. Pass ToolEngine to Hermes in remote mode

### Medium-term: Tool Organization
1. Move all tools to a shared location
2. Register memory tools in ToolEngine base (not just hermes/cli.py)
3. Consolidate dual definitions (Pydantic models + JSON schema)

### Long-term: Consider Structured Tool Calling
If Elpis adds support for grammar-constrained generation, could implement proper structured tool calling instead of text parsing.
