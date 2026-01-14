# Week 3 Architecture: Agent & REPL

## Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         ElpisREPL                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  prompt_toolkit (async input)                         │  │
│  │  - Command history                                    │  │
│  │  - Special commands: /help, /clear, /exit            │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Rich (output formatting)                             │  │
│  │  - Markdown rendering                                 │  │
│  │  - Code syntax highlighting                           │  │
│  │  - Colored panels                                     │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │ user_input
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    AgentOrchestrator                         │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Message History                                       │ │
│  │  [{"role": "user", "content": "..."},                 │ │
│  │   {"role": "assistant", "content": "..."},            │ │
│  │   {"role": "tool", "content": "..."}]                 │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ReAct Loop (max 10 iterations)                       │ │
│  │                                                        │ │
│  │  1. REASON: Build prompt with history                 │ │
│  │       ↓                                                │ │
│  │  2. ACT: Call LLM (text or tool calls)               │ │
│  │       ↓                                                │ │
│  │  3. OBSERVE: Execute tools concurrently               │ │
│  │       ↓                                                │ │
│  │  4. REPEAT or RETURN final response                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                  │
└───────────────────────────┼──────────────────────────────────┘
                            │
                ┌───────────┴──────────┐
                │                      │
                ▼                      ▼
      ┌──────────────────┐   ┌──────────────────┐
      │  LlamaInference  │   │   ToolEngine     │
      │   (Week 1)       │   │    (Week 2)      │
      │                  │   │                  │
      │ - chat_completion│   │ - get_tool_schemas│
      │ - function_call  │   │ - execute_tool_call│
      └──────────────────┘   └──────────────────┘
```

## Data Flow

### User Input Flow
```
User types: "Read the README.md file"
    │
    ▼
ElpisREPL.run()
    │
    ▼
AgentOrchestrator.process(user_input)
    │
    ├─► Add to history: {"role": "user", "content": "Read..."}
    │
    ├─► Build messages with system prompt
    │
    ├─► LlamaInference.function_call(messages, tools)
    │       │
    │       └─► Returns: [{"id": "call_1", "function": {"name": "read_file", ...}}]
    │
    ├─► Execute tools concurrently
    │       │
    │       └─► ToolEngine.execute_tool_call() for each tool
    │               │
    │               └─► Returns: {"success": True, "result": {...}}
    │
    ├─► Add tool results to history
    │
    ├─► LlamaInference.function_call(messages, tools) [iteration 2]
    │       │
    │       └─► Returns: None (no more tools needed)
    │
    ├─► LlamaInference.chat_completion(messages)
    │       │
    │       └─► Returns: "Here's what I found in README.md: ..."
    │
    └─► Return final response
            │
            ▼
        ElpisREPL._display_response()
            │
            └─► Rich formatted output to console
```

### ReAct Loop Detail

```
Iteration 1:
  REASON  → Build context: [system, user: "Read file"]
  ACT     → LLM returns: tool_call(read_file, path="README.md")
  OBSERVE → Execute read_file, get: "# Elpis\n..."
  REPEAT  → Continue to iteration 2

Iteration 2:
  REASON  → Build context: [system, user, assistant+tools, tool_results]
  ACT     → LLM returns: No tool calls (ready to respond)
  ACT     → LLM returns: "I've read the file. It contains..."
  RETURN  → Final response to user
```

## Message History Structure

### Example History After Tool Use

```python
message_history = [
    # User request
    {
        "role": "user",
        "content": "Read the README.md file"
    },

    # Assistant decides to use tool
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc123",
                "function": {
                    "name": "read_file",
                    "arguments": '{"file_path": "README.md"}'
                }
            }
        ]
    },

    # Tool execution results
    {
        "role": "tool",
        "content": "Tool call_abc123 succeeded (took 5.23ms):\n{'success': True, 'content': '# Elpis\\n...'}"
    },

    # Final assistant response
    {
        "role": "assistant",
        "content": "I've read the README.md file. It contains information about..."
    }
]
```

## Concurrent Tool Execution

```
Tool Call 1: read_file("file1.py")    ┐
Tool Call 2: read_file("file2.py")    ├─► asyncio.gather()
Tool Call 3: read_file("file3.py")    ┘
         │                │                │
         ▼                ▼                ▼
    Result 1         Result 2         Result 3
         │                │                │
         └────────────────┴────────────────┘
                          │
                          ▼
                    All results combined
                          │
                          ▼
                   Added to history
```

## Error Handling Flow

```
User Input
    │
    ▼
try {
    AgentOrchestrator.process()
        │
        ├─► try { LLM call }
        │       catch { Log error, return error message }
        │
        ├─► try { Tool execution }
        │       catch { Convert to error result, continue loop }
        │
        └─► Max iterations check
                │
                └─► If exceeded: Return "reasoning limit" message
} catch {
    Log exception
    Return error message to user
}
    │
    ▼
ElpisREPL.display_error() or ._display_response()
```

## Special Commands Flow

```
User types: "/clear"
    │
    ▼
ElpisREPL.run()
    │
    ├─► Detects "/" prefix
    │
    └─► ElpisREPL._handle_special_command("/clear")
            │
            ├─► Parse command
            │
            ├─► Execute: AgentOrchestrator.clear_history()
            │
            └─► Display: "Conversation history cleared"
```

## Integration Points

### Week 1 Dependencies
- `LlamaInference`: Async LLM wrapper
  - Used for both `chat_completion()` and `function_call()`
  - Runs in thread pool via `asyncio.to_thread()`

- `build_system_prompt()`: System prompt builder
  - Called in `_build_messages()`
  - Includes tool descriptions

### Week 2 Dependencies
- `ToolEngine`: Tool execution orchestrator
  - `get_tool_schemas()`: Get OpenAI-compatible schemas
  - `execute_tool_call()`: Execute individual tools

### Week 4 Integration
- CLI will initialize all components
- Wire together LLM + Tools + Agent + REPL
- Handle configuration and error cases

## Performance Characteristics

- **Concurrent Tool Execution**: O(max(tool_times)) instead of O(sum(tool_times))
- **History Growth**: O(n) where n = number of messages
- **Iteration Limit**: Max 10 iterations prevents infinite loops
- **Async I/O**: All blocking operations in thread pool

## Key Design Decisions

1. **ReAct Pattern**: Industry-standard for LLM agents
2. **Max Iterations**: Safety limit of 10 prevents runaway loops
3. **Concurrent Tools**: Uses `asyncio.gather()` for parallelism
4. **History Format**: OpenAI-compatible message format
5. **Error Recovery**: Graceful degradation, never crash
6. **Special Commands**: Familiar slash-command syntax
7. **Rich Formatting**: Automatic markdown detection and rendering

## Testing Strategy

- **Unit Tests**: Mock LLM and tools, test logic in isolation
- **Integration Tests**: Mock but realistic behavior, test full flow
- **Future E2E Tests**: Week 4 will test with real LLM and tools
