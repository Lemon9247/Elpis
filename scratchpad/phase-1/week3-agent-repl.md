# Week 3: Agent & REPL Implementation

This document describes the Week 3 implementation of the Elpis agent orchestrator and REPL interface.

## Overview

Week 3 delivers the core agent reasoning loop and user interface:
- **AgentOrchestrator**: Implements the ReAct (Reasoning + Acting) pattern
- **ElpisREPL**: Async REPL interface with rich formatting

## AgentOrchestrator

### Location
`src/elpis/agent/orchestrator.py`

### Purpose
The orchestrator manages the agent's main reasoning loop, coordinating between the LLM and tool execution.

### Key Features

1. **ReAct Pattern Implementation**
   - **REASON**: Builds context with system prompt and conversation history
   - **ACT**: Generates LLM response (text or tool calls)
   - **OBSERVE**: Executes tools concurrently and collects results
   - **REPEAT**: Iterates until final answer is reached

2. **Concurrent Tool Execution**
   - Uses `asyncio.gather()` to run multiple tools in parallel
   - Handles exceptions gracefully, converting them to error results

3. **Safety Features**
   - Maximum iteration limit (10) to prevent infinite loops
   - Comprehensive error handling with fallback messages
   - Exception logging with loguru

4. **Message History Management**
   - Maintains full conversation context
   - Supports clearing history
   - Helper methods for history inspection

### API Reference

```python
class AgentOrchestrator:
    def __init__(
        self,
        llm: LlamaInference,
        tools: ToolEngine,
        settings: Any = None,
    ):
        """Initialize the agent orchestrator."""

    async def process(self, user_input: str) -> str:
        """
        Process user input using the ReAct pattern.

        Returns:
            Agent's final response to the user
        """

    def clear_history(self) -> None:
        """Clear the conversation history."""

    def get_history_length(self) -> int:
        """Get the current length of message history."""

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """Get the last message in history."""
```

### Usage Example

```python
from elpis.llm.inference import LlamaInference
from elpis.tools.tool_engine import ToolEngine
from elpis.agent.orchestrator import AgentOrchestrator

# Initialize components
llm = LlamaInference(settings.model)
tools = ToolEngine(settings.tools.workspace_dir, settings)
agent = AgentOrchestrator(llm, tools, settings)

# Process user input
response = await agent.process("Read the README.md file")
print(response)

# Clear history when needed
agent.clear_history()
```

### Message Flow

1. User input → Added to history as `{"role": "user", "content": "..."}`
2. LLM generates response:
   - If tool calls → Add to history as `{"role": "assistant", "content": None, "tool_calls": [...]}`
   - If text response → Add to history as `{"role": "assistant", "content": "..."}`
3. Tool results → Added to history as `{"role": "tool", "content": "..."}`
4. Loop continues until LLM returns text response (no tool calls)

### Error Handling

- **Tool execution errors**: Converted to error results, loop continues
- **LLM errors**: Logged and returned as error message to user
- **Max iterations**: Returns fallback message asking user to rephrase
- **Unexpected exceptions**: Caught, logged, and returned as error message

## ElpisREPL

### Location
`src/elpis/agent/repl.py`

### Purpose
Provides an interactive command-line interface for users to interact with the agent.

### Key Features

1. **Async Input Handling**
   - Uses `prompt_toolkit` for async, non-blocking input
   - Persistent command history (saved to `.elpis_history`)
   - Supports multiline input (configurable)

2. **Rich Output Formatting**
   - Automatic markdown rendering for responses with markdown syntax
   - Syntax highlighting for code blocks
   - Color-coded panels for different message types
   - Clean, professional output using Rich library

3. **Special Commands**
   - `/help` - Display welcome banner and help
   - `/clear` - Clear conversation history
   - `/exit` or `/quit` - Exit the REPL
   - `/status` - Display debug info (hidden command)

4. **Error Handling**
   - Graceful Ctrl+C handling (continues REPL)
   - Graceful Ctrl+D handling (exits REPL)
   - Exception display with error formatting

### API Reference

```python
class ElpisREPL:
    def __init__(
        self,
        agent: AgentOrchestrator,
        history_file: str = ".elpis_history",
    ):
        """Initialize the REPL interface."""

    async def run(self) -> None:
        """
        Main REPL loop.

        Continuously prompts for input and processes through agent.
        """

    def display_error(self, error_message: str) -> None:
        """Display an error message with red formatting."""

    def display_info(self, info_message: str) -> None:
        """Display an informational message."""

    def display_success(self, success_message: str) -> None:
        """Display a success message with green formatting."""
```

### Usage Example

```python
from elpis.agent.orchestrator import AgentOrchestrator
from elpis.agent.repl import ElpisREPL

# Create orchestrator (see above)
agent = AgentOrchestrator(llm, tools, settings)

# Create and run REPL
repl = ElpisREPL(agent)
await repl.run()
```

### Output Formatting

The REPL automatically formats responses based on content:

1. **Markdown Content**: Renders with full markdown support
   - Headers, bold, italic, code, lists, etc.
   - Displayed in a green-bordered panel

2. **Plain Text**: Simple text in a panel
   - Used when no markdown syntax detected

3. **Code Blocks**: Syntax highlighting
   - Automatically detected from triple backticks
   - Language-specific highlighting

### Command History

- Saved to home directory: `~/.elpis_history`
- Persists across sessions
- Searchable with Ctrl+R (reverse search)
- Navigate with up/down arrows

## Testing

### Unit Tests

**test_orchestrator.py** (18 test cases)
- Initialization and basic processing
- ReAct loop iterations
- Tool execution (single and concurrent)
- History management
- Error handling
- Max iteration limit
- Message building
- Tool result formatting

**test_repl.py** (19 test cases)
- Initialization
- Special command handling
- Response formatting (markdown, code, plain text)
- Display methods (error, info, success)
- User input processing
- Multiple conversation turns
- Error handling (KeyboardInterrupt, EOFError, exceptions)

### Integration Tests

**test_agent_workflow.py** (11 test cases)
- Full ReAct loop with tool execution
- Multiple conversation turns
- History persistence
- REPL integration with agent
- Error recovery
- Concurrent tool execution
- Message building and formatting

### Running Tests

```bash
# Run all Week 3 tests
pytest tests/unit/test_orchestrator.py tests/unit/test_repl.py -v

# Run integration tests
pytest tests/integration/test_agent_workflow.py -v

# Check coverage
pytest tests/unit/test_orchestrator.py tests/unit/test_repl.py --cov=src/elpis/agent
```

## Integration Notes for Week 4

### Dependencies

Week 3 components depend on:
- `LlamaInference` from Week 1 (currently using real implementation)
- `ToolEngine` from Week 2 (currently using real implementation)
- `build_system_prompt()` from Week 1 (placeholder implemented)

### CLI Integration

The Week 4 CLI should:
1. Load settings from config file
2. Initialize LLM, tools, and agent
3. Create and run REPL
4. Handle startup errors gracefully

Example CLI structure:
```python
async def main():
    # Load settings
    settings = Settings()

    # Initialize components
    llm = LlamaInference(settings.model)
    tools = ToolEngine(settings.tools.workspace_dir, settings)
    agent = AgentOrchestrator(llm, tools, settings)

    # Run REPL
    repl = ElpisREPL(agent)
    await repl.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration

No Week 3-specific configuration required. Uses:
- `settings.model` for LLM configuration
- `settings.tools` for tool configuration
- `settings.logging` for logging configuration

### Logging

Week 3 uses loguru for logging:
- `logger.info()` - Important events (tool execution, history clearing)
- `logger.debug()` - Detailed flow (ReAct iterations, message building)
- `logger.error()` - Errors and exceptions
- `logger.exception()` - Exceptions with full traceback

### Performance Considerations

1. **Concurrent Tool Execution**: Multiple tools run in parallel via `asyncio.gather()`
2. **Async Throughout**: All I/O operations are async
3. **History Management**: History grows indefinitely; users should use `/clear` periodically
4. **Max Iterations**: Prevents runaway loops at cost of incomplete responses

### Known Limitations

1. **History Size**: No automatic truncation or summarization
2. **Context Window**: No handling of context length limits
3. **Streaming**: LLM responses are not streamed (all-at-once)
4. **Tool Timeouts**: Individual tool timeouts not implemented at orchestrator level

## Future Enhancements (Phase 2+)

- Streaming responses with Rich Live display
- Context window management and summarization
- Emotion integration (emotional state tracking)
- Memory integration (long-term memory retrieval)
- Tool retry logic with backoff
- Conversation branching and rollback
- Multi-turn planning with sub-goals

## Conclusion

Week 3 delivers a fully functional agent orchestrator with ReAct pattern and a beautiful REPL interface. All components are well-tested, async-first, and ready for integration in Week 4.
