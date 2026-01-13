# Psyche

Memory server and user client for Elpis inference.

Named after the Greek goddess of the soul/mind.

## Overview

Psyche is the harness component of the Elpis system, providing:

- **MCP Client**: Connects to the Elpis inference server via MCP protocol
- **Memory Server**: Continuous inference loop with context compaction
- **User Client**: Interactive REPL with rich terminal display
- **Tool System**: File operations, bash execution, directory listing, code search

## Architecture

```
User Input -> Psyche REPL -> Memory Server -> MCP Client -> Elpis Server -> LLM
                 ^                |
                 |                v
              Display <- Response/Thoughts
```

## Features

### Continuous Thinking
- Idle thought generation between user interactions
- Configurable thinking interval and max idle thoughts
- Callbacks for displaying internal thoughts

### Memory Management
- Context compaction using sliding window strategy
- Token tracking and estimation
- Optional summarization for compacted context
- Configurable token limits and reserve

### Tool Execution
- **read_file**: Read file contents with size limits
- **write_file**: Create or overwrite files
- **execute_bash**: Run shell commands with safety checks
- **list_directory**: List directory contents with patterns
- **search_codebase**: Search files with ripgrep

### Emotional Integration
- Receives emotional state from Elpis server
- Updates emotion based on interaction quality
- Optional emotional state display in REPL

## Installation

```bash
# From the Elpis root directory
cd psyche
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## Usage

```bash
# Start Psyche - it will automatically spawn the Elpis server
psyche
```

**Note:** Psyche uses MCP's stdio transport, which means it spawns `elpis-server` as a subprocess automatically. You do NOT need to start `elpis-server` separately.

### REPL Commands

- `/help` - Show available commands
- `/status` - Show current context and emotional state
- `/clear` - Clear conversation history
- `/emotion` - Toggle emotional state display
- `/quit` - Exit the REPL

## Configuration

The server can be configured via `ServerConfig`:

```python
from psyche.memory.server import ServerConfig, MemoryServer

config = ServerConfig(
    idle_think_interval=30.0,  # Seconds between idle thoughts
    max_idle_thoughts=3,       # Max thoughts before waiting
    think_temperature=0.9,     # Higher temp for creative thinking
    max_context_tokens=6000,   # Maximum context window
    reserve_tokens=2000,       # Tokens reserved for response
    emotional_modulation=True, # Enable emotional parameter modulation
)
```

## Project Structure

```
psyche/
├── src/psyche/
│   ├── cli.py             # Entry point
│   ├── client/            # User interface
│   │   ├── display.py     # Rich terminal output
│   │   └── repl.py        # Interactive REPL
│   ├── mcp/               # MCP client
│   │   └── client.py      # ElpisClient for server connection
│   ├── memory/            # Memory management
│   │   ├── compaction.py  # Context compaction strategies
│   │   └── server.py      # MemoryServer with inference loop
│   └── tools/             # Tool system
│       ├── tool_engine.py       # Tool orchestrator
│       ├── tool_definitions.py  # Pydantic models for tools
│       └── implementations/     # Individual tool implementations
└── tests/
    ├── unit/              # Unit tests (163 tests)
    └── integration/       # Integration tests
```

## Development

### Run tests

```bash
# All tests
pytest

# With coverage
pytest --cov

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/
```

### Test Results

- 163 tests passing
- 69% code coverage

## API Reference

### ElpisClient

```python
from psyche.mcp.client import ElpisClient

async with ElpisClient().connect() as client:
    # Generate text
    result = await client.generate(
        messages=[{"role": "user", "content": "Hello"}],
        emotional_modulation=True
    )
    print(result.content)
    print(result.emotional_state.quadrant)  # e.g., "excited"

    # Update emotional state
    state = await client.update_emotion("success", intensity=1.0)
```

### MemoryServer

```python
from psyche.mcp.client import ElpisClient
from psyche.memory.server import MemoryServer, ServerConfig

client = ElpisClient()
server = MemoryServer(
    client,
    on_thought=lambda t: print(f"[Thought] {t.content}"),
    on_response=lambda r: print(f"Response: {r}")
)

# Start the inference loop
await server.start()

# Submit user input
server.submit_input("What is the meaning of life?")
```

### ToolEngine

```python
from psyche.tools.tool_engine import ToolEngine, ToolSettings

engine = ToolEngine("/workspace", ToolSettings())

# Execute a tool call
result = await engine.execute_tool_call({
    "function": {
        "name": "read_file",
        "arguments": '{"path": "example.txt"}'
    }
})
```

## License

Psyche is part of the Elpis project, licensed under GNU GPLv3.
