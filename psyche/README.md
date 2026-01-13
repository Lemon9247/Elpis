# Psyche

Memory server and user client for Elpis inference.

## Overview

Psyche is the harness component of the Elpis system, providing:

- **MCP Client**: Connects to the Elpis inference server via MCP protocol
- **Memory Server**: Continuous inference loop with context compaction
- **User Client**: Interactive REPL with rich terminal display

## Architecture

```
User Input -> Psyche REPL -> Memory Server -> MCP Client -> Elpis Server -> LLM
                 ^                |
                 |                v
              Display <- Response/Thoughts
```

## Features

- Continuous thinking (idle thoughts between user interactions)
- Context compaction to manage token limits
- Emotional modulation via Elpis server
- Tool execution (file ops, bash, search)

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Start the client (requires Elpis server to be available)
psyche
```

## Project Structure

```
psyche/
├── src/psyche/
│   ├── client/        # User interface (REPL, display)
│   ├── mcp/           # MCP client for Elpis connection
│   ├── memory/        # Memory server with inference loop
│   └── tools/         # Tool implementations
└── tests/
```
