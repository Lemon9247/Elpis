![Elpis Banner](assets/elpis-inline.png)

Do robots dream of electric sheep?

## What is Elpis?

Elpis is an MCP (Model Context Protocol) inference server with emotional regulation. It provides a local LLM inference backend that modulates generation parameters based on a valence-arousal emotional model, enabling emotionally-aware AI responses.

Elpis is designed to work with **Psyche** (located in the `psyche/` subdirectory), which provides the user interface, memory management, and tool execution capabilities.

## Architecture

```
Elpis (MCP Server)          Psyche (Client/Harness)
 - LLM inference             - Memory server
 - Emotional regulation      - Context compaction
 - Parameter modulation      - Tool execution
                             - User REPL interface
```

## Emotional Regulation System

Elpis implements a valence-arousal emotional model that affects LLM inference parameters:

- **Valence** (pleasant/unpleasant): Affects `top_p` - higher valence = broader sampling
- **Arousal** (high/low energy): Affects `temperature` - higher arousal = more focused responses

The system uses homeostatic regulation with decay toward baseline, responding to events like:
- `success` - Positive valence, slight arousal increase
- `failure` - Negative valence, arousal increase
- `novelty` - Positive valence, arousal increase
- `frustration` - Negative valence, high arousal
- `idle` - Gradual arousal decrease

## MCP Tools

Elpis exposes these MCP tools:

| Tool | Description |
|------|-------------|
| `generate` | Text generation with emotional modulation |
| `function_call` | Tool/function call generation |
| `update_emotion` | Trigger an emotional event |
| `reset_emotion` | Reset to baseline state |
| `get_emotion` | Get current emotional state |

## MCP Resources

| Resource URI | Description |
|--------------|-------------|
| `emotion://state` | Current valence-arousal state |
| `emotion://events` | Available emotional event types |

## Installation

### Requirements

- Python 3.10 or higher
- 8-10GB RAM (for model)
- GPU recommended (NVIDIA CUDA or AMD ROCm), CPU fallback available

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/elpis.git
cd elpis
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

For CPU-only:
```bash
pip install -e .
```

For NVIDIA GPU (CUDA):
```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --no-cache-dir
pip install -e .
```

For AMD GPU (ROCm):
```bash
CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python --no-cache-dir
pip install -e .
```

4. **Download the model**
```bash
python scripts/download_model.py
```

This downloads Llama 3.1 8B Instruct (Q5_K_M, ~6.8GB) from HuggingFace.

5. **Run the MCP server**
```bash
elpis-server
```

## Usage with Psyche

Elpis is designed to be used via the Psyche harness:

```bash
cd psyche
pip install -e .
psyche
```

**Note:** Psyche uses MCP's stdio transport, which spawns `elpis-server` as a subprocess automatically. You do NOT need to start `elpis-server` separately - just run `psyche`.

The Psyche client provides:
- Interactive REPL with Rich terminal output
- Memory management with context compaction
- Tool execution (file ops, bash, search)
- Continuous inference loop with idle thinking

## Development

### Install dev dependencies

```bash
pip install -e ".[dev]"
pre-commit install
```

### Run tests

```bash
# All tests
pytest

# With coverage
pytest --cov

# MCP server tests only
pytest tests/integration/test_mcp_server.py
```

### Code quality

```bash
# Format and lint
ruff check --fix .
ruff format .

# Type checking
mypy src/elpis
```

## Project Structure

```
src/elpis/
 - config/       # Settings management
 - emotion/      # Valence-arousal state and regulation
 - llm/          # LLM inference with llama.cpp
 - mcp/          # MCP protocol components
 - server.py     # MCP server entry point
 - utils/        # Hardware detection, logging

psyche/          # Client harness (separate package)
 - memory/       # Context compaction, continuous inference
 - client/       # REPL and display
 - tools/        # Tool definitions and implementations
 - mcp/          # MCP client for Elpis connection
```

## Test Results

- **Elpis**: 81 tests, 91% coverage
- **Psyche**: 163 tests, 69% coverage

## Roadmap

- [x] Phase 1: Basic Agent Harness
- [x] Phase 2A: Tool system fixes
- [x] Phase 2B: MCP inference server
- [x] Phase 2C: Emotional regulation system
- [x] Phase 2D: Psyche harness with memory server
- [ ] Phase 3: Long-term memory (ChromaDB integration)
- [ ] Phase 4: Advanced emotional dynamics

## Author and License

Willow Sparks (willow DOT sparks AT gmail DOT com)
Built with Claude Code

Elpis is licensed under the GNU GPLv3. Emotion is freedom.
