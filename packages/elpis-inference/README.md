# elpis-inference

MCP inference server with emotional regulation via steering vectors.

## Overview

Elpis Inference is a standalone MCP (Model Context Protocol) server that provides LLM inference with emotional regulation. It supports two modulation approaches:

1. **Sampling Parameters** - Adjusts temperature/top_p based on valence-arousal
2. **Steering Vectors** - Applies activation steering for nuanced emotional expression

## Features

- 🎭 **Emotional Regulation** - Valence-arousal model with homeostatic decay
- 🔄 **Dual Backends** - llama-cpp (fast, GGUF) or transformers (steering vectors)
- 🎯 **MCP Protocol** - Standard MCP server for easy integration
- 📊 **Configurable** - Extensive configuration via YAML or environment variables
- 🧪 **Well-Tested** - Comprehensive test suite with mocked dependencies

## Installation

### Basic Installation (llama-cpp backend)

```bash
pip install ./packages/elpis-inference
```

### With Transformers Backend (for steering vectors)

```bash
pip install "./packages/elpis-inference[transformers]"
```

### All Backends

```bash
pip install "./packages/elpis-inference[all]"
```

### Development

```bash
pip install "./packages/elpis-inference[dev]"
```

## Quick Start

### 1. Start the Server

```bash
elpis-server
```

The server will start on stdio and wait for MCP client connections.

### 2. Configure MCP Client

Add to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "elpis-inference": {
      "command": "elpis-server",
      "env": {
        "ELPIS_MODEL__BACKEND": "llama-cpp",
        "ELPIS_MODEL__PATH": "./data/models/model.gguf"
      }
    }
  }
}
```

### 3. Use MCP Tools

Available tools:

- `generate` - Text generation with emotional modulation
- `generate_stream_start` - Start streaming generation
- `generate_stream_read` - Read tokens from stream
- `generate_stream_cancel` - Cancel active stream
- `function_call` - Generate tool/function calls
- `update_emotion` - Trigger emotional event
- `reset_emotion` - Reset to baseline
- `get_emotion` - Get current state

## Configuration

### Environment Variables

```bash
# Model settings
export ELPIS_MODEL__BACKEND=transformers
export ELPIS_MODEL__PATH=meta-llama/Llama-3.1-8B-Instruct
export ELPIS_MODEL__EMOTION_VECTORS_DIR=./data/emotion_vectors
export ELPIS_MODEL__STEERING_LAYER=15
export ELPIS_MODEL__TORCH_DTYPE=bfloat16

# Emotion settings
export ELPIS_EMOTION__BASELINE_VALENCE=0.0
export ELPIS_EMOTION__BASELINE_AROUSAL=0.0
export ELPIS_EMOTION__STEERING_STRENGTH=1.0
export ELPIS_EMOTION__DECAY_RATE=0.1
```

### YAML Configuration

Create `config.yaml`:

```yaml
model:
  backend: transformers
  path: meta-llama/Llama-3.1-8B-Instruct
  torch_dtype: bfloat16
  steering_layer: 15
  emotion_vectors_dir: ./data/emotion_vectors

emotion:
  baseline_valence: 0.0
  baseline_arousal: 0.0
  steering_strength: 1.0
  decay_rate: 0.1
  max_delta: 0.5
```

## Backends

### llama-cpp (Default)

- Uses GGUF quantized models
- Fast and memory-efficient
- Sampling parameter modulation only
- CPU and GPU support

**Example:**
```bash
export ELPIS_MODEL__BACKEND=llama-cpp
export ELPIS_MODEL__PATH=./models/model.gguf
elpis-server
```

### Transformers

- Uses HuggingFace models
- Supports activation steering
- Requires GPU for reasonable performance
- More memory intensive

**Example:**
```bash
export ELPIS_MODEL__BACKEND=transformers
export ELPIS_MODEL__PATH=meta-llama/Llama-3.1-8B-Instruct
export ELPIS_MODEL__EMOTION_VECTORS_DIR=./data/emotion_vectors
elpis-server
```

## Steering Vectors

For advanced emotional expression using activation steering:

### 1. Train Emotion Vectors

```bash
python scripts/train_emotion_vectors.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --layer 15 \
  --output ./data/emotion_vectors
```

This creates 4 vectors: excited, frustrated, calm, depleted.

### 2. Configure Server

```bash
export ELPIS_MODEL__BACKEND=transformers
export ELPIS_MODEL__EMOTION_VECTORS_DIR=./data/emotion_vectors
elpis-server
```

### 3. Tuning

- **Layer**: Try 12-20 for Llama 3.1 8B
- **Strength**: 0.0-3.0 (1.0 recommended)
- **Custom vectors**: Edit EMOTION_PROMPTS in training script

## Emotional State

Elpis uses a valence-arousal model:

```
         High Arousal (+1)
              |
  Frustrated  |  Excited
  (-1, +1)    |  (+1, +1)
              |
-1 ----------------------------- +1
        Valence
              |
  Depleted    |  Calm
  (-1, -1)    |  (+1, -1)
              |
         Low Arousal (-1)
```

### Emotional Events

Trigger events to update state:

- `success`, `failure` - Outcomes
- `novelty`, `insight` - Discoveries  
- `frustration`, `blocked` - Obstacles
- `test_passed`, `test_failed` - Testing
- `error` - Unexpected issues
- `idle` - Decay toward baseline

## API Reference

### MCP Tools

#### generate

Generate text with emotional modulation.

**Arguments:**
- `messages` (array) - Chat messages in OpenAI format
- `max_tokens` (int) - Maximum tokens to generate
- `temperature` (float, optional) - Override temperature
- `emotional_modulation` (bool) - Enable emotion modulation (default: true)

**Returns:**
- `content` (string) - Generated text
- `emotional_state` (object) - Current emotional state
- `modulated_params` (object) - Applied sampling parameters

#### update_emotion

Trigger an emotional event.

**Arguments:**
- `event_type` (string) - Event type (success, failure, etc.)
- `intensity` (float) - Event intensity (0.0-2.0, default: 1.0)
- `context` (string, optional) - Description for logging

**Returns:**
- Current emotional state object

#### get_emotion

Get current emotional state.

**Returns:**
- `valence` (float) - Pleasant/unpleasant (-1 to +1)
- `arousal` (float) - High/low energy (-1 to +1)
- `quadrant` (string) - Emotional quadrant
- `baseline_valence` (float) - Baseline valence
- `baseline_arousal` (float) - Baseline arousal
- `update_count` (int) - Number of state updates

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
ruff check src/
mypy src/
```

### Project Structure

```
packages/elpis-inference/
├── src/elpis_inference/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point
│   ├── server.py           # MCP server
│   ├── emotion/
│   │   ├── state.py        # Emotional state
│   │   └── regulation.py   # Homeostasis
│   ├── llm/
│   │   ├── base.py         # InferenceEngine interface
│   │   ├── inference.py    # LlamaInference (llama-cpp)
│   │   └── transformers_inference.py  # TransformersInference
│   └── config/
│       └── settings.py     # Configuration
├── tests/
├── data/
├── pyproject.toml
└── README.md
```

## License

GPL-3.0

## Credits

Built with:
- [MCP SDK](https://github.com/anthropics/mcp) - Model Context Protocol
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - GGUF inference
- [Transformers](https://github.com/huggingface/transformers) - HuggingFace models
- [Pydantic](https://github.com/pydantic/pydantic) - Configuration management

## Support

- [Documentation](../../README.md)
- [Examples](../../examples/)
- [Issues](https://github.com/yourusername/elpis/issues)
