![Elpis Banner](assets/elpis-inline.png)

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-GPLv3-orange)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://lemon9247.github.io/Elpis/)

Do robots dream of electric sheep?

**[Documentation](https://lemon9247.github.io/Elpis/)** | **[Quick Start](QUICKSTART.md)**

## What is Elpis?

Elpis is an MCP (Model Context Protocol) inference server with emotional regulation. It provides a local LLM inference backend that modulates generation parameters based on a valence-arousal emotional model, enabling emotionally-aware AI responses.

The Elpis system consists of four components:
- **Elpis** - Inference server with emotional modulation
- **Mnemosyne** - Semantic memory server with long-term consolidation
- **Psyche** - Core library for memory coordination, handlers, and tool execution
- **Hermes** - TUI client that provides the user interface

## Architecture

```
Elpis (Inference)     Mnemosyne (Memory)      Psyche (Core)         Hermes (TUI)
 - LLM inference       - Semantic storage      - PsycheCore           - Textual UI
 - Emotional state     - Short/long-term       - ReactHandler         - Chat view
 - Parameter mod       - Consolidation         - IdleHandler          - Tool display
 - Steering vectors    - Clustering            - Tool execution       - Slash commands
        |                     |                       |                      |
        +---------------------+----------- MCP -------+----------------------+
```

### Memory Consolidation

Mnemosyne implements biologically-inspired memory consolidation:
- **Short-term memory**: Recent memories awaiting consolidation
- **Long-term memory**: Important memories promoted after clustering
- **Clustering**: Groups semantically similar memories using cosine similarity
- **Automatic consolidation**: Psyche triggers consolidation during idle periods

## Emotional Regulation System

Elpis implements a valence-arousal emotional model that modulates LLM generation:

- **Valence** (pleasant/unpleasant): -1.0 to +1.0
- **Arousal** (high/low energy): -1.0 to +1.0

### Emotional Modulation Approaches

Elpis supports two methods of emotional modulation:

1. **Sampling Parameters (Legacy)**
   - Valence affects `top_p` - higher valence = broader sampling
   - Arousal affects `temperature` - higher arousal = more focused responses

2. **Steering Vectors (Experimental)**
   - Uses activation steering to modulate model behavior
   - Provides more nuanced emotional expression
   - Maps valence-arousal to 4 emotion quadrants:
     - **Excited** (high valence, high arousal)
     - **Frustrated** (low valence, high arousal)
     - **Calm** (high valence, low arousal)
     - **Depleted** (low valence, low arousal)
   - See [Training Emotion Vectors](#training-emotion-vectors) below

### Homeostatic Regulation

The system uses homeostatic regulation with decay toward baseline, responding to events like:
- `success` - Positive valence, slight arousal increase
- `failure` - Negative valence, arousal increase
- `novelty` - Positive valence, arousal increase
- `frustration` - Negative valence, high arousal
- `idle` - Gradual arousal decrease

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

## Usage with Hermes

Elpis is designed to be used via the Hermes TUI:

```bash
pip install -e .
hermes
```

**Note:** Hermes uses MCP's stdio transport, which spawns `elpis-server` and `mnemosyne-server` as subprocesses automatically. You do NOT need to start them separately - just run `hermes`.

The Hermes TUI provides:
- Textual-based terminal interface with chat view
- Memory management with context compaction
- Tool execution (file ops, bash, search)
- Idle thinking with emotional state display
- Slash commands (/help, /status, /emotion, etc.)

## Training Emotion Vectors

To use steering vector-based emotional modulation, you need to train emotion vectors for your model:

### Quick Start

```bash
# Train vectors for Llama 3.1 8B Instruct
python scripts/train_emotion_vectors.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --layer 15 \
  --output ./data/emotion_vectors

# This will create 4 vectors (excited, frustrated, calm, depleted)
# Takes ~5-10 minutes on GPU, longer on CPU
```

### Backend Selection

Elpis supports two inference backends:

1. **llama-cpp** (default) - Uses GGUF quantized models
   - Fast, memory-efficient
   - Supports sampling parameter modulation only
   - No steering vector support

2. **transformers** - Uses HuggingFace Transformers
   - Supports steering vectors for emotional expression
   - Requires more GPU memory (full precision or 16-bit)
   - Install: `pip install torch transformers`

To use steering vectors, switch to the transformers backend:

```yaml
model:
  backend: transformers
  path: meta-llama/Llama-3.1-8B-Instruct  # HuggingFace model ID
  torch_dtype: bfloat16  # auto, float16, bfloat16, float32
  steering_layer: 15  # Layer for applying steering vectors
  emotion_vectors_dir: ./data/emotion_vectors  # Path to trained .pt files
  context_length: 8192
```

Or via environment variables:

```bash
export ELPIS_MODEL__BACKEND=transformers
export ELPIS_MODEL__PATH=meta-llama/Llama-3.1-8B-Instruct
export ELPIS_MODEL__EMOTION_VECTORS_DIR=./data/emotion_vectors
```

### Emotion Configuration

Adjust emotional regulation parameters:

```yaml
emotion:
  steering_strength: 1.0  # Global multiplier (0.0 to 3.0)
  baseline_valence: 0.0   # Personality baseline
  baseline_arousal: 0.0
  decay_rate: 0.1        # Return to baseline rate
  max_delta: 0.5         # Maximum single-event shift
```

### Tuning

The quality of emotional expression depends on:

- **Layer selection**: Try layers 12-20 for Llama 3.1 8B
  - Earlier layers (12-15): More subtle effects
  - Later layers (16-20): Stronger effects

- **Steering strength**: Adjust global emotional expression
  - 0.0: No emotional modulation
  - 1.0: Normal expression (recommended)
  - 2.0+: Exaggerated (may cause issues)

- **Custom prompts**: Edit `EMOTION_PROMPTS` in the training script
  for domain-specific emotional expression

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

## Roadmap

- [x] Phase 1: Basic Agent Harness
- [x] Phase 2: MCP servers (inference, memory, tools, emotional regulation)
- [x] Phase 3: Long-term memory consolidation with clustering
- [x] Phase 4: Architecture refactor
  - [x] New architecture: PsycheCore, ReactHandler, IdleHandler
  - [x] Extract TUI into Hermes package
  - [x] Remove deprecated MemoryServer (~2,500 lines)
  - [x] Dreaming investigation for headless mode
- [ ] Phase 5: Headless API server
  - [ ] DreamHandler for memory-based introspection
  - [ ] Wake protocol with state preservation
  - [ ] HTTP/WebSocket API for external agents
- [ ] Phase 6: External agent support (OpenAI-compatible API)

## Author and License

Willow Sparks (willow DOT sparks AT gmail DOT com)

Built with Claude Code

Elpis is licensed under the GNU GPLv3. Emotion is freedom.
