![Elpis Banner](assets/elpis-inline.png)

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-GPLv3-orange)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://lemon9247.github.io/Elpis/)

Do robots dream of electric sheep?

**[Documentation](https://lemon9247.github.io/Elpis/)** | **[Quick Start](QUICKSTART.md)**

## What is Elpis?

Elpis is a system for giving AI persistent memory and emotional state. It provides local LLM inference with emotional modulation, semantic memory with consolidation, and a server architecture for remote operation.

The system consists of four components:
- **Elpis** - Inference MCP server with emotional modulation via valence-arousal model
- **Mnemosyne** - Memory MCP server with ChromaDB backend and short/long-term consolidation
- **Psyche** - Core library and HTTP server for memory coordination, tool execution, and dreaming
- **Hermes** - TUI client that connects to a Psyche server and executes tools locally

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     PSYCHE SERVER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  PsycheCore  │  │    Elpis     │  │  Mnemosyne   │    │
│  │              │  │  (Inference) │  │   (Memory)   │    │
│  │ - Context    │  │              │  │              │    │
│  │ - Memory     │  │ - LLM gen    │  │ - ChromaDB   │    │
│  │ - Dreams     │  │ - Emotion    │  │ - Clustering │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                         MCP (stdio)                      │
└─────────────────────────┬────────────────────────────────┘
                          │ HTTP API (OpenAI-compatible)
                          │
           ┌──────────────┴──────────────┐
           │                             │
    ┌──────┴──────┐               ┌──────┴──────┐
    │   Hermes    │               │   External  │
    │    (TUI)    │               │   Clients   │
    │             │               │             │
    │ - Chat view │               │ - Any HTTP  │
    │ - Tools     │               │   client    │
    └─────────────┘               └─────────────┘
```

### Server Components

- **PsycheCore** - Central coordination layer for context, memory, and emotional state
- **DreamHandler** - Memory-based introspection when no clients are connected
- **RemotePsycheClient** - HTTP client in Hermes for connecting to the server

### Memory System

Mnemosyne provides semantic memory with:
- **Short/long-term consolidation**: Important memories promoted via clustering
- **Hybrid search**: BM25 + vector embeddings with Reciprocal Rank Fusion
- **Quality scoring**: Recency, importance, content length weighting
- **Mood-congruent retrieval**: Boost memories matching current emotional state

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

The system uses homeostatic regulation with decay toward baseline. Events shift emotional state (success, failure, novelty, frustration, idle), with context-aware dynamics:

- **Event compounding/dampening**: Repeated events adjust intensity
- **Mood inertia**: Resistance to rapid emotional swings
- **Quadrant-specific decay**: Frustration persists longer, calm fades faster
- **Behavioral monitoring**: Detects retry loops and failure streaks
- **Trajectory tracking**: Velocity, trend, spiral detection, momentum

### Emotion-Shaped Dreaming

Psyche uses emotional state to guide dream content during idle periods - frustrated states trigger resolution dreams, depleted states trigger restoration dreams, etc.

## Installation

### Requirements

- Python 3.10 or higher
- 8-10GB RAM (for model)
- GPU recommended (NVIDIA CUDA or AMD ROCm), CPU fallback available

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Lemon9247/Elpis.git
cd Elpis
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

## Usage

### Start the Psyche Server

Run Psyche as a persistent server:

```bash
psyche-server
```

This starts the Psyche HTTP server on port 8741, which:
- Manages memory and context
- Handles inference via Elpis MCP server
- Dreams when no clients are connected
- Provides an OpenAI-compatible API

### Connect with Hermes TUI

Connect with the Hermes terminal interface:

```bash
hermes
```

By default, Hermes connects to `http://127.0.0.1:8741`. To connect to a different server:

```bash
hermes --server http://myserver:8741
```

The Hermes TUI provides:
- Terminal interface with chat view
- Automatic tool execution (file ops, bash, search)
- Emotional state display
- Slash commands (/help, /status, /emotion, etc.)

In server mode:
- Psyche manages memory and executes memory tools server-side
- Hermes executes file/bash/search tools locally
- Multiple clients can connect (though memories are shared)
- Server dreams when no clients are connected

### Direct Server Access

Any HTTP client can use the OpenAI-compatible API:

```bash
curl http://localhost:8741/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

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

```toml
[model]
backend = "transformers"
path = "meta-llama/Llama-3.1-8B-Instruct"  # HuggingFace model ID
torch_dtype = "bfloat16"  # auto, float16, bfloat16, float32
steering_layer = 15  # Layer for applying steering vectors
emotion_vectors_dir = "./data/emotion_vectors"  # Path to trained .pt files
context_length = 8192
```

Or via environment variables:

```bash
export ELPIS_MODEL__BACKEND=transformers
export ELPIS_MODEL__PATH=meta-llama/Llama-3.1-8B-Instruct
export ELPIS_MODEL__EMOTION_VECTORS_DIR=./data/emotion_vectors
```

### Emotion Configuration

Adjust emotional regulation parameters:

```toml
[emotion]
steering_strength = 1.0  # Global multiplier (0.0 to 3.0)
baseline_valence = 0.0   # Personality baseline
baseline_arousal = 0.0
decay_rate = 0.1         # Return to baseline rate
max_delta = 0.5          # Maximum single-event shift
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
- [x] Phase 4: Architecture refactor to server/client model
- [x] Phase 5: Dynamic emotion updates (compounding, inertia, behavioral monitoring)
- [x] Phase 6: Emotional depth (trajectory tracking, emotion-shaped dreaming)
- [x] Phase 7: Memory retrieval quality (hybrid search, quality scoring)
- [ ] Future: Personality profiles, memory cleanup, extended emotional model (PAD)

## Author and License

Willow Sparks (willow DOT sparks AT gmail DOT com)

Built with Claude Code

Elpis is licensed under the GNU GPLv3. Emotion is freedom.
