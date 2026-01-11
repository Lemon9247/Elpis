![Elpis Banner](assets/elpis-inline.png)

Do robots dream of electric sheep?

## What is Elpis?

Elpis (WIP) is an agentic harness and inference package for local LLMs. Alongside standard features of "coding" harnesses like tool use and filesystem access, Elpis is designed to let agents run continuously and statefully with long-term memory and an "emotional regulation" system

## Artificial "Emotion/Memory" System

We do *not* claim that Elpis magically gives your LLM real emotions with the full depth of human emotions, that would be plain silly. Instead, Elpis implements a secondary machine learned model which controls a set of "hormone functions". This system then interacts with the LLM during inference to globally modulate the weights and outputs of the model, in a similar manner to how biological hormones affect global changes to the nervous system. You can think of this being a bit like an "artificial amygdala" for an "artificial intelligence".

This may seem romanticised and useless, but I believe this is crucial for agents which have internal state that run over long-term periods. There are other projects which implement long-term memory systems for agents, however these typically focus on static distinctions between short-term and long-term memory. The onus of telling the agent to remember what context is important - and what is *not* - may fall on the user of the system.

Implementing an amygdala-inspired system is my proposed solution to this approach, as it allows for intuitive mapping of contextual importance and topics. e.g. context which is unimportant can "feel boring" to the model and get discarded, whereas important stuff "feels exciting" and is remembered in greater detail.

Other ideas for this system include tying compaction into biologically-inspired sleep mechanisms, so that long-term memories can be classified, pruned and recontextualised rather than filling your drive with slop.

## Current Status

**Phase 1: Basic Agent Harness** (In Progress)

We're currently implementing the foundational agent harness with:
- ✅ Project structure and configuration
- 🚧 LLM inference with llama.cpp
- 🚧 Tool system (file ops, bash, search)
- 🚧 Async REPL interface
- ⏳ Memory and emotion systems (Phase 2-3)

## Technologies

Phase 1 uses these open-source technologies:
- **llama-cpp-python** - Fast local inference with CUDA/ROCm/CPU support
- **Llama 3.1 8B Instruct** - Function-calling capable language model
- **Pydantic** - Data validation and settings management
- **Rich & prompt-toolkit** - Beautiful async terminal interface
- **pytest-asyncio** - Comprehensive async testing

Future phases will add ChromaDB for vector memory and PyTorch for emotion modeling.

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

5. **Run Elpis**
```bash
elpis
```

Or with debug logging:
```bash
elpis --debug
```

## Usage

Once running, interact with Elpis naturally:

```
elpis> Read the file workspace/example.py and explain what it does

elpis> Write a function to calculate fibonacci numbers to workspace/math.py

elpis> Search for "TODO" in all Python files
```

### Special Commands

- `/help` - Show help message
- `/clear` - Clear conversation history
- `/exit` - Exit the REPL

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

# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration
```

### Code quality

```bash
# Format and lint
ruff check --fix .
ruff format .

# Type checking
mypy src/elpis
```

## Configuration

Configuration is managed via `configs/config.default.toml` and can be overridden with environment variables:

```bash
# Use CPU instead of GPU
export ELPIS_MODEL__GPU_LAYERS=0

# Change workspace directory
export ELPIS_TOOLS__WORKSPACE_DIR=/custom/path

# Enable debug logging
export ELPIS_LOGGING__LEVEL=DEBUG
```

Create `configs/config.local.toml` for local overrides (git-ignored).

## Architecture

```
src/elpis/
├── config/       # Pydantic settings and TOML configs
├── llm/          # LLM inference wrapper (async)
├── tools/        # Tool definitions and implementations
├── agent/        # ReAct pattern orchestrator
└── utils/        # Hardware detection, logging, exceptions
```

See `scratchpad/phase1-implementation-plan.md` for detailed architecture documentation.

## Troubleshooting

### Model download fails
- Check internet connection
- Ensure you have ~7GB free disk space
- Try downloading manually from [HuggingFace](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)

### llama-cpp-python won't build
- Ensure you have CMake and a C++ compiler installed
- For CUDA: Install CUDA Toolkit 11.7+
- For ROCm: Install ROCm 5.0+
- Try prebuilt wheels: `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`

### Out of memory
- Reduce `gpu_layers` in config (or set to 0 for CPU-only)
- Use a smaller quantization (Q4_K_M instead of Q5_K_M)
- Close other applications

## Roadmap

- [x] Phase 1: Basic Agent Harness (4 weeks)
  - [x] Project setup and configuration
  - [ ] LLM inference with GPU support
  - [ ] Tool system (5 core tools)
  - [ ] Async REPL interface
  - [ ] Comprehensive tests (>80% coverage)

- [ ] Phase 2: Memory System (4 weeks)
  - [ ] Three-tier memory (sensory, short-term, long-term)
  - [ ] ChromaDB vector storage
  - [ ] Memory consolidation
  - [ ] Importance scoring

- [ ] Phase 3: Emotional System (4 weeks)
  - [ ] Neuromodulator states (dopamine, serotonin, etc.)
  - [ ] Emotional event triggers
  - [ ] Behavior modulation
  - [ ] Emotion-aware memory

- [ ] Phase 4: Integration & Polish (2 weeks)
  - [ ] Tune emotional dynamics
  - [ ] Optimize memory consolidation
  - [ ] Performance improvements
  - [ ] Documentation

## Author and License

Willow Sparks (willow DOT sparks AT gmail DOT com)
Built with Claude Code

Elpis is licensed under the GNU GPLv3. Emotion is freedom.