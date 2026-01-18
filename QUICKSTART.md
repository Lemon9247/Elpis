# Elpis Quickstart Guide

Get up and running with Elpis in 5 minutes.

## What is Elpis?

Elpis is a system for giving AI persistent memory and emotional state. It provides:

- **Persistent memory** via Mnemosyne with ChromaDB backend
- **Emotional regulation** using a valence-arousal model
- **Local and remote modes** for flexible deployment
- **Dreaming** when idle (memory-based introspection)

## Prerequisites

- Python 3.10 or higher
- 8-10GB RAM minimum
- GPU recommended (NVIDIA CUDA or AMD ROCm)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Lemon9247/Elpis.git
cd elpis
```

### 2. Install Dependencies

**Basic installation** (llama-cpp backend):
```bash
pip install -e .
```

**With GPU support**:
```bash
# NVIDIA CUDA
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install -e .

# AMD ROCm
CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install -e .
```

**For steering vectors** (optional):
```bash
pip install torch transformers
```

### 3. Download a Model

Download a GGUF model (for llama-cpp backend):

```bash
mkdir -p data/models

# Using huggingface-cli
huggingface-cli download \
  TheBloke/Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf \
  --local-dir data/models --local-dir-use-symlinks False
```

Or download manually from [HuggingFace](https://huggingface.co/models?search=gguf).

## Quick Start Options

Choose your path based on what you want to do:

### Option A: Local Mode with Hermes (Recommended)

Run Hermes directly - it spawns Elpis and Mnemosyne as subprocesses:

```bash
hermes
```

That's it! Hermes manages the MCP servers automatically via stdio transport.

### Option B: Server Mode (Remote Access)

Run Psyche as a persistent server for remote connections:

```bash
# Terminal 1: Start the server
psyche-server

# Terminal 2: Connect with Hermes
hermes --server http://localhost:8741
```

This mode enables:
- Remote access from different machines
- Persistent memory across sessions
- Dreaming when no clients are connected

### Option C: Run Examples

Run the included examples to see the inference engine:

```bash
# Basic inference
python examples/01_basic_inference.py

# Emotional modulation
python examples/02_emotional_modulation.py

# Steering vectors (requires setup)
python examples/03_steering_vectors.py
```

See [examples/README.md](examples/README.md) for details.

### Option D: Use Programmatically

Use Elpis in your own Python code:

```python
import asyncio
from elpis.llm.inference import LlamaInference
from elpis.config.settings import ModelSettings
from elpis.emotion.state import EmotionalState

async def main():
    # Configure
    settings = ModelSettings(
        path="./data/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        context_length=8192,
    )

    # Initialize
    llm = LlamaInference(settings)
    emotion = EmotionalState(valence=0.7, arousal=0.5)  # Excited

    # Generate
    response = await llm.chat_completion(
        messages=[{"role": "user", "content": "Hello!"}],
        **emotion.get_modulated_params()
    )

    print(response)

asyncio.run(main())
```

## Configuration

### Basic Configuration

Create a config file at `config.yaml`:

```yaml
model:
  backend: llama-cpp  # or 'transformers'
  path: ./data/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
  context_length: 8192
  gpu_layers: 35
  temperature: 0.7
  top_p: 0.9

emotion:
  baseline_valence: 0.0   # Neutral baseline
  baseline_arousal: 0.0
  decay_rate: 0.1         # Return to baseline rate
  steering_strength: 1.0  # Global emotional strength
```

### Environment Variables

Or use environment variables:

```bash
# Model settings
export ELPIS_MODEL__PATH=./data/models/model.gguf
export ELPIS_MODEL__CONTEXT_LENGTH=8192

# Emotion settings
export ELPIS_EMOTION__BASELINE_VALENCE=0.0
export ELPIS_EMOTION__STEERING_STRENGTH=1.0
```

## Advanced: Steering Vectors

For more nuanced emotional expression, use steering vectors with the transformers backend.

### 1. Install Dependencies

```bash
pip install torch transformers
```

### 2. Train Emotion Vectors

```bash
python scripts/train_emotion_vectors.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --layer 15 \
  --output ./data/emotion_vectors
```

This takes ~5-10 minutes on GPU and creates 4 emotion vectors.

### 3. Configure Transformers Backend

Update your config:

```yaml
model:
  backend: transformers
  path: meta-llama/Llama-3.1-8B-Instruct
  torch_dtype: bfloat16
  steering_layer: 15
  emotion_vectors_dir: ./data/emotion_vectors
```

Or via environment:

```bash
export ELPIS_MODEL__BACKEND=transformers
export ELPIS_MODEL__PATH=meta-llama/Llama-3.1-8B-Instruct
export ELPIS_MODEL__EMOTION_VECTORS_DIR=./data/emotion_vectors
```

### 4. Use Steering Vectors

```python
from elpis.llm.transformers_inference import TransformersInference
from elpis.emotion.state import EmotionalState

# Initialize with steering support
llm = TransformersInference(settings)

# Create emotional state
emotion = EmotionalState(valence=0.8, arousal=0.6)
coeffs = emotion.get_steering_coefficients()

# Generate with steering
response = await llm.chat_completion(
    messages=[{"role": "user", "content": "How do you feel?"}],
    emotion_coefficients=coeffs
)
```

## Understanding Emotional States

Elpis uses a **valence-arousal** model:

- **Valence**: Pleasant (+1) to unpleasant (-1)
- **Arousal**: High energy (+1) to low energy (-1)

These map to **four quadrants**:

| Quadrant | Valence | Arousal | Examples |
|----------|---------|---------|----------|
| **Excited** | High (+) | High (+) | Energized, enthusiastic, thrilled |
| **Frustrated** | Low (-) | High (+) | Annoyed, stressed, agitated |
| **Calm** | High (+) | Low (-) | Peaceful, content, relaxed |
| **Depleted** | Low (-) | Low (-) | Tired, sad, drained |

### Setting Emotional State

**Directly:**
```python
state = EmotionalState(valence=0.7, arousal=0.5)
```

**Via events:**
```python
from elpis.emotion.regulation import HomeostasisRegulator

regulator = HomeostasisRegulator(state)
regulator.process_event("success", intensity=1.2)
regulator.process_event("frustration", intensity=0.8)
```

**Available events:**
- `success`, `failure` - Outcomes
- `novelty`, `insight` - Discoveries
- `frustration`, `blocked` - Obstacles
- `test_passed`, `test_failed` - Testing
- `error` - Unexpected issues
- `idle` - Decay toward baseline

## Debugging & Tools

Elpis includes developer utilities:

### Visualize Emotional States

```bash
# Show grid of valence-arousal mappings
python scripts/debug_emotion_state.py --grid

# Analyze specific state
python scripts/debug_emotion_state.py --state 0.8 0.6

# Show transition paths
python scripts/debug_emotion_state.py --transitions
```

### Inspect Emotion Vectors

```bash
# Analyze trained vectors
python scripts/inspect_emotion_vectors.py ./data/emotion_vectors --all

# Check quality
python scripts/inspect_emotion_vectors.py ./data/emotion_vectors --quality
```

### Interactive REPL

```bash
# Experiment with emotional states
python scripts/emotion_repl.py
```

```
emotion> set 0.8 0.6
emotion> event success 1.5
emotion> info
```

## Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run specific test file
pytest tests/elpis/unit/test_emotion.py

# With coverage
pytest --cov=src --cov-report=term-missing
```

## Troubleshooting

### "Module not found" errors

Make sure you installed Elpis:
```bash
pip install -e .
```

### Model fails to load

Check:
1. Model file exists: `ls -lh data/models/`
2. Enough RAM available (8GB+ needed)
3. GPU drivers installed (for GPU inference)

### "transformers not installed"

For steering vectors:
```bash
pip install torch transformers
```

### Out of memory

Try:
- Smaller model (7B instead of 70B)
- Lower `gpu_layers` (or 0 for CPU-only)
- Reduce `context_length`
- Use quantized model (Q4_K_M instead of Q5_K_M)

### Slow generation

Check:
- GPU is being used: `nvidia-smi` or `rocm-smi`
- `gpu_layers` set appropriately (35+ for full offload)
- Using quantized model (GGUF format)

### Steering vectors not working

Ensure:
1. Vectors trained: `ls data/emotion_vectors/`
2. Backend set to `transformers`
3. Path configured: `emotion_vectors_dir`
4. Sufficient GPU memory (~16GB for 8B model)

## Next Steps

1. ✅ **Run examples** - Try `examples/01_basic_inference.py`
2. 📊 **Experiment** - Use `scripts/emotion_repl.py` to explore states
3. 🎯 **Train vectors** - Set up steering for advanced expression
4. 🔧 **Integrate** - Use Elpis in your own projects
5. 🚀 **Deploy** - Run as MCP server with Psyche client

## Resources

- [Full Documentation](https://lemon9247.github.io/Elpis/)
- [Examples](examples/)
- [Issue Tracker](https://github.com/Lemon9247/Elpis/issues)

## Getting Help

- **Documentation**: [https://lemon9247.github.io/Elpis/](https://lemon9247.github.io/Elpis/)
- **Examples**: Check [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/Lemon9247/Elpis/issues)

---

**Ready to start?** Try running your first example:

```bash
python examples/01_basic_inference.py
```

Happy inferring! 🚀
