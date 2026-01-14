# Elpis Examples

This directory contains example scripts demonstrating various features of Elpis.

## Prerequisites

```bash
# Install Elpis
cd /path/to/Elpis
pip install -e .

# For steering vector examples, also install:
pip install torch transformers
```

## Examples

### 01. Basic Inference (`01_basic_inference.py`)

Demonstrates basic LLM inference using the llama-cpp backend with GGUF models.

**Requirements:**
- GGUF model downloaded to `./data/models/`
- `llama-cpp-python` installed

**Run:**
```bash
python examples/01_basic_inference.py
```

**What it shows:**
- Loading a GGUF model
- Simple chat completion
- Basic error handling

---

### 02. Emotional Modulation (`02_emotional_modulation.py`)

Shows how emotional state affects sampling parameters (temperature, top_p) in responses.

**Requirements:**
- GGUF model downloaded
- `llama-cpp-python` installed

**Run:**
```bash
python examples/02_emotional_modulation.py
```

**What it shows:**
- Creating different emotional states
- How emotions map to sampling parameters
- Comparing responses across emotional states
- Valence-arousal quadrant system

**Key concepts:**
- High arousal → Lower temperature (more focused)
- High valence → Higher top_p (broader sampling)
- Four quadrants: excited, frustrated, calm, depleted

---

### 03. Steering Vectors (`03_steering_vectors.py`)

Demonstrates advanced emotional modulation using activation steering with the transformers backend.

**Requirements:**
- `torch` and `transformers` installed
- Emotion vectors trained (see below)
- ~16GB GPU memory (or CPU with patience)

**Setup:**
```bash
# Install dependencies
pip install torch transformers

# Train emotion vectors (takes 5-10 minutes on GPU)
python scripts/train_emotion_vectors.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --layer 15 \
  --output ./data/emotion_vectors
```

**Run:**
```bash
python examples/03_steering_vectors.py
```

**What it shows:**
- Using the transformers backend
- Loading trained emotion steering vectors
- Applying activation steering during generation
- Bilinear blending of emotion vectors
- More nuanced emotional expression than parameter tuning

**Key concepts:**
- Steering vectors modify internal activations
- Four base emotions blended by coefficients
- Different from sampling parameter modulation
- More computationally expensive but more expressive

---

### 04. MCP Server Usage (`04_mcp_server_usage.py`)

Demonstrates interaction patterns with Elpis when run as an MCP server.

**Requirements:**
- None (this is a simulation/demonstration)

**Run:**
```bash
python examples/04_mcp_server_usage.py
```

**What it shows:**
- MCP tool interface (generate, update_emotion, etc.)
- Emotional event management
- Streaming generation
- State management

**Note:** This is a demonstration of the API. For actual MCP usage:
```bash
# Terminal 1: Start the server
elpis-server

# Terminal 2: Connect with Psyche client
psyche
```

---

## Quick Reference

### Emotional States

The valence-arousal model maps to four quadrants:

```
         High Arousal
              +1
               |
  Frustrated   |   Excited
  (-1, +1)     |   (+1, +1)
               |
-1 ----------------------------- +1
        Valence
               |
  Depleted     |   Calm
  (-1, -1)     |   (+1, -1)
               |
              -1
         Low Arousal
```

### Backend Comparison

| Feature | llama-cpp | transformers |
|---------|-----------|--------------|
| Model format | GGUF (quantized) | HuggingFace |
| Speed | Fast | Slower |
| Memory | Low (~6-8GB) | High (~16GB+) |
| Emotional modulation | Sampling params | Steering vectors |
| Setup | Simple | Requires training |
| Expression quality | Good | Excellent |

### Emotional Events

Available event types for `update_emotion`:
- `success` - Positive outcome (+valence, +arousal)
- `failure` - Negative outcome (-valence, +arousal)
- `novelty` - New discovery (+valence, +arousal)
- `insight` - Understanding gained (+valence, -arousal)
- `frustration` - Blocked progress (-valence, +arousal)
- `blocked` - Can't proceed (-valence, +arousal)
- `test_passed` - Test success (+valence, slight +arousal)
- `test_failed` - Test failure (-valence, +arousal)
- `error` - Unexpected error (-valence, +arousal)
- `idle` - Nothing happening (decay toward baseline)

## Troubleshooting

### "No module named pytest"

The examples don't require pytest. This message indicates pytest isn't installed for tests, not for examples.

### "Model not found"

Download a model first:
```bash
mkdir -p data/models
# Download from HuggingFace
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF \
  llama-2-7b-chat.Q5_K_M.gguf \
  --local-dir data/models
```

### "transformers not installed"

For steering vector examples:
```bash
pip install torch transformers
```

### "Emotion vectors not found"

Train them first:
```bash
python scripts/train_emotion_vectors.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output ./data/emotion_vectors
```

### Out of memory

For transformers backend:
- Use smaller model (7B instead of 70B)
- Use quantization: `torch_dtype="float16"`
- Reduce `context_length`
- Use CPU (slower): `hardware_backend="cpu"`

## Next Steps

1. **Try the examples** in order (01 → 02 → 03)
2. **Experiment** with different emotional states
3. **Train custom vectors** with your own prompt examples
4. **Integrate with Psyche** for full interactive experience
5. **Build your own** MCP client using these patterns

## Related Documentation

- [Main README](../README.md) - Project overview
- [Training Guide](../README.md#training-emotion-vectors) - Emotion vector training
- [MCP Tools](../README.md#mcp-tools) - Server tool reference
- [Developer Scripts](../scripts/) - Debugging utilities
