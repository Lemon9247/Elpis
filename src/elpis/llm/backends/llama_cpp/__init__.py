"""llama-cpp-python backend for GGUF model inference.

This backend provides:
- GGUF quantized model support (4-bit, 5-bit, 8-bit quantization)
- CPU/CUDA/ROCm hardware acceleration via llama-cpp-python
- Emotional modulation via sampling parameter adjustment (temperature, top_p)

Note: This backend does NOT support steering vectors. For activation-level
emotional modulation, use the transformers backend instead.
"""

from elpis.llm.backends.llama_cpp.config import LlamaCppConfig
from elpis.llm.backends.llama_cpp.inference import LlamaInference

__all__ = ["LlamaInference", "LlamaCppConfig"]
