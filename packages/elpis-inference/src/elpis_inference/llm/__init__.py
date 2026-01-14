"""LLM inference engines."""

from elpis_inference.llm.base import InferenceEngine

# Conditional imports for optional backends
try:
    from elpis_inference.llm.inference import LlamaInference
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

try:
    from elpis_inference.llm.transformers_inference import TransformersInference
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

__all__ = ["InferenceEngine"]

if LLAMA_CPP_AVAILABLE:
    __all__.append("LlamaInference")

if TRANSFORMERS_AVAILABLE:
    __all__.append("TransformersInference")
