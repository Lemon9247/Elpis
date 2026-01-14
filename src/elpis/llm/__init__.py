"""LLM inference and prompt management for Elpis - backward compatibility wrapper."""

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

# Legacy imports
try:
    from elpis.llm.prompts import build_system_prompt
    HAS_PROMPTS = True
except ImportError:
    HAS_PROMPTS = False

__all__ = ["InferenceEngine"]

if LLAMA_CPP_AVAILABLE:
    __all__.append("LlamaInference")

if TRANSFORMERS_AVAILABLE:
    __all__.append("TransformersInference")

if HAS_PROMPTS:
    __all__.append("build_system_prompt")
