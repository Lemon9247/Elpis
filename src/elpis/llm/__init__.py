"""LLM inference engines.

This module provides the public API for LLM inference, including:
- InferenceEngine: Abstract base class for all backends
- create_backend: Factory function to create backend instances
- get_available_backends: Query available backends and their status

For backward compatibility, individual backend classes can also be
imported directly, though this usage is deprecated:

    # Deprecated (still works but will emit warning):
    from elpis.llm.inference import LlamaInference

    # Preferred:
    from elpis.llm import create_backend
    engine = create_backend(settings)
"""

from elpis.llm.base import InferenceEngine
from elpis.llm.backends import (
    create_backend,
    get_available_backends,
    is_backend_available,
    register_backend,
)

# Conditional imports for optional backends (backward compatibility)
try:
    from elpis.llm.inference import LlamaInference
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

try:
    from elpis.llm.transformers_inference import TransformersInference
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

__all__ = [
    # Core API
    "InferenceEngine",
    "create_backend",
    "get_available_backends",
    "is_backend_available",
    "register_backend",
    # Availability flags
    "LLAMA_CPP_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
]

# Add backend classes to __all__ only if available
if LLAMA_CPP_AVAILABLE:
    __all__.append("LlamaInference")

if TRANSFORMERS_AVAILABLE:
    __all__.append("TransformersInference")
