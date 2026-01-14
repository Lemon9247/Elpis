"""LlamaInference - backward compatibility wrapper.

This module re-exports from elpis_inference.llm.inference for backward compatibility.
All new code should import directly from elpis_inference.
"""

try:
    from elpis_inference.llm.inference import *  # noqa: F403, F401
except ImportError:
    # llama-cpp-python not installed - LlamaInference unavailable
    pass
