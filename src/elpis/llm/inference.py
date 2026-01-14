"""Backward compatibility shim for LlamaInference.

DEPRECATED: This module is maintained for backward compatibility only.
New code should import from elpis.llm.backends.llama_cpp instead:

    # Old (deprecated):
    from elpis.llm.inference import LlamaInference

    # New (preferred):
    from elpis.llm.backends.llama_cpp import LlamaInference

    # Or use the factory (recommended):
    from elpis.llm import create_backend
    engine = create_backend(settings)
"""

import warnings

# Re-export from new location
from elpis.llm.backends.llama_cpp import LlamaInference, LlamaCppConfig

# Emit deprecation warning on import
warnings.warn(
    "Importing from elpis.llm.inference is deprecated. "
    "Use elpis.llm.backends.llama_cpp or elpis.llm.create_backend() instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["LlamaInference", "LlamaCppConfig"]
