"""Backward compatibility shim for TransformersInference.

DEPRECATED: This module is maintained for backward compatibility only.
New code should import from elpis.llm.backends.transformers instead:

    # Old (deprecated):
    from elpis.llm.transformers_inference import TransformersInference

    # New (preferred):
    from elpis.llm.backends.transformers import TransformersInference

    # Or use the factory (recommended):
    from elpis.llm import create_backend
    settings.backend = "transformers"
    engine = create_backend(settings)
"""

import warnings

# Re-export from new location
from elpis.llm.backends.transformers import (
    TransformersInference,
    TransformersConfig,
    SteeringManager,
)

# Emit deprecation warning on import
warnings.warn(
    "Importing from elpis.llm.transformers_inference is deprecated. "
    "Use elpis.llm.backends.transformers or elpis.llm.create_backend() instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TransformersInference", "TransformersConfig", "SteeringManager"]
