"""HuggingFace Transformers backend with steering vector support.

This backend provides emotional modulation through steering vector injection
at the activation level, enabling nuanced emotional expression during inference.

Requires: torch, transformers
"""

try:
    from elpis.llm.backends.transformers.config import TransformersConfig
    from elpis.llm.backends.transformers.inference import TransformersInference
    from elpis.llm.backends.transformers.steering import SteeringManager

    AVAILABLE = True
except ImportError:
    AVAILABLE = False

__all__ = ["TransformersInference", "TransformersConfig", "SteeringManager", "AVAILABLE"]
