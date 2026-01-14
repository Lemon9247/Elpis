"""LLM backend registry and factory.

This module provides a plugin-style architecture for LLM backends,
enabling runtime discovery and instantiation of available backends.
"""

from typing import TYPE_CHECKING, Callable, Dict, Tuple

from loguru import logger

from elpis.llm.base import InferenceEngine

if TYPE_CHECKING:
    from elpis.config.settings import ModelSettings


# Registry of available backends
# Maps backend name -> (loader_func, is_available)
_BACKENDS: Dict[str, Tuple[Callable[["ModelSettings"], InferenceEngine], bool]] = {}


def register_backend(
    name: str,
    loader: Callable[["ModelSettings"], InferenceEngine],
    available: bool = True,
) -> None:
    """Register a backend with the registry.

    Args:
        name: Backend identifier (e.g., "llama-cpp", "transformers")
        loader: Factory function that takes ModelSettings and returns InferenceEngine
        available: Whether the backend's dependencies are installed
    """
    _BACKENDS[name] = (loader, available)
    logger.debug(f"Registered backend '{name}' (available={available})")


def get_available_backends() -> Dict[str, bool]:
    """Get mapping of backend names to their availability status.

    Returns:
        Dict mapping backend name to whether it's available
    """
    return {name: avail for name, (_, avail) in _BACKENDS.items()}


def is_backend_available(name: str) -> bool:
    """Check if a specific backend is available.

    Args:
        name: Backend identifier

    Returns:
        True if backend is registered and available
    """
    if name not in _BACKENDS:
        return False
    _, available = _BACKENDS[name]
    return available


def create_backend(settings: "ModelSettings") -> InferenceEngine:
    """Create an inference engine based on settings.

    Args:
        settings: Model settings including backend choice

    Returns:
        Configured InferenceEngine instance

    Raises:
        ValueError: If backend is unknown or unavailable
    """
    backend_name = settings.backend

    if backend_name not in _BACKENDS:
        available = list(get_available_backends().keys())
        raise ValueError(
            f"Unknown backend '{backend_name}'. "
            f"Available backends: {available}"
        )

    loader, available = _BACKENDS[backend_name]

    if not available:
        raise ValueError(
            f"Backend '{backend_name}' is not available. "
            f"Required dependencies may not be installed."
        )

    logger.info(f"Creating backend: {backend_name}")
    return loader(settings)


# =============================================================================
# Backend Registration
# =============================================================================

# Register llama-cpp backend
def _load_llama_cpp(settings: "ModelSettings") -> InferenceEngine:
    """Load llama-cpp backend with converted config."""
    from elpis.llm.backends.llama_cpp import LlamaInference
    config = settings.to_llama_cpp_config()
    return LlamaInference(config)


try:
    from elpis.llm.backends.llama_cpp import LlamaInference  # noqa: F401
    register_backend("llama-cpp", _load_llama_cpp, available=True)
except ImportError:
    register_backend("llama-cpp", _load_llama_cpp, available=False)
    logger.debug("llama-cpp backend unavailable (llama-cpp-python not installed)")


# Register transformers backend
def _load_transformers(settings: "ModelSettings") -> InferenceEngine:
    """Load transformers backend with converted config."""
    from elpis.llm.backends.transformers import TransformersInference
    config = settings.to_transformers_config()
    return TransformersInference(config)


try:
    from elpis.llm.backends.transformers import TransformersInference  # noqa: F401
    register_backend("transformers", _load_transformers, available=True)
except ImportError:
    register_backend("transformers", _load_transformers, available=False)
    logger.debug("transformers backend unavailable (torch/transformers not installed)")


__all__ = [
    "register_backend",
    "get_available_backends",
    "is_backend_available",
    "create_backend",
]
