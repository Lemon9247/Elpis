"""Utility functions and helpers for Elpis."""

from elpis.utils.exceptions import (
    ConfigurationError,
    ElpisError,
    HardwareDetectionError,
    LLMInferenceError,
    ToolExecutionError,
)
from elpis.utils.hardware import HardwareBackend, detect_hardware

__all__ = [
    "HardwareBackend",
    "detect_hardware",
    "ElpisError",
    "ConfigurationError",
    "HardwareDetectionError",
    "LLMInferenceError",
    "ToolExecutionError",
]
