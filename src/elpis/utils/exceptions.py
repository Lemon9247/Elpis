"""Custom exceptions for Elpis agent."""


class ElpisError(Exception):
    """Base exception for all Elpis errors."""

    pass


class ConfigurationError(ElpisError):
    """Configuration loading or validation error."""

    pass


class HardwareDetectionError(ElpisError):
    """Hardware detection failed."""

    pass


class LLMInferenceError(ElpisError):
    """LLM inference failed."""

    pass


class ModelLoadError(LLMInferenceError):
    """Failed to load LLM model."""

    pass


class ToolExecutionError(ElpisError):
    """Tool execution failed."""

    pass


class PathSafetyError(ToolExecutionError):
    """Path is outside workspace or contains unsafe characters."""

    pass


class CommandSafetyError(ToolExecutionError):
    """Command contains dangerous operations."""

    pass
