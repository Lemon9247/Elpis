"""Elpis Inference - MCP server with emotional regulation."""

__version__ = "0.1.0"

from elpis_inference.emotion.state import EmotionalState
from elpis_inference.emotion.regulation import HomeostasisRegulator

__all__ = ["EmotionalState", "HomeostasisRegulator"]
