"""Abstract base class for LLM inference engines."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional


class InferenceEngine(ABC):
    """
    Abstract base class for LLM inference engines.

    Defines the interface that all inference implementations must provide.
    Supports both sampling parameter modulation and steering vector coefficients.
    """

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        emotion_coefficients: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Generate chat completion asynchronously.

        Args:
            messages: List of chat messages with role and content
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            emotion_coefficients: Optional steering vector blend weights
                {
                    "excited": float,     # high valence, high arousal
                    "frustrated": float,  # low valence, high arousal
                    "calm": float,        # high valence, low arousal
                    "depleted": float     # low valence, low arousal
                }

        Returns:
            Generated response text
        """
        pass

    @abstractmethod
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        emotion_coefficients: Optional[Dict[str, float]] = None,
    ) -> AsyncIterator[str]:
        """
        Generate chat completion with streaming.

        Args:
            messages: List of chat messages with role and content
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            emotion_coefficients: Optional steering vector blend weights

        Yields:
            Tokens as they are generated
        """
        pass

    @abstractmethod
    async def function_call(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        emotion_coefficients: Optional[Dict[str, float]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate function/tool calls.

        Args:
            messages: List of chat messages with role and content
            tools: List of tool definitions
            temperature: Sampling temperature
            emotion_coefficients: Optional steering vector blend weights

        Returns:
            List of tool calls, or None if no tools should be called
        """
        pass
