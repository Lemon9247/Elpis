"""Async LLM inference wrapper using llama-cpp-python."""

import asyncio
import queue
import threading
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from llama_cpp import Llama
from loguru import logger

from elpis.config.settings import ModelSettings
from elpis.llm.base import InferenceEngine
from elpis.utils.exceptions import LLMInferenceError, ModelLoadError
from elpis.utils.hardware import HardwareBackend, detect_hardware


class LlamaInference(InferenceEngine):
    """Async wrapper around llama-cpp-python for LLM inference."""

    def __init__(self, settings: ModelSettings):
        """Initialize LLM with settings and load model."""
        self.settings = settings
        self.backend = self._detect_backend()
        logger.info(f"Detected hardware backend: {self.backend.value}")
        self.model = self._load_model()

    def _detect_backend(self) -> HardwareBackend:
        """Detect or use configured hardware backend."""
        if self.settings.hardware_backend == "auto":
            return detect_hardware()
        return HardwareBackend(self.settings.hardware_backend)

    def _load_model(self) -> Llama:
        """Load GGUF model with appropriate backend."""
        try:
            logger.info(f"Loading model from: {self.settings.path}")
            model = Llama(
                model_path=self.settings.path,
                n_ctx=self.settings.context_length,
                n_gpu_layers=self.settings.gpu_layers,
                n_threads=self.settings.n_threads,
                chat_format="llama-3",
                verbose=False,
            )
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Failed to load model: {e}") from e

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
            emotion_coefficients: Steering vector coefficients (ignored by llama-cpp backend)

        Returns:
            Generated response text
        """
        if emotion_coefficients:
            logger.debug(
                "Emotion coefficients provided but ignored by llama-cpp backend. "
                "Use transformers backend for steering vector support."
            )
        return await asyncio.to_thread(
            self._chat_completion_sync, messages, max_tokens, temperature, top_p
        )

    def _chat_completion_sync(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> str:
        """Synchronous chat completion implementation."""
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens or self.settings.max_tokens,
                temperature=temperature or self.settings.temperature,
                top_p=top_p or self.settings.top_p,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise LLMInferenceError(f"Inference failed: {e}") from e

    async def function_call(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        emotion_coefficients: Optional[Dict[str, float]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate function calls using OpenAI-compatible format.

        Args:
            messages: List of chat messages
            tools: Available tools/functions
            temperature: Sampling temperature
            emotion_coefficients: Steering vector coefficients (ignored by llama-cpp backend)

        Returns:
            List of tool calls or None if no tools should be called
        """
        if emotion_coefficients:
            logger.debug("Emotion coefficients ignored by llama-cpp backend")
        return await asyncio.to_thread(self._function_call_sync, messages, tools, temperature)

    def _function_call_sync(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: Optional[float],
    ) -> Optional[List[Dict[str, Any]]]:
        """Synchronous function call generation."""
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature or self.settings.temperature,
            )

            message = response["choices"][0]["message"]
            return message.get("tool_calls")
        except Exception as e:
            logger.error(f"Function call error: {e}")
            raise LLMInferenceError(f"Function call failed: {e}") from e

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

        Yields tokens as they are generated, enabling real-time display.

        Args:
            messages: List of chat messages with role and content
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            emotion_coefficients: Steering vector coefficients (ignored by llama-cpp backend)

        Yields:
            Individual tokens as they are generated
        """
        if emotion_coefficients:
            logger.debug("Emotion coefficients ignored by llama-cpp backend")
        async for token in self._stream_in_thread(messages, max_tokens, temperature, top_p):
            yield token

    def _chat_completion_stream_sync(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> Iterator[str]:
        """
        Synchronous streaming chat completion using llama-cpp-python's stream mode.

        Yields tokens as they are generated by the model.
        """
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens or self.settings.max_tokens,
                temperature=temperature or self.settings.temperature,
                top_p=top_p or self.settings.top_p,
                stream=True,
            )
            for chunk in response:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
        except Exception as e:
            logger.error(f"Streaming inference error: {e}")
            raise LLMInferenceError(f"Streaming inference failed: {e}") from e

    async def _stream_in_thread(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> AsyncIterator[str]:
        """
        Bridge synchronous streaming to async using a queue.

        Runs the sync generator in a thread and yields tokens asynchronously.
        """
        token_queue: queue.Queue[Optional[str]] = queue.Queue()
        error_holder: List[Exception] = []

        def producer() -> None:
            """Thread function that generates tokens and puts them in queue."""
            try:
                for token in self._chat_completion_stream_sync(
                    messages, max_tokens, temperature, top_p
                ):
                    token_queue.put(token)
            except Exception as e:
                error_holder.append(e)
            finally:
                token_queue.put(None)  # Sentinel to signal completion

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        try:
            while True:
                # Use asyncio-friendly polling to avoid blocking event loop
                while token_queue.empty():
                    await asyncio.sleep(0.01)

                token = token_queue.get_nowait()
                if token is None:
                    break
                yield token
        finally:
            thread.join(timeout=1.0)

        # Re-raise any error from the producer thread
        if error_holder:
            raise error_holder[0]
