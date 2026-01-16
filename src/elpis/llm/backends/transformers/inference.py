"""Async LLM inference using HuggingFace Transformers with steering vector support.

This module provides an inference engine that uses HuggingFace Transformers
and supports emotional modulation via activation steering.
"""

import asyncio
import threading
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional

from loguru import logger

# Runtime imports (for actual usage)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None  # type: ignore
    logger.warning(
        "transformers/torch not installed - TransformersInference unavailable. "
        "Install with: pip install torch transformers"
    )

# Type-checking imports (for annotations only)
if TYPE_CHECKING:
    import torch

from elpis.llm.base import InferenceEngine
from elpis.llm.backends.transformers.config import TransformersConfig
from elpis.llm.backends.transformers.steering import SteeringManager


class TransformersInference(InferenceEngine):
    """Async inference engine using HuggingFace Transformers.

    Supports emotional modulation via steering vectors applied during
    the forward pass, enabling nuanced emotional expression beyond
    sampling parameter adjustment.

    Attributes:
        SUPPORTS_STEERING: Whether this backend supports steering vectors.
        MODULATION_TYPE: The type of emotional modulation this backend uses.
    """

    SUPPORTS_STEERING: bool = True
    MODULATION_TYPE: str = "steering"

    def __init__(self, config: TransformersConfig):
        """Initialize the inference engine.

        Args:
            config: Transformers backend configuration

        Raises:
            RuntimeError: If transformers/torch not installed
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers and torch are required for TransformersInference. "
                "Install with: pip install torch transformers"
            )

        self.config = config
        self.context_length = config.context_length

        # Resolve device
        if config.hardware_backend == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif config.hardware_backend in ("cuda", "rocm"):
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Resolve dtype
        if config.torch_dtype == "auto":
            self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        else:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            self.torch_dtype = dtype_map.get(config.torch_dtype, torch.float32)

        logger.info(
            f"Initializing TransformersInference: device={self.device}, "
            f"dtype={self.torch_dtype}"
        )

        # Load model and tokenizer
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        # Initialize steering manager
        self.steering = SteeringManager(
            device=self.device,
            steering_layer=config.steering_layer,
        )

        # Load emotion vectors if configured
        if config.emotion_vectors_dir:
            self.steering.load_vectors(config.emotion_vectors_dir)
        else:
            logger.info(
                "No emotion_vectors_dir configured - steering unavailable. "
                "Set ELPIS_TRANSFORMERS_EMOTION_VECTORS_DIR to enable."
            )

    def _load_tokenizer(self) -> "AutoTokenizer":
        """Load the tokenizer."""
        logger.info(f"Loading tokenizer from {self.config.path}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.path,
            trust_remote_code=True,
        )
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self) -> "AutoModelForCausalLM":
        """Load the model with appropriate settings."""
        logger.info(f"Loading model from {self.config.path}")

        model = AutoModelForCausalLM.from_pretrained(
            self.config.path,
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True,
        )

        if self.device == "cpu":
            model = model.to(self.device)

        model.eval()
        logger.info("Model loaded successfully")
        return model

    # =========================================================================
    # InferenceEngine Interface Implementation
    # =========================================================================

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        emotion_coefficients: Optional[Dict[str, float]] = None,
    ) -> str:
        """Generate chat completion asynchronously.

        Args:
            messages: List of chat messages with role and content
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            emotion_coefficients: Emotional steering blend weights

        Returns:
            Generated response text
        """
        return await asyncio.to_thread(
            self._chat_completion_sync,
            messages,
            max_tokens,
            temperature,
            top_p,
            emotion_coefficients,
        )

    def _chat_completion_sync(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        emotion_coefficients: Optional[Dict[str, float]],
    ) -> str:
        """Synchronous chat completion implementation."""
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_length,
        ).to(self.device)

        # Apply emotional steering if configured
        steering_vector = None
        if emotion_coefficients:
            steering_vector = self.steering.compute_blended_vector(emotion_coefficients)
            if steering_vector is not None:
                self.steering.apply_hook(self.model, steering_vector)

        try:
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or self.config.max_tokens,
                    temperature=temperature or self.config.temperature,
                    top_p=top_p or self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            return response

        finally:
            # Always clean up the hook
            if steering_vector is not None:
                self.steering.remove_hook()

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        emotion_coefficients: Optional[Dict[str, float]] = None,
    ) -> AsyncIterator[str]:
        """Generate chat completion with streaming.

        Yields tokens as they are generated, enabling real-time display.
        Uses a single internal thread for TextIteratorStreamer (required by
        HuggingFace's streaming API), but avoids the redundant outer threading
        layer that was causing issues.
        """
        for token in self._chat_completion_stream_sync(
            messages, max_tokens, temperature, top_p, emotion_coefficients
        ):
            yield token
            await asyncio.sleep(0)  # Cooperative yield to event loop

    def _chat_completion_stream_sync(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        emotion_coefficients: Optional[Dict[str, float]],
    ) -> Iterator[str]:
        """Synchronous streaming implementation using TextIteratorStreamer."""
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_length,
        ).to(self.device)

        # Set up streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Apply emotional steering if configured
        steering_vector = None
        if emotion_coefficients:
            steering_vector = self.steering.compute_blended_vector(emotion_coefficients)
            if steering_vector is not None:
                self.steering.apply_hook(self.model, steering_vector)

        # Generation happens in a thread, streamer yields tokens
        def generate():
            try:
                with torch.no_grad():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens or self.config.max_tokens,
                        temperature=temperature or self.config.temperature,
                        top_p=top_p or self.config.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        streamer=streamer,
                    )
            finally:
                if steering_vector is not None:
                    self.steering.remove_hook()

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()

        for token in streamer:
            yield token

        thread.join(timeout=1.0)

    async def function_call(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        emotion_coefficients: Optional[Dict[str, float]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate function calls.

        Note: This is a simplified implementation. For robust tool use,
        consider using a fine-tuned model or more sophisticated prompting.

        Args:
            messages: Chat messages
            tools: Available tools in OpenAI format
            temperature: Sampling temperature
            emotion_coefficients: Emotional steering weights

        Returns:
            List of tool calls, or None if none generated
        """
        return await asyncio.to_thread(
            self._function_call_sync, messages, tools, temperature, emotion_coefficients
        )

    def _function_call_sync(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: Optional[float],
        emotion_coefficients: Optional[Dict[str, float]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Synchronous function call generation.

        This is a basic implementation - models with native tool support
        would use their built-in tool calling format.
        """
        # For now, append tool descriptions to the system message
        # and parse the response. A more robust implementation would
        # use the model's native tool calling format.

        tool_desc = self._format_tools_for_prompt(tools)

        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[0]["role"] == "system":
            enhanced_messages[0]["content"] += f"\n\nAvailable tools:\n{tool_desc}"
        else:
            enhanced_messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"You have access to the following tools:\n{tool_desc}",
                },
            )

        response = self._chat_completion_sync(
            enhanced_messages,
            max_tokens=1024,
            temperature=temperature,
            top_p=0.9,
            emotion_coefficients=emotion_coefficients,
        )

        # Parse tool calls from response (this is model-dependent)
        return self._parse_tool_calls(response)

    def _format_tools_for_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools for inclusion in prompt."""
        lines = []
        for tool in tools:
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            lines.append(f"- {name}: {desc}")
            if params.get("properties"):
                for param_name, param_info in params["properties"].items():
                    param_desc = param_info.get("description", "")
                    lines.append(f"    - {param_name}: {param_desc}")
        return "\n".join(lines)

    def _parse_tool_calls(self, response: str) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls from model response.

        This is a placeholder - implement based on your tool call format.
        For Llama 3.1, tool calls typically appear in a specific format.
        """
        # TODO: Implement parsing based on expected format
        return None

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "steering"):
            self.steering.remove_hook()
