"""Async LLM inference using HuggingFace Transformers with steering vector support.

This module provides an alternative to LlamaInference that uses HuggingFace
Transformers and supports emotional modulation via activation steering.
"""

import asyncio
import queue
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from loguru import logger

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers/torch not installed - TransformersInference unavailable. "
        "Install with: pip install torch transformers"
    )

from elpis_inference.config.settings import ModelSettings
from elpis_inference.llm.base import InferenceEngine


class TransformersInference(InferenceEngine):
    """
    Async inference engine using HuggingFace Transformers.

    Supports emotional modulation via steering vectors applied during
    the forward pass, enabling nuanced emotional expression beyond
    sampling parameter adjustment.
    """

    def __init__(self, settings: ModelSettings):
        """
        Initialize the inference engine.

        Args:
            settings: Model configuration settings

        Raises:
            RuntimeError: If transformers/torch not installed
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers and torch are required for TransformersInference. "
                "Install with: pip install torch transformers"
            )

        self.settings = settings
        self.context_length = settings.context_length

        # Resolve device
        if settings.hardware_backend == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif settings.hardware_backend in ("cuda", "rocm"):
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Resolve dtype
        if settings.torch_dtype == "auto":
            self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        else:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            self.torch_dtype = dtype_map.get(settings.torch_dtype, torch.float32)

        logger.info(
            f"Initializing TransformersInference: device={self.device}, "
            f"dtype={self.torch_dtype}"
        )

        # Load model and tokenizer
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        # Emotion vectors (load if configured)
        self.emotion_vectors: Dict[str, torch.Tensor] = {}
        if settings.emotion_vectors_dir:
            self._load_emotion_vectors(settings.emotion_vectors_dir)
        else:
            logger.info(
                "No emotion_vectors_dir configured - steering unavailable. "
                "Set ELPIS_MODEL__EMOTION_VECTORS_DIR to enable."
            )

        # Hook handle for cleanup
        self._steering_hook_handle = None

    def _load_tokenizer(self) -> "AutoTokenizer":
        """Load the tokenizer."""
        logger.info(f"Loading tokenizer from {self.settings.path}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.settings.path,
            trust_remote_code=True,
        )
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self) -> "AutoModelForCausalLM":
        """Load the model with appropriate settings."""
        logger.info(f"Loading model from {self.settings.path}")

        model = AutoModelForCausalLM.from_pretrained(
            self.settings.path,
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True,
        )

        if self.device == "cpu":
            model = model.to(self.device)

        model.eval()
        logger.info("Model loaded successfully")
        return model

    def _load_emotion_vectors(self, directory: str) -> None:
        """
        Load pre-trained emotion steering vectors from disk.

        Args:
            directory: Path to directory containing .pt vector files
        """
        path = Path(directory)
        if not path.exists():
            logger.warning(f"Emotion vectors directory not found: {path}")
            return

        for vector_file in path.glob("*.pt"):
            name = vector_file.stem
            try:
                vector = torch.load(vector_file, map_location=self.device, weights_only=True)
                self.emotion_vectors[name] = vector
                logger.info(f"Loaded emotion vector: {name} (shape={vector.shape})")
            except Exception as e:
                logger.error(f"Failed to load {vector_file}: {e}")

        if self.emotion_vectors:
            logger.info(f"Loaded {len(self.emotion_vectors)} emotion vectors")
        else:
            logger.warning(f"No emotion vectors found in {path}")

    def _compute_blended_steering(
        self,
        emotion_coefficients: Dict[str, float],
    ) -> Optional[torch.Tensor]:
        """
        Blend emotion vectors based on current emotional state.

        Args:
            emotion_coefficients: Mapping of emotion name to blend weight

        Returns:
            Blended steering vector, or None if no vectors available
        """
        if not self.emotion_vectors:
            return None

        # Global strength from emotion settings (passed via coefficients sum)
        blended = None
        total_weight = 0.0

        for emotion_name, coef in emotion_coefficients.items():
            if emotion_name not in self.emotion_vectors:
                continue

            vector = self.emotion_vectors[emotion_name]
            weight = coef

            if weight < 0.01:  # Skip negligible contributions
                continue

            if blended is None:
                blended = vector * weight
            else:
                blended = blended + (vector * weight)

            total_weight += weight

        if blended is not None and total_weight > 0:
            logger.debug(
                f"Blended steering: total_weight={total_weight:.3f}, "
                f"norm={blended.norm().item():.3f}"
            )

        return blended

    def _apply_steering_hook(self, steering_vector: torch.Tensor) -> None:
        """
        Apply a forward hook to inject steering vector at target layer.

        Args:
            steering_vector: The steering vector to inject
        """
        layer_idx = self.settings.steering_layer

        # Access the layer (Llama-style architecture assumed)
        try:
            if hasattr(self.model, "model"):
                # Model has .model attribute (common in HF models)
                layers = self.model.model.layers
            else:
                # Direct access
                layers = self.model.layers

            target_layer = layers[layer_idx]
        except (AttributeError, IndexError) as e:
            logger.error(f"Cannot access layer {layer_idx}: {e}")
            return

        def hook_fn(module, input, output):
            """Add steering vector to the output activations."""
            # output is typically a tuple (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Add steering vector (broadcast across batch and sequence)
            # steering_vector shape: (hidden_dim,)
            # hidden_states shape: (batch, seq_len, hidden_dim)
            steered = hidden_states + steering_vector.to(hidden_states.dtype).to(hidden_states.device)

            if isinstance(output, tuple):
                return (steered,) + output[1:]
            else:
                return steered

        self._steering_hook_handle = target_layer.register_forward_hook(hook_fn)
        logger.debug(f"Steering hook registered on layer {layer_idx}")

    def _remove_steering_hook(self) -> None:
        """Remove the steering hook if active."""
        if self._steering_hook_handle is not None:
            self._steering_hook_handle.remove()
            self._steering_hook_handle = None
            logger.debug("Steering hook removed")

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
        """
        Generate chat completion asynchronously.

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
            steering_vector = self._compute_blended_steering(emotion_coefficients)
            if steering_vector is not None:
                self._apply_steering_hook(steering_vector)

        try:
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or self.settings.max_tokens,
                    temperature=temperature or self.settings.temperature,
                    top_p=top_p or self.settings.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            return response

        finally:
            # Always clean up the hook
            if steering_vector is not None:
                self._remove_steering_hook()

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
        """
        async for token in self._stream_in_thread(
            messages, max_tokens, temperature, top_p, emotion_coefficients
        ):
            yield token

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
            steering_vector = self._compute_blended_steering(emotion_coefficients)
            if steering_vector is not None:
                self._apply_steering_hook(steering_vector)

        # Generation happens in a thread, streamer yields tokens
        def generate():
            try:
                with torch.no_grad():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens or self.settings.max_tokens,
                        temperature=temperature or self.settings.temperature,
                        top_p=top_p or self.settings.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        streamer=streamer,
                    )
            finally:
                if steering_vector is not None:
                    self._remove_steering_hook()

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()

        for token in streamer:
            yield token

        thread.join(timeout=1.0)

    async def _stream_in_thread(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        emotion_coefficients: Optional[Dict[str, float]],
    ) -> AsyncIterator[str]:
        """Bridge synchronous streaming to async using a queue."""
        token_queue: queue.Queue[Optional[str]] = queue.Queue()
        error_holder: List[Exception] = []

        def producer() -> None:
            try:
                for token in self._chat_completion_stream_sync(
                    messages, max_tokens, temperature, top_p, emotion_coefficients
                ):
                    token_queue.put(token)
            except Exception as e:
                error_holder.append(e)
            finally:
                token_queue.put(None)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        try:
            while True:
                while token_queue.empty():
                    await asyncio.sleep(0.01)

                token = token_queue.get_nowait()
                if token is None:
                    break
                yield token
        finally:
            thread.join(timeout=1.0)

        if error_holder:
            raise error_holder[0]

    async def function_call(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        emotion_coefficients: Optional[Dict[str, float]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate function calls.

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
        """
        Synchronous function call generation.

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
        """
        Parse tool calls from model response.

        This is a placeholder - implement based on your tool call format.
        For Llama 3.1, tool calls typically appear in a specific format.
        """
        # TODO: Implement parsing based on expected format
        return None

    def __del__(self):
        """Cleanup on deletion."""
        self._remove_steering_hook()
