"""Async LLM inference wrapper using HuggingFace Transformers with steering vectors.

Drop-in replacement for LlamaInference that enables emotional modulation
via activation steering rather than sampling parameter adjustment.
"""

import asyncio
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Optional: steering-vectors library for emotional modulation
try:
    from steering_vectors import SteeringVector, train_steering_vector
    STEERING_AVAILABLE = True
except ImportError:
    STEERING_AVAILABLE = False
    logger.warning("steering-vectors not installed - emotional steering disabled")


@dataclass
class EmotionalSteeringConfig:
    """Configuration for emotional steering vectors."""

    # Layer to apply steering (typically middle-to-late layers work best)
    # For Llama-3.1-8B, layers 12-20 are usually good candidates
    steering_layer: int = 15

    # Global multiplier for steering strength (tune this!)
    steering_strength: float = 1.0

    # Per-emotion strength multipliers if you want asymmetric effects
    emotion_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.emotion_weights is None:
            self.emotion_weights = {
                "excited": 1.0,
                "frustrated": 1.0,
                "calm": 1.0,
                "depleted": 1.0,
            }


class TransformersInference:
    """
    Async wrapper around HuggingFace Transformers for LLM inference.

    Supports emotional modulation via steering vectors applied during
    the forward pass, enabling more nuanced emotional expression than
    sampling parameter adjustment alone.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        context_length: int = 8192,
        steering_config: Optional[EmotionalSteeringConfig] = None,
    ):
        """
        Initialize the inference engine.

        Args:
            model_name_or_path: HuggingFace model ID or local path
            device: Device to load model on ("auto", "cuda", "cpu")
            torch_dtype: Data type ("auto", "float16", "bfloat16", "float32")
            context_length: Maximum context length
            steering_config: Configuration for emotional steering
        """
        self.model_name_or_path = model_name_or_path
        self.context_length = context_length
        self.steering_config = steering_config or EmotionalSteeringConfig()

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Resolve dtype
        if torch_dtype == "auto":
            if self.device == "cuda":
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float32
        else:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            self.torch_dtype = dtype_map.get(torch_dtype, torch.float32)

        logger.info(f"Loading model on {self.device} with dtype {self.torch_dtype}")

        # Load model and tokenizer
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        # Emotional steering vectors (populated by train_emotion_vectors or load)
        self.emotion_vectors: Dict[str, "SteeringVector"] = {}
        self._steering_hook = None

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer."""
        logger.info(f"Loading tokenizer from {self.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self) -> AutoModelForCausalLM:
        """Load the model with appropriate settings."""
        logger.info(f"Loading model from {self.model_name_or_path}")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
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
    # Emotional Steering Vector Management
    # =========================================================================

    def train_emotion_vectors(
        self,
        emotion_prompts: Dict[str, tuple[List[str], List[str]]],
    ) -> None:
        """
        Train steering vectors for each emotion from contrastive prompt pairs.

        Args:
            emotion_prompts: Dict mapping emotion name to (positive_prompts, negative_prompts)
                Example:
                {
                    "excited": (
                        ["I feel energized and enthusiastic!", "This is amazing!"],
                        ["I feel tired and bored.", "This is dull."],
                    ),
                    "calm": (
                        ["I feel peaceful and relaxed.", "Everything is fine."],
                        ["I feel anxious and stressed!", "Everything is wrong!"],
                    ),
                    ...
                }
        """
        if not STEERING_AVAILABLE:
            logger.warning("steering-vectors not available, skipping training")
            return

        logger.info("Training emotional steering vectors...")

        for emotion_name, (positive, negative) in emotion_prompts.items():
            logger.info(f"  Training vector for: {emotion_name}")

            vector = train_steering_vector(
                model=self.model,
                tokenizer=self.tokenizer,
                positive_examples=positive,
                negative_examples=negative,
                layers=[self.steering_config.steering_layer],
            )

            self.emotion_vectors[emotion_name] = vector
            logger.info(f"  ✓ {emotion_name} vector trained")

        logger.info(f"Trained {len(self.emotion_vectors)} emotion vectors")

    def save_emotion_vectors(self, directory: str) -> None:
        """Save trained emotion vectors to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        for name, vector in self.emotion_vectors.items():
            vector_path = path / f"{name}.pt"
            torch.save(vector, vector_path)
            logger.info(f"Saved {name} vector to {vector_path}")

    def load_emotion_vectors(self, directory: str) -> None:
        """Load pre-trained emotion vectors from disk."""
        path = Path(directory)

        for vector_file in path.glob("*.pt"):
            name = vector_file.stem
            vector = torch.load(vector_file, map_location=self.device)
            self.emotion_vectors[name] = vector
            logger.info(f"Loaded {name} vector from {vector_file}")

    def _compute_blended_steering(
        self,
        emotion_coefficients: Dict[str, float],
    ) -> Optional["SteeringVector"]:
        """
        Blend emotion vectors based on current emotional state.

        Args:
            emotion_coefficients: Mapping of emotion name to blend weight (0-1)

        Returns:
            Blended steering vector, or None if no vectors available
        """
        if not self.emotion_vectors or not STEERING_AVAILABLE:
            return None

        # Weight each vector by its coefficient and the global strength
        strength = self.steering_config.steering_strength
        weights = self.steering_config.emotion_weights

        blended = None
        for emotion_name, coef in emotion_coefficients.items():
            if emotion_name not in self.emotion_vectors:
                continue

            vector = self.emotion_vectors[emotion_name]
            weight = coef * strength * weights.get(emotion_name, 1.0)

            if weight < 0.01:  # Skip negligible contributions
                continue

            if blended is None:
                blended = vector * weight
            else:
                blended = blended + (vector * weight)

        return blended

    # =========================================================================
    # Chat Completion (mirrors LlamaInference interface)
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

        # Generate with or without steering
        with self._steering_context(steering_vector):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or 512,
                temperature=temperature or 0.7,
                top_p=top_p or 0.9,
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

    def _steering_context(self, steering_vector: Optional["SteeringVector"]):
        """
        Context manager that applies steering vector during generation.

        Uses the steering-vectors library's built-in application mechanism.
        """
        if steering_vector is None or not STEERING_AVAILABLE:
            # No-op context manager
            return _NullContext()

        # The steering-vectors library provides this
        return steering_vector.apply(self.model)

    # =========================================================================
    # Streaming Chat Completion
    # =========================================================================

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

        # Generation happens in a thread, streamer yields tokens
        def generate():
            with self._steering_context(steering_vector):
                self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or 512,
                    temperature=temperature or 0.7,
                    top_p=top_p or 0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                )

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

    # =========================================================================
    # Function Calling (simplified - may need enhancement for your use case)
    # =========================================================================

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
        you may want to use a fine-tuned model or more sophisticated prompting.
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

        This is a basic implementation - Llama 3.1 has native tool support
        but the exact format depends on your setup.
        """
        # For now, append tool descriptions to the system message
        # and parse the response. A more robust implementation would
        # use the model's native tool calling format.

        tool_desc = self._format_tools_for_prompt(tools)

        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[0]["role"] == "system":
            enhanced_messages[0]["content"] += f"\n\nAvailable tools:\n{tool_desc}"
        else:
            enhanced_messages.insert(0, {
                "role": "system",
                "content": f"You have access to the following tools:\n{tool_desc}"
            })

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
        """
        # TODO: Implement parsing based on your expected format
        # For Llama 3.1, tool calls typically appear in a specific format
        # that you'd need to parse here.
        return None


class _NullContext:
    """No-op context manager for when steering is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


# =============================================================================
# Example emotion prompts for training steering vectors
# =============================================================================

EXAMPLE_EMOTION_PROMPTS = {
    "excited": (
        # Positive examples (excited state)
        [
            "I feel so energized and enthusiastic right now!",
            "This is amazing! I can't wait to explore this further!",
            "I'm thrilled about this new discovery!",
            "Everything feels possible and exciting!",
            "I have so much energy and motivation!",
        ],
        # Negative examples (opposite of excited)
        [
            "I feel tired and drained.",
            "This is boring and uninteresting.",
            "I don't really care about this.",
            "Everything feels like a chore.",
            "I have no energy or motivation.",
        ],
    ),
    "frustrated": (
        # Positive examples (frustrated state)
        [
            "This is so frustrating! Nothing is working!",
            "I keep hitting walls and can't make progress.",
            "Why won't this work? I've tried everything!",
            "I'm stuck and getting increasingly annoyed.",
            "This error keeps happening no matter what I do!",
        ],
        # Negative examples (opposite of frustrated)
        [
            "Everything is going smoothly.",
            "I'm making great progress on this.",
            "This is working exactly as expected.",
            "I feel calm and in control.",
            "Problems are resolving themselves nicely.",
        ],
    ),
    "calm": (
        # Positive examples (calm state)
        [
            "I feel peaceful and centered.",
            "Everything is fine, there's no rush.",
            "I'm relaxed and taking my time.",
            "Things are proceeding at a comfortable pace.",
            "I feel serene and unhurried.",
        ],
        # Negative examples (opposite of calm)
        [
            "I need to hurry! There's no time!",
            "Everything is urgent and stressful!",
            "I'm anxious and on edge.",
            "So much pressure! I can't relax!",
            "My mind is racing with worries.",
        ],
    ),
    "depleted": (
        # Positive examples (depleted state)
        [
            "I feel exhausted and empty.",
            "I don't have the energy for this.",
            "Everything feels heavy and difficult.",
            "I'm running on fumes here.",
            "I need a break, I'm worn out.",
        ],
        # Negative examples (opposite of depleted)
        [
            "I feel refreshed and ready!",
            "I have plenty of energy for this.",
            "Everything feels light and easy.",
            "I'm full of vitality!",
            "I'm well-rested and capable.",
        ],
    ),
}


# =============================================================================
# Integration helper: update EmotionalState to output steering coefficients
# =============================================================================

def get_steering_coefficients(valence: float, arousal: float) -> Dict[str, float]:
    """
    Convert valence-arousal coordinates to steering vector coefficients.

    Maps the 2D emotional space to blend weights for each quadrant emotion.

    Args:
        valence: -1.0 (unpleasant) to +1.0 (pleasant)
        arousal: -1.0 (low energy) to +1.0 (high energy)

    Returns:
        Dictionary of emotion names to blend coefficients (0-1)
    """
    # Normalize to 0-1 range
    v = (valence + 1) / 2  # 0 = negative valence, 1 = positive valence
    a = (arousal + 1) / 2  # 0 = low arousal, 1 = high arousal

    return {
        "excited": v * a,           # high valence, high arousal
        "frustrated": (1 - v) * a,  # low valence, high arousal
        "calm": v * (1 - a),        # high valence, low arousal
        "depleted": (1 - v) * (1 - a),  # low valence, low arousal
    }


# =============================================================================
# Usage example
# =============================================================================

if __name__ == "__main__":
    # Example usage

    # Initialize inference engine
    engine = TransformersInference(
        model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        device="auto",
        steering_config=EmotionalSteeringConfig(
            steering_layer=15,
            steering_strength=1.0,
        ),
    )

    # Train emotion vectors (do this once, then save)
    engine.train_emotion_vectors(EXAMPLE_EMOTION_PROMPTS)
    engine.save_emotion_vectors("./emotion_vectors")

    # Or load pre-trained vectors
    # engine.load_emotion_vectors("./emotion_vectors")

    # Generate with emotional steering
    messages = [
        {"role": "user", "content": "How are you feeling today?"}
    ]

    # Simulate an "excited" emotional state
    coefficients = get_steering_coefficients(valence=0.7, arousal=0.8)

    response = asyncio.run(engine.chat_completion(
        messages=messages,
        emotion_coefficients=coefficients,
    ))

    print(f"Response: {response}")
