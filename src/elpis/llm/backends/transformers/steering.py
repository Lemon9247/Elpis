"""Steering vector management for emotional expression.

This module handles the loading, blending, and application of steering vectors
for direct activation modulation during model inference.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

# Runtime imports
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

# Type-checking imports
if TYPE_CHECKING:
    import torch


class SteeringManager:
    """Manages steering vector loading, blending, and hook application.

    Steering vectors are pre-trained direction vectors in activation space
    that encode emotional states. By adding these vectors to intermediate
    layer activations during inference, we can modulate the emotional
    expression of generated text.
    """

    def __init__(self, device: str, steering_layer: int):
        """Initialize the steering manager.

        Args:
            device: Device to load vectors onto (cuda, cpu, etc.)
            steering_layer: Layer index where steering vectors will be injected
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for SteeringManager. "
                "Install with: pip install torch"
            )

        self.device = device
        self.steering_layer = steering_layer
        self.vectors: Dict[str, "torch.Tensor"] = {}
        self._hook_handle: Optional[Any] = None

    def load_vectors(self, directory: str) -> None:
        """Load pre-trained emotion steering vectors from disk.

        Scans the specified directory for .pt files and loads each as a
        named steering vector. The filename (without extension) becomes
        the vector name.

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
                vector = torch.load(
                    vector_file, map_location=self.device, weights_only=True
                )
                self.vectors[name] = vector
                logger.info(f"Loaded emotion vector: {name} (shape={vector.shape})")
            except Exception as e:
                logger.error(f"Failed to load {vector_file}: {e}")

        if self.vectors:
            logger.info(f"Loaded {len(self.vectors)} emotion vectors")
        else:
            logger.warning(f"No emotion vectors found in {path}")

    def compute_blended_vector(
        self, coefficients: Dict[str, float]
    ) -> Optional["torch.Tensor"]:
        """Blend steering vectors based on emotional coefficients.

        Creates a weighted combination of loaded steering vectors using
        the provided coefficients. Coefficients below 0.01 are ignored
        as negligible contributions.

        Args:
            coefficients: Mapping of emotion name to blend weight.
                Example: {"excited": 0.7, "calm": 0.3}

        Returns:
            Blended steering vector, or None if no vectors available
            or no matching emotions found.
        """
        if not self.vectors:
            return None

        blended: Optional["torch.Tensor"] = None
        total_weight = 0.0

        for emotion_name, coef in coefficients.items():
            if emotion_name not in self.vectors:
                continue

            vector = self.vectors[emotion_name]
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

    def apply_hook(self, model: Any, steering_vector: "torch.Tensor") -> None:
        """Register forward hook to inject steering vector at target layer.

        The hook adds the steering vector to the hidden state activations
        at the configured layer during the forward pass.

        Args:
            model: The HuggingFace model to attach the hook to
            steering_vector: The steering vector to inject
        """
        layer_idx = self.steering_layer

        # Access the layer (Llama-style architecture assumed)
        try:
            if hasattr(model, "model"):
                # Model has .model attribute (common in HF models)
                layers = model.model.layers
            else:
                # Direct access
                layers = model.layers

            target_layer = layers[layer_idx]
        except (AttributeError, IndexError) as e:
            logger.error(f"Cannot access layer {layer_idx}: {e}")
            return

        def hook_fn(
            module: Any, input: Any, output: Any
        ) -> Any:
            """Add steering vector to the output activations."""
            # output is typically a tuple (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Add steering vector (broadcast across batch and sequence)
            # steering_vector shape: (hidden_dim,)
            # hidden_states shape: (batch, seq_len, hidden_dim)
            steered = hidden_states + steering_vector.to(hidden_states.dtype).to(
                hidden_states.device
            )

            if isinstance(output, tuple):
                return (steered,) + output[1:]
            else:
                return steered

        self._hook_handle = target_layer.register_forward_hook(hook_fn)
        logger.debug(f"Steering hook registered on layer {layer_idx}")

    def remove_hook(self) -> None:
        """Remove the active steering hook if present."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            logger.debug("Steering hook removed")

    @property
    def has_vectors(self) -> bool:
        """Check if any steering vectors are loaded."""
        return len(self.vectors) > 0

    @property
    def available_emotions(self) -> list[str]:
        """Get list of available emotion vector names."""
        return list(self.vectors.keys())
