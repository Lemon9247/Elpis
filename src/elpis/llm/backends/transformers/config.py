"""Configuration for HuggingFace Transformers backend.

This backend supports steering vector injection for direct activation
modulation of emotional expression. Requires torch and transformers.
"""

from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TransformersConfig(BaseSettings):
    """Configuration for HuggingFace Transformers inference backend.

    This backend provides:
    - Full HuggingFace model support (Llama, Mistral, etc.)
    - Steering vector injection for activation-level emotional modulation
    - GPU acceleration with automatic dtype selection

    Emotional modulation is achieved by injecting pre-trained steering vectors
    into model activations during the forward pass.
    """

    path: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="HuggingFace model ID or local path",
    )
    context_length: int = Field(
        default=32768,
        ge=512,
        le=131072,
        description="Context window size in tokens",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default sampling temperature",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Default top-p sampling parameter",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=32768,
        description="Default maximum tokens to generate",
    )
    hardware_backend: Literal["auto", "cuda", "rocm", "cpu"] = Field(
        default="auto",
        description="Hardware backend for inference",
    )
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = Field(
        default="auto",
        description="Torch dtype for model weights",
    )
    steering_layer: int = Field(
        default=15,
        ge=0,
        le=80,
        description="Layer index for steering vector injection",
    )
    emotion_vectors_dir: Optional[str] = Field(
        default=None,
        description="Directory containing trained emotion vectors (.pt files)",
    )

    model_config = SettingsConfigDict(
        env_prefix="ELPIS_TRANSFORMERS_",
        extra="ignore",
    )
