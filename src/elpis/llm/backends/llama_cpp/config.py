"""Configuration for llama-cpp-python backend.

This backend uses GGUF quantized models and supports CPU/CUDA/ROCm.
Emotional modulation is achieved via sampling parameter adjustment
(temperature, top_p) rather than activation steering.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlamaCppConfig(BaseSettings):
    """Configuration for llama-cpp-python inference backend.

    This backend provides:
    - GGUF quantized model support (4-bit, 5-bit, 8-bit)
    - CPU/CUDA/ROCm hardware acceleration
    - Emotional modulation via sampling parameters

    Note: emotion_coefficients are ignored by this backend. For steering
    vector support, use the transformers backend instead.
    """

    path: str = Field(
        default="./data/models/model.gguf",
        description="Path to GGUF model file",
    )
    context_length: int = Field(
        default=32768,
        ge=512,
        le=131072,
        description="Context window size in tokens",
    )
    gpu_layers: int = Field(
        default=35,
        ge=0,
        le=100,
        description="Number of layers to offload to GPU",
    )
    n_threads: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Number of CPU threads for inference",
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
    chat_format: str = Field(
        default="llama-3",
        description="Chat template format for the model",
    )

    model_config = SettingsConfigDict(
        env_prefix="ELPIS_LLAMA_CPP_",
        extra="ignore",
    )
