"""Pydantic settings models for Elpis configuration."""

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Type

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

if TYPE_CHECKING:
    from elpis.llm.backends.llama_cpp.config import LlamaCppConfig
    from elpis.llm.backends.transformers.config import TransformersConfig


class ModelSettings(BaseSettings):
    """LLM configuration (llama-cpp backend)."""

    backend: Literal["llama-cpp", "transformers"] = Field(
        default="llama-cpp",
        description="Inference backend: llama-cpp (GGUF) or transformers (HuggingFace)",
    )
    path: str = Field(
        default="./data/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        description="Path to GGUF model file or HuggingFace model ID",
    )
    # 4096 context + 20 GPU layers fits in 6GB VRAM (~3.9GB used)
    context_length: int = Field(
        default=4096,
        ge=512,
        le=131072,
        description="Context window size in tokens",
    )
    # Set to 0 for CPU-only inference, or increase for GPU offloading
    gpu_layers: int = Field(default=0, ge=0, le=100)
    # For CPU-only inference, use more threads. For GPU, keep low to avoid race conditions.
    n_threads: int = Field(default=4, ge=1, le=64)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    hardware_backend: str = Field(
        default="cpu", description="Hardware backend: auto, cuda, rocm, cpu"
    )

    # Transformers-specific settings
    torch_dtype: str = Field(
        default="auto",
        description="Torch dtype for transformers: auto, float16, bfloat16, float32",
    )
    steering_layer: int = Field(
        default=15,
        ge=0,
        le=80,
        description="Layer to apply steering vectors (transformers only)",
    )
    emotion_vectors_dir: Optional[str] = Field(
        default=None,
        description="Directory containing trained emotion vectors (.pt files)",
    )

    model_config = SettingsConfigDict(env_prefix="ELPIS_MODEL_")

    def to_llama_cpp_config(self) -> "LlamaCppConfig":
        """Convert to llama-cpp backend config.

        Returns:
            LlamaCppConfig instance with relevant settings copied
        """
        from elpis.llm.backends.llama_cpp.config import LlamaCppConfig

        return LlamaCppConfig(
            path=self.path,
            context_length=self.context_length,
            gpu_layers=self.gpu_layers,
            n_threads=self.n_threads,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            hardware_backend=self.hardware_backend,
        )

    def to_transformers_config(self) -> "TransformersConfig":
        """Convert to transformers backend config.

        Returns:
            TransformersConfig instance with relevant settings copied
        """
        from elpis.llm.backends.transformers.config import TransformersConfig

        return TransformersConfig(
            path=self.path,
            context_length=self.context_length,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            hardware_backend=self.hardware_backend,
            torch_dtype=self.torch_dtype,
            steering_layer=self.steering_layer,
            emotion_vectors_dir=self.emotion_vectors_dir,
        )


class EmotionSettings(BaseSettings):
    """Emotional regulation configuration."""

    baseline_valence: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Baseline valence (pleasant/unpleasant)",
    )
    baseline_arousal: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Baseline arousal (high/low energy)",
    )
    decay_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Rate of return to baseline per second",
    )
    max_delta: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Maximum single-event emotional shift",
    )
    steering_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=3.0,
        description="Global steering strength multiplier",
    )

    model_config = SettingsConfigDict(env_prefix="ELPIS_EMOTION_")


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO")
    output_file: str = Field(default="./logs/elpis.log")
    format: str = Field(default="json")

    model_config = SettingsConfigDict(env_prefix="ELPIS_LOGGING_")


class Settings(BaseSettings):
    """Root configuration for Elpis."""

    model: ModelSettings = Field(default_factory=ModelSettings)
    emotion: EmotionSettings = Field(default_factory=EmotionSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(
        toml_file="configs/elpis.toml",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Configure settings sources with TOML support.

        Priority (highest to lowest):
        1. Init settings (constructor arguments)
        2. Environment variables
        3. TOML config file
        4. Default values
        """
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
        )
