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

    # Trajectory tracking thresholds
    trajectory_history_size: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Number of states to keep in trajectory history",
    )
    momentum_positive_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=0.5,
        description="Valence velocity above which momentum is 'positive'",
    )
    momentum_negative_threshold: float = Field(
        default=-0.01,
        ge=-0.5,
        le=0.0,
        description="Valence velocity below which momentum is 'negative'",
    )
    trend_improving_threshold: float = Field(
        default=0.02,
        ge=0.0,
        le=0.5,
        description="Valence velocity above which trend is 'improving'",
    )
    trend_declining_threshold: float = Field(
        default=-0.02,
        ge=-0.5,
        le=0.0,
        description="Valence velocity below which trend is 'declining'",
    )
    spiral_history_count: int = Field(
        default=5,
        ge=3,
        le=20,
        description="Number of recent states to check for spiral detection",
    )
    spiral_increasing_threshold: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Minimum increasing distances to detect a spiral",
    )

    # Context-aware intensity (event compounding/dampening)
    streak_compounding_enabled: bool = Field(
        default=True,
        description="Enable event compounding for repeated failures and success dampening",
    )
    streak_compounding_factor: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Intensity change per repeated event",
    )

    # Mood inertia (resistance to rapid emotional changes)
    mood_inertia_enabled: bool = Field(
        default=True,
        description="Enable mood inertia to resist rapid emotional swings",
    )
    mood_inertia_resistance: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Maximum resistance factor for counter-momentum events",
    )

    # Quadrant-specific decay multipliers
    # Lower values = emotion persists longer, higher = decays faster
    decay_multiplier_excited: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Decay rate multiplier for excited quadrant",
    )
    decay_multiplier_frustrated: float = Field(
        default=0.7,
        ge=0.1,
        le=3.0,
        description="Decay rate multiplier for frustrated quadrant (persists longer)",
    )
    decay_multiplier_calm: float = Field(
        default=1.2,
        ge=0.1,
        le=3.0,
        description="Decay rate multiplier for calm quadrant (decays slightly faster)",
    )
    decay_multiplier_depleted: float = Field(
        default=0.8,
        ge=0.1,
        le=3.0,
        description="Decay rate multiplier for depleted quadrant (persists)",
    )

    # Response analysis settings
    response_analysis_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold to trigger emotion from response analysis",
    )

    # Behavioral monitoring
    behavioral_monitoring_enabled: bool = Field(
        default=True,
        description="Enable behavioral pattern monitoring (retry loops, failure streaks)",
    )
    retry_loop_threshold: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Number of same-tool calls to detect a retry loop",
    )
    failure_streak_threshold: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of consecutive failures to trigger compounding",
    )
    long_generation_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Duration in seconds to consider a generation 'long'",
    )
    idle_period_seconds: float = Field(
        default=120.0,
        ge=30.0,
        le=600.0,
        description="Duration without activity to trigger calming idle event",
    )

    # LLM-based emotion analysis (optional enhancement)
    llm_emotion_analysis_enabled: bool = Field(
        default=False,
        description="Enable LLM-based emotion analysis for deeper response understanding",
    )
    llm_analysis_min_length: int = Field(
        default=200,
        ge=50,
        le=1000,
        description="Minimum response length (chars) to trigger LLM analysis",
    )
    use_local_sentiment_model: bool = Field(
        default=True,
        description="Use lightweight local sentiment model instead of full LLM",
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
