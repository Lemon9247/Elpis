"""Pydantic settings models for Elpis configuration."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """LLM configuration."""

    path: str = Field(
        default="./data/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        description="Path to GGUF model file",
    )
    context_length: int = Field(
        default=32768,
        ge=512,
        le=131072,
        description="Context window size in tokens",
    )
    gpu_layers: int = Field(default=35, ge=0, le=100)
    n_threads: int = Field(default=8, ge=1, le=64)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    hardware_backend: str = Field(
        default="auto", description="Hardware backend: auto, cuda, rocm, cpu"
    )

    model_config = SettingsConfigDict(env_prefix="ELPIS_MODEL_")


class ToolSettings(BaseSettings):
    """Tool execution configuration."""

    workspace_dir: str = Field(default="./workspace")
    max_bash_timeout: int = Field(default=30, ge=1, le=300)
    max_file_size: int = Field(default=10485760, ge=1024)  # 10MB
    enable_dangerous_commands: bool = Field(default=False)

    model_config = SettingsConfigDict(env_prefix="ELPIS_TOOLS_")


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
    tools: ToolSettings = Field(default_factory=ToolSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(
        env_file=".env", env_nested_delimiter="__", case_sensitive=False
    )
