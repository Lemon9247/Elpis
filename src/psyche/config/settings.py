"""Pydantic settings models for Psyche configuration."""

from __future__ import annotations

from typing import List, Optional, Tuple, Type

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from shared.constants import (
    AUTO_STORAGE_THRESHOLD,
    CONSOLIDATION_IMPORTANCE_THRESHOLD,
    CONSOLIDATION_SIMILARITY_THRESHOLD,
)


class ContextSettings(BaseSettings):
    """Context management configuration."""

    max_context_tokens: int = Field(
        default=24000,
        ge=1000,
        description="Maximum tokens for working context (will be overridden by Elpis)",
    )
    reserve_tokens: int = Field(
        default=4000,
        ge=100,
        description="Tokens reserved for response generation",
    )
    enable_checkpoints: bool = Field(
        default=True,
        description="Enable periodic context checkpointing",
    )
    checkpoint_interval: int = Field(
        default=20,
        ge=1,
        description="Save checkpoint every N messages",
    )

    model_config = SettingsConfigDict(env_prefix="PSYCHE_CONTEXT_")

    @classmethod
    def from_elpis_capabilities(
        cls,
        context_length: int,
        context_ratio: float = 0.75,
        reserve_ratio: float = 0.20,
    ) -> ContextSettings:
        """Create settings based on Elpis context window.

        Args:
            context_length: Total context window from Elpis
            context_ratio: Ratio of context to use for working memory (default 0.75)
            reserve_ratio: Ratio to reserve for response (default 0.20)

        Returns:
            ContextSettings configured for the given context window
        """
        return cls(
            max_context_tokens=int(context_length * context_ratio),
            reserve_tokens=int(context_length * reserve_ratio),
        )


class MemorySettings(BaseSettings):
    """Memory handler configuration."""

    enable_auto_retrieval: bool = Field(
        default=True,
        description="Automatically retrieve relevant memories on user input",
    )
    auto_retrieval_count: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Number of memories to auto-retrieve",
    )
    auto_storage: bool = Field(
        default=True,
        description="Automatically store important exchanges",
    )
    auto_storage_threshold: float = Field(
        default=AUTO_STORAGE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum importance score for auto-storage",
    )

    model_config = SettingsConfigDict(env_prefix="PSYCHE_MEMORY_")


class ReasoningSettings(BaseSettings):
    """Reasoning mode configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable <reasoning> tags in responses",
    )

    model_config = SettingsConfigDict(env_prefix="PSYCHE_REASONING_")


class ConsolidationSettings(BaseSettings):
    """Memory consolidation settings (triggered during idle)."""

    enabled: bool = Field(
        default=True,
        description="Enable memory consolidation during idle",
    )
    check_interval: float = Field(
        default=300.0,
        ge=1.0,
        description="Seconds between consolidation checks",
    )
    importance_threshold: float = Field(
        default=CONSOLIDATION_IMPORTANCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum importance for promotion",
    )
    similarity_threshold: float = Field(
        default=CONSOLIDATION_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for clustering",
    )

    model_config = SettingsConfigDict(env_prefix="PSYCHE_CONSOLIDATION_")


class ServerSettings(BaseSettings):
    """Psyche server daemon configuration."""

    http_host: str = Field(
        default="127.0.0.1",
        description="HTTP server bind address",
    )
    http_port: int = Field(
        default=8741,
        ge=1,
        le=65535,
        description="HTTP server port",
    )
    mcp_enabled: bool = Field(
        default=False,
        description="Enable MCP server (future)",
    )
    elpis_command: str = Field(
        default="elpis-server",
        description="Command to launch Elpis MCP server",
    )
    mnemosyne_command: Optional[str] = Field(
        default="mnemosyne-server",
        description="Command to launch Mnemosyne MCP server (None to disable)",
    )
    dream_enabled: bool = Field(
        default=True,
        description="Enable dreaming when no clients connected",
    )
    dream_delay_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description="Seconds to wait before starting to dream",
    )
    model_name: str = Field(
        default="psyche",
        description="Model name for API responses",
    )

    model_config = SettingsConfigDict(env_prefix="PSYCHE_SERVER_")


class ToolSettings(BaseSettings):
    """Tool engine configuration."""

    bash_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Bash command timeout in seconds",
    )
    max_file_size: int = Field(
        default=1_000_000,
        ge=1024,
        description="Maximum file size in bytes",
    )
    tool_timeout: float = Field(
        default=60.0,
        ge=1.0,
        description="Default tool execution timeout in seconds",
    )
    allowed_extensions: Optional[List[str]] = Field(
        default=None,
        description="Allowed file extensions (None = all)",
    )

    model_config = SettingsConfigDict(env_prefix="PSYCHE_TOOLS_")


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR)",
    )

    model_config = SettingsConfigDict(env_prefix="PSYCHE_LOGGING_")


class Settings(BaseSettings):
    """Root configuration for Psyche."""

    context: ContextSettings = Field(default_factory=ContextSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    reasoning: ReasoningSettings = Field(default_factory=ReasoningSettings)
    # Note: idle settings removed - now in hermes.config.settings
    consolidation: ConsolidationSettings = Field(default_factory=ConsolidationSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # General flags
    emotional_modulation: bool = Field(
        default=True,
        description="Enable emotional modulation in generation",
    )

    model_config = SettingsConfigDict(
        toml_file="configs/psyche.toml",
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
