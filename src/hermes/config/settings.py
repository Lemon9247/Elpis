"""Pydantic settings models for Hermes configuration."""

from typing import Optional, Tuple, Type

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class IdleSettings(BaseSettings):
    """Idle behavior configuration (client-side).

    Moved from psyche.config.settings as part of making Psyche stateless.
    """

    post_interaction_delay: float = Field(
        default=60.0,
        ge=0.0,
        description="Seconds to wait after user input before idle thinking",
    )
    idle_tool_cooldown_seconds: float = Field(
        default=300.0,
        ge=0.0,
        description="Minimum seconds between idle tool uses",
    )
    startup_warmup_seconds: float = Field(
        default=120.0,
        ge=0.0,
        description="Seconds after startup before tools allowed in idle mode",
    )
    max_idle_tool_iterations: int = Field(
        default=3,
        ge=0,
        description="Maximum tool iterations per idle thought",
    )
    max_idle_result_chars: int = Field(
        default=8000,
        ge=100,
        description="Truncate tool results to this size",
    )
    think_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for reflection generation",
    )
    generation_timeout: float = Field(
        default=120.0,
        ge=1.0,
        description="Generation timeout in seconds",
    )
    allow_idle_tools: bool = Field(
        default=True,
        description="Allow tool use during idle reflection",
    )
    emotional_modulation: bool = Field(
        default=True,
        description="Use emotional modulation during idle",
    )

    model_config = SettingsConfigDict(env_prefix="HERMES_IDLE_")


class ConnectionSettings(BaseSettings):
    """Server connection configuration."""

    server_url: Optional[str] = Field(
        default=None,
        description="Psyche server URL (None = local mode)",
    )
    elpis_command: str = Field(
        default="elpis-server",
        description="Command to launch Elpis MCP server (local mode)",
    )
    mnemosyne_command: str = Field(
        default="mnemosyne-server",
        description="Command to launch Mnemosyne MCP server (local mode)",
    )

    model_config = SettingsConfigDict(env_prefix="HERMES_CONNECTION_")


class WorkspaceSettings(BaseSettings):
    """Workspace configuration."""

    path: str = Field(
        default=".",
        description="Workspace directory path",
    )

    model_config = SettingsConfigDict(env_prefix="HERMES_WORKSPACE_")


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    debug: bool = Field(
        default=False,
        description="Enable debug logging",
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path (None = no file logging)",
    )

    model_config = SettingsConfigDict(env_prefix="HERMES_LOGGING_")


class Settings(BaseSettings):
    """Root configuration for Hermes."""

    connection: ConnectionSettings = Field(default_factory=ConnectionSettings)
    workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    idle: IdleSettings = Field(default_factory=IdleSettings)

    # Feature flags
    enable_memory: bool = Field(
        default=True,
        description="Enable memory via Mnemosyne",
    )
    enable_idle: bool = Field(
        default=True,
        description="Enable idle thinking",
    )

    model_config = SettingsConfigDict(
        toml_file="configs/hermes.toml",
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
