"""Pydantic settings models for Hermes configuration."""

from typing import Optional, Tuple, Type

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class ConnectionSettings(BaseSettings):
    """Server connection configuration."""

    server_url: str = Field(
        default="http://127.0.0.1:8741",
        description="Psyche server URL",
    )

    model_config = SettingsConfigDict(env_prefix="HERMES_CONNECTION_")


class WorkspaceSettings(BaseSettings):
    """Workspace configuration."""

    path: str = Field(
        default=".",
        description="Workspace directory path for local tool execution",
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
