"""Pydantic settings models for Hermes configuration."""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Feature flags
    enable_memory: bool = Field(
        default=True,
        description="Enable memory via Mnemosyne",
    )
    enable_idle: bool = Field(
        default=True,
        description="Enable idle thinking",
    )
    enable_consolidation: bool = Field(
        default=True,
        description="Enable memory consolidation during idle",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
    )
