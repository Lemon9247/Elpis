"""Tests for Hermes settings."""

import os
from unittest.mock import patch

import pytest

from hermes.config.settings import (
    ConnectionSettings,
    IdleSettings,
    LoggingSettings,
    Settings,
    WorkspaceSettings,
)


class TestIdleSettings:
    """Test IdleSettings configuration."""

    def test_defaults(self):
        """Test default values."""
        settings = IdleSettings()
        assert settings.post_interaction_delay == 60.0
        assert settings.idle_tool_cooldown_seconds == 300.0
        assert settings.startup_warmup_seconds == 120.0
        assert settings.max_idle_tool_iterations == 3
        assert settings.max_idle_result_chars == 8000
        assert settings.think_temperature == 0.7
        assert settings.generation_timeout == 120.0
        assert settings.allow_idle_tools is True
        assert settings.emotional_modulation is True

    def test_env_override_post_interaction_delay(self):
        """Test environment variable override for post_interaction_delay."""
        with patch.dict(
            os.environ,
            {"HERMES_IDLE_POST_INTERACTION_DELAY": "45.0"},
        ):
            settings = IdleSettings()
            assert settings.post_interaction_delay == 45.0

    def test_env_override_idle_tool_cooldown(self):
        """Test environment variable override for idle_tool_cooldown_seconds."""
        with patch.dict(
            os.environ,
            {"HERMES_IDLE_IDLE_TOOL_COOLDOWN_SECONDS": "180.0"},
        ):
            settings = IdleSettings()
            assert settings.idle_tool_cooldown_seconds == 180.0

    def test_env_override_think_temperature(self):
        """Test environment variable override for think_temperature."""
        with patch.dict(
            os.environ,
            {"HERMES_IDLE_THINK_TEMPERATURE": "0.5"},
        ):
            settings = IdleSettings()
            assert settings.think_temperature == 0.5

    def test_env_override_allow_idle_tools(self):
        """Test environment variable override for allow_idle_tools."""
        with patch.dict(
            os.environ,
            {"HERMES_IDLE_ALLOW_IDLE_TOOLS": "false"},
        ):
            settings = IdleSettings()
            assert settings.allow_idle_tools is False


class TestConnectionSettings:
    """Test ConnectionSettings configuration."""

    def test_defaults(self):
        """Test default values."""
        settings = ConnectionSettings()
        assert settings.server_url is None
        assert settings.elpis_command == "elpis-server"
        assert settings.mnemosyne_command == "mnemosyne-server"

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(
            os.environ,
            {"HERMES_CONNECTION_SERVER_URL": "http://localhost:9000"},
        ):
            settings = ConnectionSettings()
            assert settings.server_url == "http://localhost:9000"


class TestWorkspaceSettings:
    """Test WorkspaceSettings configuration."""

    def test_defaults(self):
        """Test default values."""
        settings = WorkspaceSettings()
        assert settings.path == "."

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(
            os.environ,
            {"HERMES_WORKSPACE_PATH": "/custom/path"},
        ):
            settings = WorkspaceSettings()
            assert settings.path == "/custom/path"


class TestLoggingSettings:
    """Test LoggingSettings configuration."""

    def test_defaults(self):
        """Test default values."""
        settings = LoggingSettings()
        assert settings.debug is False
        assert settings.log_file is None

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(
            os.environ,
            {"HERMES_LOGGING_DEBUG": "true"},
        ):
            settings = LoggingSettings()
            assert settings.debug is True


class TestSettings:
    """Test root Settings configuration."""

    def test_nested_settings(self):
        """Test nested settings are properly initialized."""
        settings = Settings()
        assert isinstance(settings.connection, ConnectionSettings)
        assert isinstance(settings.workspace, WorkspaceSettings)
        assert isinstance(settings.logging, LoggingSettings)
        assert isinstance(settings.idle, IdleSettings)

    def test_feature_flags_defaults(self):
        """Test feature flag defaults."""
        settings = Settings()
        assert settings.enable_memory is True
        assert settings.enable_idle is True
