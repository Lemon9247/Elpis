"""Tests for Hermes settings."""

import os
from unittest.mock import patch

import pytest

from hermes.config.settings import (
    ConnectionSettings,
    LoggingSettings,
    Settings,
    WorkspaceSettings,
)


class TestConnectionSettings:
    """Test ConnectionSettings configuration."""

    def test_defaults(self):
        """Test default values."""
        settings = ConnectionSettings()
        assert settings.server_url == "http://127.0.0.1:8741"

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
