"""Tests for Mnemosyne settings."""

import os
from unittest.mock import patch

import pytest

from mnemosyne.config.settings import (
    ConsolidationSettings,
    LoggingSettings,
    Settings,
    StorageSettings,
)


class TestStorageSettings:
    """Test StorageSettings configuration."""

    def test_defaults(self):
        """Test default values."""
        settings = StorageSettings()
        assert settings.persist_directory == "./data/memory"
        assert settings.embedding_model == "all-MiniLM-L6-v2"
        assert settings.short_term_collection == "short_term_memory"
        assert settings.long_term_collection == "long_term_memory"

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(
            os.environ,
            {"MNEMOSYNE_STORAGE_PERSIST_DIRECTORY": "/custom/path"},
        ):
            settings = StorageSettings()
            assert settings.persist_directory == "/custom/path"


class TestConsolidationSettings:
    """Test ConsolidationSettings configuration."""

    def test_defaults(self):
        """Test default values match shared constants."""
        from psyche.shared.constants import (
            CONSOLIDATION_IMPORTANCE_THRESHOLD,
            CONSOLIDATION_SIMILARITY_THRESHOLD,
        )

        settings = ConsolidationSettings()
        assert settings.importance_threshold == CONSOLIDATION_IMPORTANCE_THRESHOLD
        assert settings.similarity_threshold == CONSOLIDATION_SIMILARITY_THRESHOLD
        assert settings.min_age_hours == 1
        assert settings.max_batch_size == 50

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(
            os.environ,
            {"MNEMOSYNE_CONSOLIDATION_IMPORTANCE_THRESHOLD": "0.8"},
        ):
            settings = ConsolidationSettings()
            assert settings.importance_threshold == 0.8


class TestLoggingSettings:
    """Test LoggingSettings configuration."""

    def test_defaults(self):
        """Test default values."""
        settings = LoggingSettings()
        assert settings.level == "INFO"
        assert settings.quiet is False

    def test_quiet_env_override(self):
        """Test quiet mode via environment variable."""
        with patch.dict(os.environ, {"MNEMOSYNE_LOGGING_QUIET": "true"}):
            settings = LoggingSettings()
            assert settings.quiet is True


class TestSettings:
    """Test root Settings configuration."""

    def test_nested_settings(self):
        """Test nested settings are properly initialized."""
        settings = Settings()
        assert isinstance(settings.storage, StorageSettings)
        assert isinstance(settings.consolidation, ConsolidationSettings)
        assert isinstance(settings.logging, LoggingSettings)

    def test_nested_env_override(self):
        """Test nested environment variable override.

        Note: When instantiating root Settings, nested fields use the
        env_nested_delimiter (FIELD__SUBFIELD format). The nested class's
        own env_prefix (e.g., MNEMOSYNE_STORAGE_) only works when
        instantiating that class directly.
        """
        with patch.dict(
            os.environ,
            {"STORAGE__PERSIST_DIRECTORY": "/nested/path"},
        ):
            settings = Settings()
            assert settings.storage.persist_directory == "/nested/path"
