"""Tests for Psyche settings.

Note: IdleSettings tests have been moved to tests/hermes/
as part of making Psyche a stateless API.
"""

import os
from unittest.mock import patch

import pytest

from psyche.config.settings import (
    ConsolidationSettings,
    ContextSettings,
    MemorySettings,
    ReasoningSettings,
    ServerSettings,
    Settings,
    ToolSettings,
)
from shared.constants import (
    AUTO_STORAGE_THRESHOLD,
    CONSOLIDATION_IMPORTANCE_THRESHOLD,
    CONSOLIDATION_SIMILARITY_THRESHOLD,
)


class TestContextSettings:
    """Test ContextSettings configuration."""

    def test_defaults(self):
        """Test default values."""
        settings = ContextSettings()
        assert settings.max_context_tokens == 24000
        assert settings.reserve_tokens == 4000
        assert settings.enable_checkpoints is True
        assert settings.checkpoint_interval == 20

    def test_from_elpis_capabilities(self):
        """Test creating settings from Elpis capabilities."""
        settings = ContextSettings.from_elpis_capabilities(context_length=32768)
        assert settings.max_context_tokens == int(32768 * 0.75)
        assert settings.reserve_tokens == int(32768 * 0.20)

    def test_from_elpis_capabilities_custom_ratios(self):
        """Test custom ratios."""
        settings = ContextSettings.from_elpis_capabilities(
            context_length=16384,
            context_ratio=0.80,
            reserve_ratio=0.15,
        )
        assert settings.max_context_tokens == int(16384 * 0.80)
        assert settings.reserve_tokens == int(16384 * 0.15)

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(
            os.environ,
            {"PSYCHE_CONTEXT_MAX_CONTEXT_TOKENS": "16000"},
        ):
            settings = ContextSettings()
            assert settings.max_context_tokens == 16000


class TestMemorySettings:
    """Test MemorySettings configuration."""

    def test_defaults(self):
        """Test default values match shared constants."""
        settings = MemorySettings()
        assert settings.enable_auto_retrieval is True
        assert settings.auto_retrieval_count == 3
        assert settings.auto_storage is True
        assert settings.auto_storage_threshold == AUTO_STORAGE_THRESHOLD

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(
            os.environ,
            {"PSYCHE_MEMORY_AUTO_RETRIEVAL_COUNT": "5"},
        ):
            settings = MemorySettings()
            assert settings.auto_retrieval_count == 5


class TestConsolidationSettings:
    """Test ConsolidationSettings configuration."""

    def test_defaults(self):
        """Test default values match shared constants."""
        settings = ConsolidationSettings()
        assert settings.enabled is True
        assert settings.importance_threshold == CONSOLIDATION_IMPORTANCE_THRESHOLD
        assert settings.similarity_threshold == CONSOLIDATION_SIMILARITY_THRESHOLD


class TestServerSettings:
    """Test ServerSettings configuration."""

    def test_defaults(self):
        """Test default values."""
        settings = ServerSettings()
        assert settings.http_host == "127.0.0.1"
        assert settings.http_port == 8741
        assert settings.elpis_command == "elpis-server"
        assert settings.mnemosyne_command == "mnemosyne-server"
        assert settings.dream_enabled is True

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(
            os.environ,
            {"PSYCHE_SERVER_HTTP_PORT": "9000"},
        ):
            settings = ServerSettings()
            assert settings.http_port == 9000


class TestToolSettings:
    """Test ToolSettings configuration."""

    def test_defaults(self):
        """Test default values."""
        settings = ToolSettings()
        assert settings.bash_timeout == 30
        assert settings.max_file_size == 1_000_000
        assert settings.tool_timeout == 60.0


class TestSettings:
    """Test root Settings configuration."""

    def test_nested_settings(self):
        """Test nested settings are properly initialized."""
        settings = Settings()
        assert isinstance(settings.context, ContextSettings)
        assert isinstance(settings.memory, MemorySettings)
        assert isinstance(settings.reasoning, ReasoningSettings)
        # Note: idle settings removed - now in hermes.config.settings
        assert isinstance(settings.consolidation, ConsolidationSettings)
        assert isinstance(settings.server, ServerSettings)
        assert isinstance(settings.tools, ToolSettings)

    def test_emotional_modulation_default(self):
        """Test emotional modulation default."""
        settings = Settings()
        assert settings.emotional_modulation is True
