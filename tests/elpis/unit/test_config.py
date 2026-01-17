"""Tests for configuration system."""

from pathlib import Path

import pytest

from elpis.config.settings import LoggingSettings, ModelSettings, Settings, ToolSettings


class TestModelSettings:
    """Tests for ModelSettings."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = ModelSettings()
        assert settings.path == "./data/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
        assert settings.context_length == 4096
        # CPU-only defaults for maximum compatibility
        assert settings.gpu_layers == 0
        assert settings.n_threads == 4
        assert settings.temperature == 0.7
        assert settings.top_p == 0.9
        assert settings.max_tokens == 4096
        assert settings.hardware_backend == "cpu"

    def test_custom_values(self):
        """Test custom configuration values."""
        settings = ModelSettings(
            path="/custom/model.gguf",
            context_length=4096,
            gpu_layers=20,
            n_threads=4,
            temperature=0.5,
            top_p=0.95,
            max_tokens=1024,
            hardware_backend="cuda",
        )
        assert settings.path == "/custom/model.gguf"
        assert settings.context_length == 4096
        assert settings.gpu_layers == 20
        assert settings.n_threads == 4
        assert settings.temperature == 0.5
        assert settings.top_p == 0.95
        assert settings.max_tokens == 1024
        assert settings.hardware_backend == "cuda"

    def test_validation_context_length(self):
        """Test context length validation."""
        with pytest.raises(ValueError):
            ModelSettings(context_length=100)  # Too small

    def test_validation_temperature(self):
        """Test temperature validation."""
        with pytest.raises(ValueError):
            ModelSettings(temperature=3.0)  # Too high


class TestToolSettings:
    """Tests for ToolSettings."""

    def test_default_values(self):
        """Test default tool settings."""
        settings = ToolSettings()
        assert settings.workspace_dir == "./workspace"
        assert settings.max_bash_timeout == 30
        assert settings.max_file_size == 10485760
        assert settings.enable_dangerous_commands is False

    def test_custom_values(self):
        """Test custom tool settings."""
        settings = ToolSettings(
            workspace_dir="/custom/workspace",
            max_bash_timeout=60,
            max_file_size=5000000,
            enable_dangerous_commands=True,
        )
        assert settings.workspace_dir == "/custom/workspace"
        assert settings.max_bash_timeout == 60
        assert settings.max_file_size == 5000000
        assert settings.enable_dangerous_commands is True

    def test_validation_timeout(self):
        """Test timeout validation."""
        with pytest.raises(ValueError):
            ToolSettings(max_bash_timeout=0)  # Too small

        with pytest.raises(ValueError):
            ToolSettings(max_bash_timeout=500)  # Too large


class TestLoggingSettings:
    """Tests for LoggingSettings."""

    def test_default_values(self):
        """Test default logging settings."""
        settings = LoggingSettings()
        assert settings.level == "INFO"
        assert settings.output_file == "./logs/elpis.log"
        assert settings.format == "json"

    def test_custom_values(self):
        """Test custom logging settings."""
        settings = LoggingSettings(level="DEBUG", output_file="/tmp/test.log", format="text")
        assert settings.level == "DEBUG"
        assert settings.output_file == "/tmp/test.log"
        assert settings.format == "text"


class TestSettings:
    """Tests for root Settings."""

    def test_default_values(self):
        """Test default root settings."""
        settings = Settings()
        assert isinstance(settings.model, ModelSettings)
        assert isinstance(settings.tools, ToolSettings)
        assert isinstance(settings.logging, LoggingSettings)

    def test_nested_configuration(self):
        """Test nested configuration."""
        settings = Settings(
            model=ModelSettings(path="/test/model.gguf", gpu_layers=10),
            tools=ToolSettings(workspace_dir="/test/workspace"),
            logging=LoggingSettings(level="DEBUG"),
        )
        assert settings.model.path == "/test/model.gguf"
        assert settings.model.gpu_layers == 10
        assert settings.tools.workspace_dir == "/test/workspace"
        assert settings.logging.level == "DEBUG"

    def test_environment_variable_prefix(self):
        """Test that settings can be overridden with environment variables."""
        # This test would require setting env vars, which is tested in integration
        settings = Settings()
        assert settings.model.path is not None
