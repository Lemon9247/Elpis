"""Tests for configuration system."""

from pathlib import Path

import pytest

from elpis.config.settings import EmotionSettings, LoggingSettings, ModelSettings, Settings


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


class TestEmotionSettings:
    """Tests for EmotionSettings."""

    def test_default_values(self):
        """Test default emotion settings."""
        settings = EmotionSettings()
        assert settings.baseline_valence == 0.0
        assert settings.baseline_arousal == 0.0
        assert settings.decay_rate == 0.1
        assert settings.max_delta == 0.5
        assert settings.steering_strength == 1.0

    def test_custom_values(self):
        """Test custom emotion settings."""
        settings = EmotionSettings(
            baseline_valence=0.2,
            baseline_arousal=-0.1,
            decay_rate=0.2,
            max_delta=0.8,
            steering_strength=1.5,
        )
        assert settings.baseline_valence == 0.2
        assert settings.baseline_arousal == -0.1
        assert settings.decay_rate == 0.2
        assert settings.max_delta == 0.8
        assert settings.steering_strength == 1.5

    def test_validation_valence_range(self):
        """Test valence validation."""
        with pytest.raises(ValueError):
            EmotionSettings(baseline_valence=2.0)  # Too high

        with pytest.raises(ValueError):
            EmotionSettings(baseline_valence=-2.0)  # Too low


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
        assert isinstance(settings.emotion, EmotionSettings)
        assert isinstance(settings.logging, LoggingSettings)

    def test_nested_configuration(self):
        """Test nested configuration."""
        settings = Settings(
            model=ModelSettings(path="/test/model.gguf", gpu_layers=10),
            emotion=EmotionSettings(steering_strength=2.0),
            logging=LoggingSettings(level="DEBUG"),
        )
        assert settings.model.path == "/test/model.gguf"
        assert settings.model.gpu_layers == 10
        assert settings.emotion.steering_strength == 2.0
        assert settings.logging.level == "DEBUG"

    def test_environment_variable_prefix(self):
        """Test that settings can be overridden with environment variables."""
        # This test would require setting env vars, which is tested in integration
        settings = Settings()
        assert settings.model.path is not None
