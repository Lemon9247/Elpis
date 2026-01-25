"""Tests for configuration loading."""

import pytest
import tempfile
from pathlib import Path


class TestRetrievalSettingsLoading:
    """Test that RetrievalSettings are properly loaded and applied."""

    def test_retrieval_settings_defaults(self):
        """Test default RetrievalSettings values."""
        from mnemosyne.config.settings import RetrievalSettings

        settings = RetrievalSettings()

        assert settings.emotion_weight == 0.3
        assert settings.vector_weight == 0.5
        assert settings.bm25_weight == 0.5
        assert settings.recency_weight == 0.3
        assert settings.importance_weight == 0.2
        assert settings.relevance_weight == 0.5
        assert settings.use_angular_similarity is False

    def test_retrieval_settings_override(self):
        """Test that RetrievalSettings can be overridden."""
        from mnemosyne.config.settings import RetrievalSettings

        settings = RetrievalSettings(
            emotion_weight=0.5,
            vector_weight=0.7,
            bm25_weight=0.3,
            use_angular_similarity=True,
        )

        assert settings.emotion_weight == 0.5
        assert settings.vector_weight == 0.7
        assert settings.bm25_weight == 0.3
        assert settings.use_angular_similarity is True

    def test_retrieval_settings_passed_to_store(self):
        """Test that RetrievalSettings are properly passed to ChromaMemoryStore."""
        pytest.importorskip("chromadb")
        pytest.importorskip("sentence_transformers")

        from mnemosyne.config.settings import RetrievalSettings
        from mnemosyne.storage.chroma_store import ChromaMemoryStore

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = RetrievalSettings(
                semantic_type_factor=1.5,
                assistant_role_factor=1.3,
            )

            store = ChromaMemoryStore(
                persist_directory=tmpdir,
                retrieval_settings=settings,
            )

            # Verify settings were applied
            assert store.retrieval_settings.semantic_type_factor == 1.5
            assert store.retrieval_settings.assistant_role_factor == 1.3


class TestEmotionSettingsLoading:
    """Test that EmotionSettings are properly loaded and applied."""

    def test_emotion_settings_defaults(self):
        """Test default EmotionSettings values."""
        from elpis.config.settings import EmotionSettings

        settings = EmotionSettings()

        assert settings.baseline_valence == 0.0
        assert settings.baseline_arousal == 0.0
        assert settings.trajectory_history_size == 20
        assert settings.momentum_positive_threshold == 0.01
        assert settings.spiral_history_count == 5

    def test_emotion_settings_override(self):
        """Test that EmotionSettings can be overridden."""
        from elpis.config.settings import EmotionSettings

        settings = EmotionSettings(
            baseline_valence=0.2,
            trajectory_history_size=30,
            spiral_history_count=7,
        )

        assert settings.baseline_valence == 0.2
        assert settings.trajectory_history_size == 30
        assert settings.spiral_history_count == 7


class TestTrajectoryConfigFromSettings:
    """Test TrajectoryConfig creation from EmotionSettings."""

    def test_trajectory_config_from_settings(self):
        """Test TrajectoryConfig properly loads from EmotionSettings."""
        from elpis.config.settings import EmotionSettings
        from elpis.emotion.state import TrajectoryConfig

        settings = EmotionSettings(
            trajectory_history_size=25,
            momentum_positive_threshold=0.02,
            momentum_negative_threshold=-0.02,
            trend_improving_threshold=0.03,
            trend_declining_threshold=-0.03,
            spiral_history_count=6,
            spiral_increasing_threshold=4,
        )

        config = TrajectoryConfig.from_settings(settings)

        assert config.history_size == 25
        assert config.momentum_positive_threshold == 0.02
        assert config.momentum_negative_threshold == -0.02
        assert config.trend_improving_threshold == 0.03
        assert config.trend_declining_threshold == -0.03
        assert config.spiral_history_count == 6
        assert config.spiral_increasing_threshold == 4

    def test_trajectory_config_defaults(self):
        """Test TrajectoryConfig default values."""
        from elpis.emotion.state import TrajectoryConfig

        config = TrajectoryConfig()

        assert config.history_size == 20
        assert config.momentum_positive_threshold == 0.01
        assert config.momentum_negative_threshold == -0.01


class TestSettingsIntegration:
    """Test full settings integration."""

    def test_mnemosyne_settings_nested(self):
        """Test Mnemosyne nested settings structure."""
        from mnemosyne.config.settings import Settings

        settings = Settings()

        # Check nested structure
        assert hasattr(settings, "storage")
        assert hasattr(settings, "retrieval")
        assert hasattr(settings, "consolidation")

        # Check some defaults
        assert settings.retrieval.emotion_weight == 0.3
        assert settings.storage.embedding_model == "all-MiniLM-L6-v2"

    def test_elpis_settings_nested(self):
        """Test Elpis nested settings structure."""
        from elpis.config.settings import Settings

        settings = Settings()

        # Check nested structure
        assert hasattr(settings, "model")
        assert hasattr(settings, "emotion")

        # Check some defaults
        assert settings.emotion.baseline_valence == 0.0
        assert settings.emotion.trajectory_history_size == 20
