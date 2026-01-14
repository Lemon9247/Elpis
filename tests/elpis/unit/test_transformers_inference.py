"""Unit tests for TransformersInference backend."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from elpis.config.settings import ModelSettings

# TransformersInference requires torch/transformers which may not be installed
# These tests will be skipped if dependencies are missing
try:
    import torch
    from elpis.llm.transformers_inference import TransformersInference
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers/torch not installed")
class TestTransformersInference:
    """Tests for TransformersInference class."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = ModelSettings(
            backend="transformers",
            path="test-model",
            context_length=2048,
            steering_layer=10,
            emotion_vectors_dir=None,
        )
        return settings

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Mock the model and tokenizer loading."""
        with patch("elpis.llm.transformers_inference.AutoModelForCausalLM") as mock_model_cls, \
             patch("elpis.llm.transformers_inference.AutoTokenizer") as mock_tok_cls:

            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "<pad>"
            mock_tokenizer.eos_token = "<eos>"
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.pad_token_id = 0
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer

            # Mock model
            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model

            # Mock layers for hook attachment
            mock_layer = MagicMock()
            mock_layers = [mock_layer] * 20
            mock_model.model.layers = mock_layers

            mock_model_cls.from_pretrained.return_value = mock_model

            yield mock_model, mock_tokenizer

    def test_initialization(self, mock_settings, mock_model_and_tokenizer):
        """Test that TransformersInference initializes correctly."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer

        engine = TransformersInference(mock_settings)

        assert engine.settings == mock_settings
        assert engine.context_length == 2048
        assert engine.emotion_vectors == {}  # No vectors loaded initially

    def test_device_selection_auto_cuda(self, mock_settings, mock_model_and_tokenizer):
        """Test device selection when CUDA available."""
        mock_settings.hardware_backend = "auto"

        with patch("torch.cuda.is_available", return_value=True):
            engine = TransformersInference(mock_settings)
            assert engine.device == "cuda"

    def test_device_selection_auto_cpu(self, mock_settings, mock_model_and_tokenizer):
        """Test device selection when CUDA not available."""
        mock_settings.hardware_backend = "auto"

        with patch("torch.cuda.is_available", return_value=False):
            engine = TransformersInference(mock_settings)
            assert engine.device == "cpu"

    def test_dtype_selection_auto(self, mock_settings, mock_model_and_tokenizer):
        """Test automatic dtype selection."""
        mock_settings.torch_dtype = "auto"

        with patch("torch.cuda.is_available", return_value=True):
            engine = TransformersInference(mock_settings)
            assert engine.torch_dtype == torch.bfloat16

        with patch("torch.cuda.is_available", return_value=False):
            engine = TransformersInference(mock_settings)
            assert engine.torch_dtype == torch.float32

    def test_load_emotion_vectors_success(self, mock_settings, mock_model_and_tokenizer, tmp_path):
        """Test loading emotion vectors from directory."""
        # Create temporary vector files
        vectors_dir = tmp_path / "vectors"
        vectors_dir.mkdir()

        for emotion in ["excited", "frustrated", "calm", "depleted"]:
            vector = torch.randn(4096)  # Typical hidden dim
            torch.save(vector, vectors_dir / f"{emotion}.pt")

        mock_settings.emotion_vectors_dir = str(vectors_dir)
        engine = TransformersInference(mock_settings)

        assert len(engine.emotion_vectors) == 4
        assert "excited" in engine.emotion_vectors
        assert "frustrated" in engine.emotion_vectors
        assert "calm" in engine.emotion_vectors
        assert "depleted" in engine.emotion_vectors

    def test_load_emotion_vectors_missing_directory(self, mock_settings, mock_model_and_tokenizer):
        """Test loading vectors from non-existent directory."""
        mock_settings.emotion_vectors_dir = "/nonexistent/path"
        engine = TransformersInference(mock_settings)

        # Should not raise, just log warning
        assert len(engine.emotion_vectors) == 0

    def test_compute_blended_steering_no_vectors(self, mock_settings, mock_model_and_tokenizer):
        """Test blending when no vectors loaded."""
        engine = TransformersInference(mock_settings)

        coeffs = {"excited": 0.5, "calm": 0.3, "frustrated": 0.1, "depleted": 0.1}
        result = engine._compute_blended_steering(coeffs)

        assert result is None

    def test_compute_blended_steering_with_vectors(self, mock_settings, mock_model_and_tokenizer):
        """Test blending emotion vectors."""
        engine = TransformersInference(mock_settings)

        # Add mock vectors
        hidden_dim = 4096
        engine.emotion_vectors = {
            "excited": torch.ones(hidden_dim) * 1.0,
            "frustrated": torch.ones(hidden_dim) * -1.0,
            "calm": torch.ones(hidden_dim) * 0.5,
            "depleted": torch.ones(hidden_dim) * -0.5,
        }

        coeffs = {"excited": 0.5, "calm": 0.3, "frustrated": 0.1, "depleted": 0.1}
        result = engine._compute_blended_steering(coeffs)

        assert result is not None
        assert result.shape == (hidden_dim,)

        # Check blending math: 0.5*1.0 + 0.1*(-1.0) + 0.3*0.5 + 0.1*(-0.5)
        expected_value = 0.5 * 1.0 + 0.1 * (-1.0) + 0.3 * 0.5 + 0.1 * (-0.5)
        assert torch.allclose(result[0], torch.tensor(expected_value), atol=1e-6)

    def test_compute_blended_steering_skip_negligible(self, mock_settings, mock_model_and_tokenizer):
        """Test that negligible coefficients are skipped."""
        engine = TransformersInference(mock_settings)

        hidden_dim = 4096
        engine.emotion_vectors = {
            "excited": torch.ones(hidden_dim),
            "frustrated": torch.ones(hidden_dim) * 2.0,
        }

        # Frustrated coefficient is too small (< 0.01)
        coeffs = {"excited": 1.0, "frustrated": 0.005}
        result = engine._compute_blended_steering(coeffs)

        assert result is not None
        # Should be just the excited vector
        assert torch.allclose(result, engine.emotion_vectors["excited"])

    def test_steering_hook_registration(self, mock_settings, mock_model_and_tokenizer):
        """Test that steering hook is registered correctly."""
        mock_model, _ = mock_model_and_tokenizer
        engine = TransformersInference(mock_settings)

        steering_vector = torch.randn(4096)
        engine._apply_steering_hook(steering_vector)

        assert engine._steering_hook_handle is not None

        # Verify hook was registered on correct layer
        target_layer = mock_model.model.layers[mock_settings.steering_layer]
        target_layer.register_forward_hook.assert_called_once()

    def test_steering_hook_cleanup(self, mock_settings, mock_model_and_tokenizer):
        """Test that steering hook is properly removed."""
        engine = TransformersInference(mock_settings)

        # Mock a hook handle
        mock_handle = MagicMock()
        engine._steering_hook_handle = mock_handle

        engine._remove_steering_hook()

        mock_handle.remove.assert_called_once()
        assert engine._steering_hook_handle is None

    @pytest.mark.asyncio
    async def test_chat_completion_without_steering(self, mock_settings, mock_model_and_tokenizer):
        """Test chat completion without emotion coefficients."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        engine = TransformersInference(mock_settings)

        # Mock tokenizer methods
        mock_tokenizer.apply_chat_template.return_value = "test prompt"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        # Mock model generation
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_tokenizer.decode.return_value = "Test response"

        messages = [{"role": "user", "content": "Hello"}]
        response = await engine.chat_completion(messages)

        assert response == "Test response"
        mock_model.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_completion_with_steering(self, mock_settings, mock_model_and_tokenizer, tmp_path):
        """Test chat completion with emotion coefficients."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer

        # Create vectors
        vectors_dir = tmp_path / "vectors"
        vectors_dir.mkdir()
        for emotion in ["excited", "calm"]:
            torch.save(torch.randn(4096), vectors_dir / f"{emotion}.pt")

        mock_settings.emotion_vectors_dir = str(vectors_dir)
        engine = TransformersInference(mock_settings)

        # Mock tokenizer
        mock_tokenizer.apply_chat_template.return_value = "test prompt"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        # Mock model generation
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_tokenizer.decode.return_value = "Excited response"

        messages = [{"role": "user", "content": "Hello"}]
        coeffs = {"excited": 0.8, "calm": 0.2}

        response = await engine.chat_completion(messages, emotion_coefficients=coeffs)

        assert response == "Excited response"
        mock_model.generate.assert_called_once()


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers/torch not installed")
class TestTransformersInferenceEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        return ModelSettings(
            backend="transformers",
            path="test-model",
            steering_layer=10,
        )

    def test_missing_dependencies_raises_error(self):
        """Test that missing transformers raises helpful error."""
        # This test only makes sense if we can mock the import failure
        with patch.dict("sys.modules", {"transformers": None, "torch": None}):
            with pytest.raises(RuntimeError, match="transformers and torch are required"):
                # Force reimport to trigger the check
                from importlib import reload
                import elpis.llm.transformers_inference as ti
                reload(ti)

    def test_invalid_layer_index(self, mock_settings):
        """Test handling of invalid layer index."""
        with patch("elpis.llm.transformers_inference.AutoModelForCausalLM") as mock_model_cls, \
             patch("elpis.llm.transformers_inference.AutoTokenizer") as mock_tok_cls:

            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "<pad>"
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer

            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model
            # Only 5 layers
            mock_model.model.layers = [MagicMock()] * 5
            mock_model_cls.from_pretrained.return_value = mock_model

            mock_settings.steering_layer = 10  # Out of range
            engine = TransformersInference(mock_settings)

            # Should not raise during init, only when trying to apply hook
            steering_vector = torch.randn(4096)
            # This should handle the error gracefully
            engine._apply_steering_hook(steering_vector)


@pytest.mark.skipif(TRANSFORMERS_AVAILABLE, reason="Only test import guard when deps missing")
class TestTransformersInferenceImportGuard:
    """Test that import guard works when dependencies missing."""

    def test_import_guard_active(self):
        """Verify that TRANSFORMERS_AVAILABLE is False without deps."""
        from elpis.llm import transformers_inference
        assert not transformers_inference.TRANSFORMERS_AVAILABLE
