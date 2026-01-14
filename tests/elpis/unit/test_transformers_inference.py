"""Unit tests for TransformersInference backend."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# TransformersInference requires torch/transformers which may not be installed
# These tests will be skipped if dependencies are missing
try:
    import torch
    from elpis.llm.backends.transformers import TransformersInference, TransformersConfig, SteeringManager
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers/torch not installed")
class TestTransformersInference:
    """Tests for TransformersInference class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        config = TransformersConfig(
            path="test-model",
            context_length=2048,
            steering_layer=10,
            emotion_vectors_dir=None,
        )
        return config

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Mock the model and tokenizer loading."""
        with patch("elpis.llm.backends.transformers.inference.AutoModelForCausalLM") as mock_model_cls, \
             patch("elpis.llm.backends.transformers.inference.AutoTokenizer") as mock_tok_cls:

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
            # Handle .to(device) call - return same mock
            mock_model.to.return_value = mock_model

            # Mock layers for hook attachment
            mock_layer = MagicMock()
            mock_layers = [mock_layer] * 20
            mock_model.model.layers = mock_layers

            mock_model_cls.from_pretrained.return_value = mock_model

            yield mock_model, mock_tokenizer

    def test_initialization(self, mock_config, mock_model_and_tokenizer):
        """Test that TransformersInference initializes correctly."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer

        engine = TransformersInference(mock_config)

        assert engine.config == mock_config
        assert engine.context_length == 2048
        assert not engine.steering.has_vectors  # No vectors loaded initially

    def test_device_selection_auto_cuda(self, mock_config, mock_model_and_tokenizer):
        """Test device selection when CUDA available."""
        mock_config.hardware_backend = "auto"

        with patch("torch.cuda.is_available", return_value=True):
            engine = TransformersInference(mock_config)
            assert engine.device == "cuda"

    def test_device_selection_auto_cpu(self, mock_config, mock_model_and_tokenizer):
        """Test device selection when CUDA not available."""
        mock_config.hardware_backend = "auto"

        with patch("torch.cuda.is_available", return_value=False):
            engine = TransformersInference(mock_config)
            assert engine.device == "cpu"

    def test_dtype_selection_auto(self, mock_config, mock_model_and_tokenizer):
        """Test automatic dtype selection."""
        mock_config.torch_dtype = "auto"

        with patch("torch.cuda.is_available", return_value=True):
            engine = TransformersInference(mock_config)
            assert engine.torch_dtype == torch.bfloat16

        with patch("torch.cuda.is_available", return_value=False):
            engine = TransformersInference(mock_config)
            assert engine.torch_dtype == torch.float32

    def test_load_emotion_vectors_success(self, mock_config, mock_model_and_tokenizer, tmp_path):
        """Test loading emotion vectors from directory."""
        # Create temporary vector files
        vectors_dir = tmp_path / "vectors"
        vectors_dir.mkdir()

        for emotion in ["excited", "frustrated", "calm", "depleted"]:
            vector = torch.randn(4096)  # Typical hidden dim
            torch.save(vector, vectors_dir / f"{emotion}.pt")

        mock_config.emotion_vectors_dir = str(vectors_dir)
        engine = TransformersInference(mock_config)

        assert len(engine.steering.vectors) == 4
        assert "excited" in engine.steering.vectors
        assert "frustrated" in engine.steering.vectors
        assert "calm" in engine.steering.vectors
        assert "depleted" in engine.steering.vectors

    def test_load_emotion_vectors_missing_directory(self, mock_config, mock_model_and_tokenizer):
        """Test loading vectors from non-existent directory."""
        mock_config.emotion_vectors_dir = "/nonexistent/path"
        engine = TransformersInference(mock_config)

        # Should not raise, just log warning
        assert len(engine.steering.vectors) == 0

    def test_compute_blended_steering_no_vectors(self, mock_config, mock_model_and_tokenizer):
        """Test blending when no vectors loaded."""
        engine = TransformersInference(mock_config)

        coeffs = {"excited": 0.5, "calm": 0.3, "frustrated": 0.1, "depleted": 0.1}
        result = engine.steering.compute_blended_vector(coeffs)

        assert result is None

    def test_compute_blended_steering_with_vectors(self, mock_config, mock_model_and_tokenizer):
        """Test blending emotion vectors."""
        engine = TransformersInference(mock_config)

        # Add mock vectors
        hidden_dim = 4096
        engine.steering.vectors = {
            "excited": torch.ones(hidden_dim) * 1.0,
            "frustrated": torch.ones(hidden_dim) * -1.0,
            "calm": torch.ones(hidden_dim) * 0.5,
            "depleted": torch.ones(hidden_dim) * -0.5,
        }

        coeffs = {"excited": 0.5, "calm": 0.3, "frustrated": 0.1, "depleted": 0.1}
        result = engine.steering.compute_blended_vector(coeffs)

        assert result is not None
        assert result.shape == (hidden_dim,)

        # Check blending math: 0.5*1.0 + 0.1*(-1.0) + 0.3*0.5 + 0.1*(-0.5)
        expected_value = 0.5 * 1.0 + 0.1 * (-1.0) + 0.3 * 0.5 + 0.1 * (-0.5)
        assert torch.allclose(result[0], torch.tensor(expected_value), atol=1e-6)

    def test_compute_blended_steering_skip_negligible(self, mock_config, mock_model_and_tokenizer):
        """Test that negligible coefficients are skipped."""
        engine = TransformersInference(mock_config)

        hidden_dim = 4096
        engine.steering.vectors = {
            "excited": torch.ones(hidden_dim),
            "frustrated": torch.ones(hidden_dim) * 2.0,
        }

        # Frustrated coefficient is too small (< 0.01)
        coeffs = {"excited": 1.0, "frustrated": 0.005}
        result = engine.steering.compute_blended_vector(coeffs)

        assert result is not None
        # Should be just the excited vector
        assert torch.allclose(result, engine.steering.vectors["excited"])

    def test_steering_hook_registration(self, mock_config, mock_model_and_tokenizer):
        """Test that steering hook is registered correctly."""
        mock_model, _ = mock_model_and_tokenizer
        engine = TransformersInference(mock_config)

        steering_vector = torch.randn(4096)
        engine.steering.apply_hook(engine.model, steering_vector)

        # Verify hook handle is set
        assert engine.steering._hook_handle is not None

        # Verify the hook handle is a MagicMock (from register_forward_hook return value)
        # This confirms register_forward_hook was called
        assert hasattr(engine.steering._hook_handle, "remove")

    def test_steering_hook_cleanup(self, mock_config, mock_model_and_tokenizer):
        """Test that steering hook is properly removed."""
        engine = TransformersInference(mock_config)

        # Mock a hook handle
        mock_handle = MagicMock()
        engine.steering._hook_handle = mock_handle

        engine.steering.remove_hook()

        mock_handle.remove.assert_called_once()
        assert engine.steering._hook_handle is None

    @pytest.mark.asyncio
    async def test_chat_completion_without_steering(self, mock_config, mock_model_and_tokenizer):
        """Test chat completion without emotion coefficients."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        engine = TransformersInference(mock_config)

        # Mock tokenizer methods
        mock_tokenizer.apply_chat_template.return_value = "test prompt"

        # Create a mock that supports .to() method
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, key: {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}[key]
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        # Mock model generation
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_tokenizer.decode.return_value = "Test response"

        messages = [{"role": "user", "content": "Hello"}]
        response = await engine.chat_completion(messages)

        assert response == "Test response"
        mock_model.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_completion_with_steering(self, mock_config, mock_model_and_tokenizer, tmp_path):
        """Test chat completion with emotion coefficients."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer

        # Create vectors
        vectors_dir = tmp_path / "vectors"
        vectors_dir.mkdir()
        for emotion in ["excited", "calm"]:
            torch.save(torch.randn(4096), vectors_dir / f"{emotion}.pt")

        mock_config.emotion_vectors_dir = str(vectors_dir)
        engine = TransformersInference(mock_config)

        # Mock tokenizer
        mock_tokenizer.apply_chat_template.return_value = "test prompt"

        # Create a mock that supports .to() method
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, key: {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}[key]
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        # Mock model generation
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_tokenizer.decode.return_value = "Excited response"

        messages = [{"role": "user", "content": "Hello"}]
        coeffs = {"excited": 0.8, "calm": 0.2}

        response = await engine.chat_completion(messages, emotion_coefficients=coeffs)

        assert response == "Excited response"
        mock_model.generate.assert_called_once()

    def test_capability_flags(self, mock_config, mock_model_and_tokenizer):
        """Test that capability flags are set correctly."""
        assert TransformersInference.SUPPORTS_STEERING is True
        assert TransformersInference.MODULATION_TYPE == "steering"


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers/torch not installed")
class TestTransformersInferenceEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def mock_model_and_tokenizer_for_edge_cases(self):
        """Mock the model and tokenizer loading for edge case tests."""
        with patch("elpis.llm.backends.transformers.inference.AutoModelForCausalLM") as mock_model_cls, \
             patch("elpis.llm.backends.transformers.inference.AutoTokenizer") as mock_tok_cls:

            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "<pad>"
            mock_tokenizer.eos_token = "<eos>"
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.pad_token_id = 0
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer

            # Mock model with limited layers
            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model
            # Handle .to(device) call - return same mock
            mock_model.to.return_value = mock_model
            # Only 5 layers (as a real list, not MagicMock)
            mock_layer = MagicMock()
            mock_model.model.layers = [mock_layer] * 5
            mock_model_cls.from_pretrained.return_value = mock_model

            yield mock_model, mock_tokenizer

    def test_invalid_layer_index(self, mock_model_and_tokenizer_for_edge_cases):
        """Test handling of invalid layer index."""
        mock_model, _ = mock_model_and_tokenizer_for_edge_cases

        # Create config with layer index beyond model's layer count
        config = TransformersConfig(
            path="test-model",
            steering_layer=10,  # Out of range (model has only 5 layers)
        )
        engine = TransformersInference(config)

        # Should not raise during init, only when trying to apply hook
        steering_vector = torch.randn(4096)
        # This should handle the error gracefully (logs error, doesn't crash)
        engine.steering.apply_hook(engine.model, steering_vector)

        # Hook should NOT be set because layer index is invalid
        assert engine.steering._hook_handle is None


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers/torch not installed")
class TestSteeringManager:
    """Tests for the SteeringManager class."""

    def test_steering_manager_initialization(self):
        """Test SteeringManager initializes correctly."""
        manager = SteeringManager(device="cpu", steering_layer=15)

        assert manager.device == "cpu"
        assert manager.steering_layer == 15
        assert manager.vectors == {}
        assert manager._hook_handle is None

    def test_has_vectors_property(self):
        """Test has_vectors property."""
        manager = SteeringManager(device="cpu", steering_layer=15)

        assert not manager.has_vectors

        manager.vectors["test"] = torch.randn(100)
        assert manager.has_vectors

    def test_available_emotions_property(self):
        """Test available_emotions property."""
        manager = SteeringManager(device="cpu", steering_layer=15)

        assert manager.available_emotions == []

        manager.vectors["excited"] = torch.randn(100)
        manager.vectors["calm"] = torch.randn(100)

        emotions = manager.available_emotions
        assert "excited" in emotions
        assert "calm" in emotions


@pytest.mark.skipif(TRANSFORMERS_AVAILABLE, reason="Only test import guard when deps missing")
class TestTransformersInferenceImportGuard:
    """Test that import guard works when dependencies missing."""

    def test_import_guard_active(self):
        """Verify that TRANSFORMERS_AVAILABLE is False without deps."""
        from elpis.llm.backends import transformers
        assert not transformers.AVAILABLE
