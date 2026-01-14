"""Tests for LLM inference module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elpis.config.settings import ModelSettings
from elpis.utils.exceptions import LLMInferenceError, ModelLoadError
from elpis.utils.hardware import HardwareBackend

# Check if LlamaInference is available
try:
    from elpis.llm.inference import LlamaInference
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


@pytest.mark.skipif(not LLAMA_CPP_AVAILABLE, reason="llama-cpp-python not installed")
class TestLlamaInference:
    """Tests for LlamaInference class."""

    @patch("elpis.llm.inference.Llama")
    @patch("elpis.llm.inference.detect_hardware")
    def test_init_auto_backend(self, mock_detect, mock_llama):
        """Test initialization with auto backend detection."""
        mock_detect.return_value = HardwareBackend.CPU
        mock_llama.return_value = MagicMock()

        settings = ModelSettings(
            path="/test/model.gguf",
            context_length=512,
            gpu_layers=0,
            n_threads=2,
            hardware_backend="auto",
        )

        llm = LlamaInference(settings)

        assert llm.backend == HardwareBackend.CPU
        mock_detect.assert_called_once()
        mock_llama.assert_called_once()

    @patch("elpis.llm.inference.Llama")
    @patch("elpis.llm.inference.detect_hardware")
    def test_init_explicit_backend(self, mock_detect, mock_llama):
        """Test initialization with explicit backend."""
        mock_llama.return_value = MagicMock()

        settings = ModelSettings(
            path="/test/model.gguf",
            context_length=512,
            gpu_layers=0,
            n_threads=2,
            hardware_backend="cuda",
        )

        llm = LlamaInference(settings)

        assert llm.backend == HardwareBackend.CUDA
        mock_detect.assert_not_called()  # Should not auto-detect
        mock_llama.assert_called_once()

    @patch("elpis.llm.inference.Llama")
    def test_model_load_failure(self, mock_llama):
        """Test model load failure."""
        mock_llama.side_effect = Exception("Failed to load model")

        settings = ModelSettings(path="/nonexistent/model.gguf", hardware_backend="cpu")

        with pytest.raises(ModelLoadError):
            LlamaInference(settings)

    @patch("elpis.llm.inference.Llama")
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, mock_llama):
        """Test successful chat completion."""
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_llama.return_value = mock_model

        settings = ModelSettings(path="/test/model.gguf", hardware_backend="cpu")
        llm = LlamaInference(settings)

        messages = [{"role": "user", "content": "Test message"}]
        response = await llm.chat_completion(messages)

        assert response == "Test response"
        mock_model.create_chat_completion.assert_called_once()

    @patch("elpis.llm.inference.Llama")
    @pytest.mark.asyncio
    async def test_chat_completion_with_params(self, mock_llama):
        """Test chat completion with custom parameters."""
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_llama.return_value = mock_model

        settings = ModelSettings(path="/test/model.gguf", hardware_backend="cpu")
        llm = LlamaInference(settings)

        messages = [{"role": "user", "content": "Test message"}]
        response = await llm.chat_completion(
            messages, max_tokens=100, temperature=0.5, top_p=0.8
        )

        assert response == "Test response"
        call_args = mock_model.create_chat_completion.call_args
        assert call_args[1]["max_tokens"] == 100
        assert call_args[1]["temperature"] == 0.5
        assert call_args[1]["top_p"] == 0.8

    @patch("elpis.llm.inference.Llama")
    @pytest.mark.asyncio
    async def test_chat_completion_failure(self, mock_llama):
        """Test chat completion failure."""
        mock_model = MagicMock()
        mock_model.create_chat_completion.side_effect = Exception("Inference failed")
        mock_llama.return_value = mock_model

        settings = ModelSettings(path="/test/model.gguf", hardware_backend="cpu")
        llm = LlamaInference(settings)

        messages = [{"role": "user", "content": "Test message"}]

        with pytest.raises(LLMInferenceError):
            await llm.chat_completion(messages)

    @patch("elpis.llm.inference.Llama")
    @pytest.mark.asyncio
    async def test_function_call_with_tools(self, mock_llama):
        """Test function call with tools."""
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
                            }
                        ]
                    }
                }
            ]
        }
        mock_llama.return_value = mock_model

        settings = ModelSettings(path="/test/model.gguf", hardware_backend="cpu")
        llm = LlamaInference(settings)

        messages = [{"role": "user", "content": "Test message"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        tool_calls = await llm.function_call(messages, tools)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_1"

    @patch("elpis.llm.inference.Llama")
    @pytest.mark.asyncio
    async def test_function_call_no_tools(self, mock_llama):
        """Test function call when no tools should be called."""
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Regular response"}}]
        }
        mock_llama.return_value = mock_model

        settings = ModelSettings(path="/test/model.gguf", hardware_backend="cpu")
        llm = LlamaInference(settings)

        messages = [{"role": "user", "content": "Test message"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        tool_calls = await llm.function_call(messages, tools)

        assert tool_calls is None

    @patch("elpis.llm.inference.Llama")
    @pytest.mark.asyncio
    async def test_function_call_failure(self, mock_llama):
        """Test function call failure."""
        mock_model = MagicMock()
        mock_model.create_chat_completion.side_effect = Exception("Function call failed")
        mock_llama.return_value = mock_model

        settings = ModelSettings(path="/test/model.gguf", hardware_backend="cpu")
        llm = LlamaInference(settings)

        messages = [{"role": "user", "content": "Test message"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        with pytest.raises(LLMInferenceError):
            await llm.function_call(messages, tools)
