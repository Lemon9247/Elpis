"""Tests for hardware detection module."""

from unittest.mock import MagicMock, patch

import pytest

from elpis.utils.hardware import (
    HardwareBackend,
    check_cuda_available,
    check_rocm_available,
    detect_hardware,
    get_recommended_gpu_layers,
)


class TestHardwareBackend:
    """Tests for HardwareBackend enum."""

    def test_cuda_backend(self):
        """Test CUDA backend enum value."""
        assert HardwareBackend.CUDA.value == "cuda"

    def test_rocm_backend(self):
        """Test ROCm backend enum value."""
        assert HardwareBackend.ROCM.value == "rocm"

    def test_cpu_backend(self):
        """Test CPU backend enum value."""
        assert HardwareBackend.CPU.value == "cpu"


class TestCheckCudaAvailable:
    """Tests for CUDA detection."""

    @patch("subprocess.run")
    def test_cuda_available(self, mock_run):
        """Test successful CUDA detection."""
        mock_run.return_value = MagicMock(returncode=0)
        assert check_cuda_available() is True
        mock_run.assert_called_once_with(
            ["nvidia-smi"], capture_output=True, timeout=2, check=False
        )

    @patch("subprocess.run")
    def test_cuda_not_available(self, mock_run):
        """Test CUDA not available (command fails)."""
        mock_run.return_value = MagicMock(returncode=1)
        assert check_cuda_available() is False

    @patch("subprocess.run")
    def test_cuda_command_not_found(self, mock_run):
        """Test CUDA command not found."""
        mock_run.side_effect = FileNotFoundError()
        assert check_cuda_available() is False

    @patch("subprocess.run")
    def test_cuda_timeout(self, mock_run):
        """Test CUDA detection timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 2)
        assert check_cuda_available() is False


class TestCheckRocmAvailable:
    """Tests for ROCm detection."""

    @patch("subprocess.run")
    def test_rocm_available(self, mock_run):
        """Test successful ROCm detection."""
        mock_run.return_value = MagicMock(returncode=0)
        assert check_rocm_available() is True
        mock_run.assert_called_once_with(
            ["rocm-smi"], capture_output=True, timeout=2, check=False
        )

    @patch("subprocess.run")
    def test_rocm_not_available(self, mock_run):
        """Test ROCm not available (command fails)."""
        mock_run.return_value = MagicMock(returncode=1)
        assert check_rocm_available() is False

    @patch("subprocess.run")
    def test_rocm_command_not_found(self, mock_run):
        """Test ROCm command not found."""
        mock_run.side_effect = FileNotFoundError()
        assert check_rocm_available() is False

    @patch("subprocess.run")
    def test_rocm_timeout(self, mock_run):
        """Test ROCm detection timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("rocm-smi", 2)
        assert check_rocm_available() is False


class TestDetectHardware:
    """Tests for automatic hardware detection."""

    @patch("elpis.utils.hardware.check_cuda_available")
    @patch("elpis.utils.hardware.check_rocm_available")
    def test_detect_cuda(self, mock_rocm, mock_cuda):
        """Test CUDA detection (priority over ROCm)."""
        mock_cuda.return_value = True
        mock_rocm.return_value = True  # Both available, CUDA has priority
        assert detect_hardware() == HardwareBackend.CUDA

    @patch("elpis.utils.hardware.check_cuda_available")
    @patch("elpis.utils.hardware.check_rocm_available")
    def test_detect_rocm(self, mock_rocm, mock_cuda):
        """Test ROCm detection when CUDA not available."""
        mock_cuda.return_value = False
        mock_rocm.return_value = True
        assert detect_hardware() == HardwareBackend.ROCM

    @patch("elpis.utils.hardware.check_cuda_available")
    @patch("elpis.utils.hardware.check_rocm_available")
    def test_detect_cpu_fallback(self, mock_rocm, mock_cuda):
        """Test CPU fallback when no GPU detected."""
        mock_cuda.return_value = False
        mock_rocm.return_value = False
        assert detect_hardware() == HardwareBackend.CPU


class TestGetRecommendedGpuLayers:
    """Tests for recommended GPU layers."""

    def test_cuda_layers(self):
        """Test recommended layers for CUDA."""
        assert get_recommended_gpu_layers(HardwareBackend.CUDA) == 35

    def test_rocm_layers(self):
        """Test recommended layers for ROCm."""
        assert get_recommended_gpu_layers(HardwareBackend.ROCM) == 35

    def test_cpu_layers(self):
        """Test recommended layers for CPU (should be 0)."""
        assert get_recommended_gpu_layers(HardwareBackend.CPU) == 0
