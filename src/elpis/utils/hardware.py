"""GPU hardware detection for CUDA, ROCm, and CPU fallback."""

import subprocess
from enum import Enum
from typing import Optional


class HardwareBackend(Enum):
    """Available hardware backends for LLM inference."""

    CUDA = "cuda"  # NVIDIA GPU
    ROCM = "rocm"  # AMD GPU
    CPU = "cpu"  # CPU only


def detect_hardware() -> HardwareBackend:
    """
    Detect available GPU hardware.
    Priority: CUDA > ROCm > CPU
    """
    # Check for NVIDIA GPU (CUDA)
    if check_cuda_available():
        return HardwareBackend.CUDA

    # Check for AMD GPU (ROCm)
    if check_rocm_available():
        return HardwareBackend.ROCM

    # Fallback to CPU
    return HardwareBackend.CPU


def check_cuda_available() -> bool:
    """Check if NVIDIA GPU with CUDA is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=2, check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_rocm_available() -> bool:
    """Check if AMD GPU with ROCm is available."""
    try:
        result = subprocess.run(
            ["rocm-smi"], capture_output=True, timeout=2, check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_recommended_gpu_layers(backend: HardwareBackend) -> int:
    """Get recommended GPU layers for backend."""
    return {
        HardwareBackend.CUDA: 35,  # Offload all layers
        HardwareBackend.ROCM: 35,  # Offload all layers
        HardwareBackend.CPU: 0,  # No GPU offloading
    }[backend]
