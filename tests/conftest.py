"""Pytest configuration and fixtures for Elpis tests."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from elpis.config.settings import LoggingSettings, ModelSettings, Settings, ToolSettings


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model_path(temp_dir: Path) -> Path:
    """Create a mock model file path."""
    model_path = temp_dir / "test_model.gguf"
    model_path.write_text("mock model data")
    return model_path


@pytest.fixture
def test_settings(temp_dir: Path, mock_model_path: Path) -> Settings:
    """Create test settings with temporary paths."""
    return Settings(
        model=ModelSettings(
            path=str(mock_model_path),
            context_length=512,
            gpu_layers=0,  # CPU only for tests
            n_threads=2,
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            hardware_backend="cpu",
        ),
        tools=ToolSettings(
            workspace_dir=str(temp_dir / "workspace"),
            max_bash_timeout=5,
            max_file_size=1024,
            enable_dangerous_commands=False,
        ),
        logging=LoggingSettings(
            level="DEBUG", output_file=str(temp_dir / "test.log"), format="text"
        ),
    )


@pytest.fixture
def workspace_dir(temp_dir: Path) -> Path:
    """Create a workspace directory for tool tests."""
    workspace = temp_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace
