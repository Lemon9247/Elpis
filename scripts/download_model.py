#!/usr/bin/env python3
"""
Download Llama 3.1 8B Instruct model from HuggingFace.
This script is NOT run automatically - user must execute manually.
"""

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

MODEL_REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
MODEL_FILE = "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
LOCAL_DIR = "./data/models"


def download_model():
    """Download model with progress tracking."""
    console = Console()

    console.print(f"[cyan]Downloading {MODEL_FILE} from {MODEL_REPO}[/cyan]")
    console.print(f"[yellow]Size: ~6.8GB[/yellow]")
    console.print(f"[yellow]Destination: {LOCAL_DIR}[/yellow]")
    console.print()

    # Create directory
    Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)

    # Check if already exists
    model_path = Path(LOCAL_DIR) / MODEL_FILE
    if model_path.exists():
        console.print(f"[green]✓ Model already exists at: {model_path}[/green]")
        return 0

    # Download with progress
    try:
        console.print("[cyan]Starting download...[/cyan]")

        downloaded_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
        )

        console.print(f"\n[green]✓ Download complete: {downloaded_path}[/green]")
        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Download interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]✗ Download failed: {e}[/red]", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(download_model())
