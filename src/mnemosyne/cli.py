"""CLI entry point for mnemosyne-server command."""

import asyncio
import os
import sys
from pathlib import Path

from loguru import logger

from mnemosyne.server import initialize, run_server


def setup_logging() -> None:
    """Configure logging based on environment."""
    logger.remove()

    # Check if we should suppress stderr logging (e.g., when run as subprocess by Psyche TUI)
    # MNEMOSYNE_QUIET env var is set by Psyche to prevent logging from breaking the TUI
    quiet_mode = os.environ.get("MNEMOSYNE_QUIET", "").lower() in ("1", "true", "yes")

    if quiet_mode:
        # Log to file when running as subprocess of a TUI
        log_dir = Path.home() / ".mnemosyne"
        log_dir.mkdir(exist_ok=True)
        logger.add(
            log_dir / "mnemosyne-server.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation="10 MB",
        )
    else:
        # Log to stderr when running standalone
        logger.add(
            sys.stderr,
            level="INFO",
            format="<level>{level: <8}</level> | {message}",
        )


def main() -> None:
    """Main entry point."""
    setup_logging()

    try:
        initialize()
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\nServer stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
