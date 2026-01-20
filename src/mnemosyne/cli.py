"""CLI entry point for mnemosyne-server command."""

import asyncio
import sys
from pathlib import Path

from loguru import logger

from mnemosyne.config.settings import Settings
from mnemosyne.server import initialize, run_server


def setup_logging(settings: Settings) -> None:
    """Configure logging based on settings."""
    logger.remove()

    # Check if we should suppress stderr logging (e.g., when run as subprocess by Psyche TUI)
    # Set via MNEMOSYNE_LOGGING__QUIET=true or settings.logging.quiet
    quiet_mode = settings.logging.quiet

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
            level=settings.logging.level,
            format="<level>{level: <8}</level> | {message}",
        )


def main() -> None:
    """Main entry point."""
    # Load settings from environment
    settings = Settings()

    setup_logging(settings)

    try:
        initialize(settings=settings)
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\nServer stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
