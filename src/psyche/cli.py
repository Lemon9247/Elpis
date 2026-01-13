"""Command-line interface for Psyche."""

import sys
from pathlib import Path

from loguru import logger

from psyche.client.app import PsycheApp
from psyche.mcp.client import ElpisClient
from psyche.memory.server import MemoryServer, ServerConfig


def setup_logging(debug: bool = False, log_file: str | None = None) -> None:
    """Configure logging.

    Args:
        debug: Enable debug level logging
        log_file: Path to log file. If provided, logs go to file instead of stderr.
                  This is required when using the Textual TUI to avoid breaking the display.
    """
    logger.remove()
    level = "DEBUG" if debug else "INFO"

    if log_file:
        # Log to file when running TUI (stderr breaks Textual)
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation="10 MB",
        )
    else:
        # Log to stderr for non-TUI usage
        logger.add(
            sys.stderr,
            level=level,
            format="<level>{level: <8}</level> | {message}",
        )


def main(
    server_command: str = "elpis-server",
    debug: bool = False,
    workspace: str = ".",
    log_file: str | None = None,
) -> None:
    """
    Main entry point for Psyche CLI.

    Args:
        server_command: Command to launch Elpis server
        debug: Enable debug logging
        workspace: Working directory for tool operations
        log_file: Path to log file (default: ~/.psyche/psyche.log)
    """
    # Default log file in user's home directory
    if log_file is None:
        log_dir = Path.home() / ".psyche"
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / "psyche.log")

    setup_logging(debug, log_file)

    # Create client for Elpis connection
    client = ElpisClient(server_command=server_command)

    # Configure server
    server_config = ServerConfig(
        idle_think_interval=30.0,
        emotional_modulation=True,
        workspace_dir=workspace,
        allow_idle_tools=True,  # Enable sandboxed tool use during reflection
    )

    # Create memory server
    server = MemoryServer(elpis_client=client, config=server_config)

    # Create and run Textual app
    app = PsycheApp(memory_server=server)

    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
