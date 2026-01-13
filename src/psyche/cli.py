"""Command-line interface for Psyche."""

import sys

from loguru import logger

from psyche.client.app import PsycheApp
from psyche.mcp.client import ElpisClient
from psyche.memory.server import MemoryServer, ServerConfig


def setup_logging(debug: bool = False) -> None:
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<level>{level: <8}</level> | {message}",
    )


def main(
    server_command: str = "elpis-server",
    debug: bool = False,
    workspace: str = ".",
) -> None:
    """
    Main entry point for Psyche CLI.

    Args:
        server_command: Command to launch Elpis server
        debug: Enable debug logging
        workspace: Working directory for tool operations
    """
    setup_logging(debug)

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
