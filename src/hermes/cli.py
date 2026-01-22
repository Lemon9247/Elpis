"""
Hermes CLI - Command-line entry point for the TUI client.

Usage:
    hermes                          # Connect to default server (localhost:8741)
    hermes --server URL             # Connect to specific Psyche server
    hermes --debug                  # Enable debug logging

Hermes connects to a running Psyche server via HTTP API.
File/bash/search tools execute locally, while the server handles
inference and memory.
"""

import signal
import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger


# Configure logging IMMEDIATELY to capture any logs during subsequent imports
# This MUST happen before importing any modules that use loguru
def _setup_logging_early() -> None:
    """Configure logging to file immediately to avoid TUI interference."""
    logger.remove()  # Remove default stderr handler
    log_dir = Path.home() / ".psyche"
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "psyche.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        rotation="10 MB",
    )


_setup_logging_early()

# Apply MCP library patch BEFORE any MCP imports
# Fixes: RuntimeError: dictionary keys changed during iteration
from shared.mcp_patch import apply_mcp_patch

apply_mcp_patch()

# Now safe to import modules that may use logging
from hermes.app import Hermes
from hermes.handlers import RemotePsycheClient
from hermes.tools import ToolEngine, ToolSettings


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


def _run_async_disconnect(client: RemotePsycheClient) -> None:
    """Disconnect remote client in a new event loop."""
    import asyncio

    async def disconnect():
        try:
            logger.info("Disconnecting from Psyche server...")
            await client.disconnect()
            logger.info("Disconnected")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(disconnect())
        loop.close()
    except Exception as e:
        logger.error(f"Failed to disconnect: {e}")


def _run_remote_mode(
    server_url: str,
    debug: bool,
    workspace: str,
    log_file: str,
) -> None:
    """Run Hermes, connecting to a Psyche server."""
    setup_logging(debug, log_file)

    logger.info(f"Connecting to Psyche server at {server_url}...")

    # Verify server is reachable with a simple sync check
    # (Don't use aiohttp here - it would bind to wrong event loop)
    import urllib.request
    import urllib.error

    try:
        health_url = f"{server_url.rstrip('/')}/health"
        with urllib.request.urlopen(health_url, timeout=5) as resp:
            if resp.status != 200:
                raise ConnectionError(f"Server returned {resp.status}")
        logger.info("Server health check passed")
    except (urllib.error.URLError, ConnectionError) as e:
        logger.error(f"Failed to connect to server: {e}")
        print(f"Error: Cannot connect to Psyche server at {server_url}")
        print("Make sure the server is running: psyche-server")
        sys.exit(1)

    # --- Create remote client (connects lazily inside Textual's event loop) ---
    client = RemotePsycheClient(base_url=server_url)

    # --- Create ToolEngine (tools run locally) ---
    tool_engine = ToolEngine(
        workspace_dir=workspace,
        settings=ToolSettings(),
    )

    # Set tools in OpenAI format for remote client
    client.set_tools(tool_engine.get_openai_tool_definitions())
    tool_descriptions = tool_engine.get_tool_descriptions()
    client.set_tool_descriptions(tool_descriptions)

    # --- Create Textual app ---
    app = Hermes(
        client=client,
        tool_engine=tool_engine,
    )

    # Signal handler
    def signal_handler(signum: int, frame) -> None:
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, disconnecting...")
        try:
            app.exit()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, signal_handler)

    # Redirect stderr
    stderr_log = Path.home() / ".psyche" / "stderr.log"
    original_stderr = sys.stderr

    try:
        with open(stderr_log, "a") as stderr_file:
            sys.stderr = stderr_file
            app.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        sys.stderr = original_stderr
        _run_async_disconnect(client)


# Default server URL
DEFAULT_SERVER_URL = "http://127.0.0.1:8741"


@click.command()
@click.option(
    "--server",
    "server_url",
    default=DEFAULT_SERVER_URL,
    help=f"Psyche server URL (default: {DEFAULT_SERVER_URL})",
)
@click.option(
    "--workspace",
    default=".",
    type=click.Path(exists=True),
    help="Working directory for tool operations",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
@click.option(
    "--log-file",
    type=click.Path(),
    help="Path to log file",
)
def main(
    server_url: str,
    workspace: str,
    debug: bool,
    log_file: Optional[str],
) -> None:
    """
    Hermes - TUI client for Psyche.

    Connects to a Psyche server and provides a terminal interface for
    chatting with emotional state display and local tool execution.

    Examples:

        # Connect to default server (localhost:8741)
        hermes

        # Connect to specific server
        hermes --server http://myserver:8741

        # Enable debug logging
        hermes --debug
    """
    # Default log file
    if log_file is None:
        log_dir = Path.home() / ".psyche"
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / "psyche.log")

    print(f"Connecting to Psyche server at {server_url}...")
    _run_remote_mode(
        server_url=server_url,
        debug=debug,
        workspace=workspace,
        log_file=log_file,
    )


if __name__ == "__main__":
    main()
