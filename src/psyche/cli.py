"""Command-line interface for Psyche."""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

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
from psyche.mcp_patch import apply_mcp_patch
apply_mcp_patch()

# Now safe to import modules that may use logging
from psyche.client.app import PsycheApp
from psyche.mcp.client import ElpisClient, MnemosyneClient
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


def _run_async_cleanup(server: MemoryServer) -> None:
    """Run async cleanup in a new event loop.

    This is called after the Textual app exits to ensure memory consolidation
    happens regardless of how the app was terminated.
    """
    async def cleanup():
        try:
            logger.info("Running post-exit memory consolidation...")
            await server.shutdown_with_consolidation()
            await server.stop()
            logger.info("Memory consolidation complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    try:
        # Create a new event loop for cleanup since Textual's loop is gone
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(cleanup())
        loop.close()
    except Exception as e:
        logger.error(f"Failed to run async cleanup: {e}")


def main(
    server_command: str = "elpis-server",
    mnemosyne_command: str | None = "mnemosyne-server",
    debug: bool = False,
    workspace: str = ".",
    log_file: str | None = None,
    enable_consolidation: bool = True,
) -> None:
    """
    Main entry point for Psyche CLI.

    Args:
        server_command: Command to launch Elpis server
        mnemosyne_command: Command to launch Mnemosyne server (None to disable)
        debug: Enable debug logging
        workspace: Working directory for tool operations
        log_file: Path to log file (default: ~/.psyche/psyche.log)
        enable_consolidation: Enable automatic memory consolidation
    """
    # Default log file in user's home directory
    if log_file is None:
        log_dir = Path.home() / ".psyche"
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / "psyche.log")

    setup_logging(debug, log_file)

    # Create client for Elpis connection
    client = ElpisClient(server_command=server_command)

    # Create client for Mnemosyne connection (optional)
    mnemosyne_client = None
    if mnemosyne_command and enable_consolidation:
        mnemosyne_client = MnemosyneClient(server_command=mnemosyne_command)
        logger.info(f"Mnemosyne client configured: {mnemosyne_command}")

    # Configure server
    server_config = ServerConfig(
        idle_think_interval=30.0,
        emotional_modulation=True,
        workspace_dir=workspace,
        allow_idle_tools=True,  # Enable sandboxed tool use during reflection
        enable_consolidation=enable_consolidation,
    )

    # Create memory server
    server = MemoryServer(
        elpis_client=client,
        config=server_config,
        mnemosyne_client=mnemosyne_client,
    )

    # Create Textual app
    app = PsycheApp(memory_server=server)

    # Track if cleanup already happened (e.g., via action_quit)
    cleanup_done = False

    # Signal handler for graceful shutdown
    def signal_handler(signum: int, frame) -> None:
        nonlocal cleanup_done
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        # Request app exit - this will cause app.run() to return
        try:
            app.exit()
        except Exception:
            pass  # App may not be fully initialized

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    # SIGINT is handled by Textual via Ctrl+C binding

    # Redirect stderr to file to prevent any native library output from breaking TUI
    # This catches llama-cpp-python and other C library output that bypasses Python logging
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

        # Always run cleanup after app exits (regardless of how it exited)
        # This ensures memory consolidation happens even on SIGTERM or crashes
        if not cleanup_done:
            _run_async_cleanup(server)
            cleanup_done = True


if __name__ == "__main__":
    main()
