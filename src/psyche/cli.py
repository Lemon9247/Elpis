"""
Psyche Server CLI - Persistent AI substrate with memory.

Launches the Psyche server daemon which provides:
- HTTP endpoint at /v1/chat/completions (OpenAI-compatible)
- Server-side memory tool execution (recall_memory, store_memory)
- Context window synchronization with Elpis
- Dream state when no clients connected

Usage:
    psyche-server                    # Start server on localhost:8741
    psyche-server --port 8080        # Custom port
    psyche-server --debug            # Enable debug logging

The server spawns Elpis and Mnemosyne as MCP subprocesses.

For the TUI client, use the `hermes` command instead.
For remote TUI access, use `hermes --server http://localhost:8741`.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger


def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> None:
    """Configure logging for the server."""
    logger.remove()
    level = "DEBUG" if debug else "INFO"

    if log_file:
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation="10 MB",
        )
    else:
        logger.add(
            sys.stderr,
            level=level,
            format="<level>{level: <8}</level> | {message}",
        )


@click.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind the server to",
)
@click.option(
    "--port",
    default=8741,
    type=int,
    help="Port for the HTTP server",
)
@click.option(
    "--elpis-command",
    default="elpis-server",
    help="Command to launch Elpis MCP server",
)
@click.option(
    "--mnemosyne-command",
    default="mnemosyne-server",
    help="Command to launch Mnemosyne MCP server (use 'none' to disable)",
)
@click.option(
    "--model-name",
    default="psyche",
    help="Model name to report in API responses",
)
@click.option(
    "--dream/--no-dream",
    default=True,
    help="Enable dreaming when no clients connected",
)
@click.option(
    "--dream-delay",
    default=60.0,
    type=float,
    help="Seconds to wait before entering dream state",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
@click.option(
    "--log-file",
    type=click.Path(),
    help="Path to log file (default: stderr)",
)
def main(
    host: str,
    port: int,
    elpis_command: str,
    mnemosyne_command: str,
    model_name: str,
    dream: bool,
    dream_delay: float,
    debug: bool,
    log_file: Optional[str],
) -> None:
    """
    Launch Psyche server daemon.

    The server provides an OpenAI-compatible API at /v1/chat/completions
    that external agents (Aider, OpenCode, Continue, etc.) can connect to.

    Examples:

        # Start with defaults
        psyche-server

        # Start on a different port
        psyche-server --port 9000

        # Start without Mnemosyne (no persistent memory)
        psyche-server --mnemosyne-command none

        # Disable dreaming
        psyche-server --no-dream
    """
    # Setup logging
    if log_file is None and not debug:
        log_dir = Path.home() / ".psyche"
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / "psyche-server.log")

    setup_logging(debug, log_file if not debug else None)

    # Apply MCP patch before imports
    from psyche.mcp_patch import apply_mcp_patch
    apply_mcp_patch()

    # Import after patch
    from psyche.config.settings import ContextSettings, MemorySettings
    from psyche.core import CoreConfig, ContextConfig, MemoryHandlerConfig
    from psyche.server.daemon import PsycheDaemon, ServerConfig

    # Handle 'none' for optional mnemosyne
    mnemosyne_cmd = None if mnemosyne_command.lower() == "none" else mnemosyne_command

    # Load settings from environment (can be overridden via PSYCHE_* env vars)
    context_settings = ContextSettings()
    memory_settings = MemorySettings()

    # Build configuration
    # Note: These context defaults are overridden after connecting to Elpis
    # by querying its actual context_length (see daemon._init_core_with_clients)
    core_config = CoreConfig(
        context=ContextConfig(
            max_context_tokens=context_settings.max_context_tokens,
            reserve_tokens=context_settings.reserve_tokens,
            checkpoint_interval=context_settings.checkpoint_interval,
        ),
        memory=MemoryHandlerConfig(
            enable_auto_retrieval=memory_settings.enable_auto_retrieval,
            auto_storage=memory_settings.auto_storage,
            auto_storage_threshold=memory_settings.auto_storage_threshold,
        ),
        reasoning_enabled=True,
        emotional_modulation=True,
    )

    server_config = ServerConfig(
        http_host=host,
        http_port=port,
        elpis_command=elpis_command,
        mnemosyne_command=mnemosyne_cmd,
        core=core_config,
        dream_enabled=dream,
        dream_delay_seconds=dream_delay,
        model_name=model_name,
    )

    # Create daemon
    daemon = PsycheDaemon(config=server_config)

    # Signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler(signum: int, frame) -> None:
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating shutdown...")
        loop.create_task(daemon.shutdown())

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Print startup message
    print()
    print("=" * 50)
    print("  Psyche Server")
    print("=" * 50)
    print()
    print(f"  URL:       http://{host}:{port}")
    print(f"  Model:     {model_name}")
    print(f"  Elpis:     {elpis_command}")
    print(f"  Mnemosyne: {mnemosyne_cmd or 'disabled'}")
    print(f"  Dreaming:  {'enabled' if dream else 'disabled'}")
    print()
    print("-" * 50)
    print("  Press Ctrl+C to shutdown gracefully")
    print("  (memories will be consolidated on shutdown)")
    print("-" * 50)
    print()

    # Run
    try:
        print()
        loop.run_until_complete(daemon.start())
    except KeyboardInterrupt:
        pass  # Shutdown already handled inside daemon.start()
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)
    finally:
        loop.close()
        print()
        print("Goodbye.")
        print()


if __name__ == "__main__":
    main()
