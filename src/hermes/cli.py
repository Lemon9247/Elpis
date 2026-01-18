"""Command-line interface for Hermes TUI."""

import asyncio
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
from psyche.mcp_patch import apply_mcp_patch

apply_mcp_patch()

# Now safe to import modules that may use logging
from hermes.app import Hermes
from psyche.core import CoreConfig, ContextConfig, MemoryHandlerConfig, PsycheCore
from psyche.handlers import (
    IdleConfig,
    IdleHandler,
    LocalPsycheClient,
    PsycheClient,
    ReactConfig,
    ReactHandler,
    RemotePsycheClient,
)
from psyche.mcp.client import ElpisClient, MnemosyneClient
from psyche.tools import ToolEngine
from psyche.tools.implementations.memory_tools import MemoryTools
from psyche.tools.tool_definitions import RecallMemoryInput, StoreMemoryInput, ToolDefinition
from psyche.tools.tool_engine import ToolSettings


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


def _run_async_cleanup(core: PsycheCore) -> None:
    """Run async cleanup in a new event loop.

    This is called after the Textual app exits to ensure memory consolidation
    happens regardless of how the app was terminated.
    """

    async def cleanup():
        try:
            logger.info("Running post-exit memory consolidation...")
            await core.shutdown()
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


def _run_async_disconnect(client: RemotePsycheClient) -> None:
    """Disconnect remote client in a new event loop."""

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


def _register_memory_tools(
    tool_engine: ToolEngine,
    mnemosyne_client: MnemosyneClient,
    elpis_client: ElpisClient,
) -> None:
    """Register memory tools with the tool engine."""
    memory_tools = MemoryTools(
        mnemosyne_client=mnemosyne_client,
        get_emotion_fn=elpis_client.get_emotion,
    )

    # Register recall_memory tool
    tool_engine.register_tool(
        ToolDefinition(
            name="recall_memory",
            description="Search and recall memories from long-term storage. Use this to remember past conversations, facts, or experiences.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant memories",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of memories to retrieve (1-20, default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            input_model=RecallMemoryInput,
            handler=memory_tools.recall_memory,
        )
    )

    # Register store_memory tool
    tool_engine.register_tool(
        ToolDefinition(
            name="store_memory",
            description="Store a new memory for later recall. Use this to remember important information, facts, or experiences.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content of the memory to store",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of the memory (auto-generated if not provided)",
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "Type of memory: episodic (events), semantic (facts), procedural (how-to), emotional (feelings)",
                        "default": "episodic",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags to categorize the memory",
                    },
                },
                "required": ["content"],
            },
            input_model=StoreMemoryInput,
            handler=memory_tools.store_memory,
        )
    )

    logger.info("Memory tools registered: recall_memory, store_memory")


def _run_local_mode(
    elpis_command: str,
    mnemosyne_command: Optional[str],
    debug: bool,
    workspace: str,
    log_file: str,
    enable_consolidation: bool,
) -> None:
    """Run Hermes in local mode with embedded PsycheCore."""
    setup_logging(debug, log_file)

    # --- Create MCP clients ---
    elpis_client = ElpisClient(server_command=elpis_command)

    mnemosyne_client = None
    if mnemosyne_command and enable_consolidation:
        mnemosyne_client = MnemosyneClient(server_command=mnemosyne_command)
        logger.info(f"Mnemosyne client configured: {mnemosyne_command}")

    # --- Create PsycheCore (memory coordination layer) ---
    # TODO: Query Elpis capabilities after connecting to get actual context_length
    # For now, use conservative defaults matching Elpis's default 4096 context
    # (75% for context, 20% reserve for response)
    core_config = CoreConfig(
        context=ContextConfig(
            max_context_tokens=3000,
            reserve_tokens=800,
            checkpoint_interval=20,
        ),
        memory=MemoryHandlerConfig(
            enable_auto_retrieval=True,
            auto_storage=True,
            auto_storage_threshold=0.6,
        ),
        reasoning_enabled=True,
        emotional_modulation=True,
    )

    core = PsycheCore(
        elpis_client=elpis_client,
        mnemosyne_client=mnemosyne_client,
        config=core_config,
    )

    # --- Create ToolEngine ---
    tool_engine = ToolEngine(
        workspace_dir=workspace,
        settings=ToolSettings(),
    )

    # Register memory tools if Mnemosyne is available
    if mnemosyne_client:
        _register_memory_tools(tool_engine, mnemosyne_client, elpis_client)

    # --- Set tool descriptions in core ---
    tool_descriptions = tool_engine.get_tool_descriptions()
    core.set_tool_descriptions(tool_descriptions)
    core.initialize()

    # --- Create handlers ---
    # Get the shared compactor from core's context manager
    compactor = core._context.compactor

    react_config = ReactConfig(
        max_tool_iterations=10,
        max_tool_result_chars=16000,
        generation_timeout=120.0,
        emotional_modulation=True,
        reasoning_enabled=True,
    )

    react_handler = ReactHandler(
        elpis_client=elpis_client,
        tool_engine=tool_engine,
        compactor=compactor,
        config=react_config,
        retrieve_memories_fn=core.retrieve_memories,
    )

    idle_config = IdleConfig(
        post_interaction_delay=30.0,
        idle_tool_cooldown_seconds=60.0,
        startup_warmup_seconds=120.0,
        max_idle_tool_iterations=3,
        think_temperature=0.9,
        generation_timeout=60.0,
        allow_idle_tools=True,
        emotional_modulation=True,
        workspace_dir=workspace,
        enable_consolidation=enable_consolidation,
        consolidation_check_interval=300.0,
    )

    idle_handler = IdleHandler(
        elpis_client=elpis_client,
        compactor=compactor,
        tool_engine=tool_engine,
        mnemosyne_client=mnemosyne_client,
        config=idle_config,
    )

    # --- Create client wrapper ---
    client = LocalPsycheClient(core)

    # --- Create Textual app ---
    app = Hermes(
        client=client,
        react_handler=react_handler,
        idle_handler=idle_handler,
        elpis_client=elpis_client,
        mnemosyne_client=mnemosyne_client,
    )

    # Track if cleanup already happened
    cleanup_done = False

    # Signal handler for graceful shutdown
    def signal_handler(signum: int, frame) -> None:
        nonlocal cleanup_done
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        try:
            app.exit()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, signal_handler)

    # Redirect stderr to file
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
        if not cleanup_done:
            _run_async_cleanup(core)
            cleanup_done = True


def _run_remote_mode(
    server_url: str,
    debug: bool,
    workspace: str,
    log_file: str,
) -> None:
    """Run Hermes in remote mode, connecting to a Psyche server."""
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

    # --- Create handlers ---
    # Note: In remote mode, we don't have direct access to the compactor
    # ReactHandler will need to work differently

    react_config = ReactConfig(
        max_tool_iterations=10,
        max_tool_result_chars=16000,
        generation_timeout=120.0,
        emotional_modulation=True,
        reasoning_enabled=True,
    )

    # For remote mode, we need a simplified setup
    # The idle handler won't work without elpis_client
    # For now, we run without idle behavior in remote mode

    # --- Create Textual app ---
    app = Hermes(
        client=client,
        react_handler=None,  # Will use client directly
        idle_handler=None,  # No idle in remote mode
        elpis_client=None,  # Not available in remote mode
        mnemosyne_client=None,
        tool_engine=tool_engine,  # For local tool execution
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


@click.command()
@click.option(
    "--server",
    "server_url",
    default=None,
    help="Connect to remote Psyche server (e.g., http://localhost:8741)",
)
@click.option(
    "--elpis-command",
    default="elpis-server",
    help="Command to launch Elpis server (local mode only)",
)
@click.option(
    "--mnemosyne-command",
    default="mnemosyne-server",
    help="Command to launch Mnemosyne server (local mode only, use 'none' to disable)",
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
@click.option(
    "--no-memory",
    is_flag=True,
    help="Disable memory consolidation (local mode only)",
)
def main(
    server_url: Optional[str],
    elpis_command: str,
    mnemosyne_command: str,
    workspace: str,
    debug: bool,
    log_file: Optional[str],
    no_memory: bool,
) -> None:
    """
    Hermes - TUI client for Psyche.

    Run in local mode (default) or connect to a remote Psyche server.

    Examples:

        # Local mode (spawns Elpis and Mnemosyne)
        hermes

        # Connect to remote Psyche server
        hermes --server http://localhost:8741

        # Local mode without memory
        hermes --no-memory
    """
    # Default log file
    if log_file is None:
        log_dir = Path.home() / ".psyche"
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / "psyche.log")

    # Handle 'none' for mnemosyne
    mnemosyne_cmd = None if mnemosyne_command.lower() == "none" else mnemosyne_command

    if server_url:
        # Remote mode
        print(f"Connecting to Psyche server at {server_url}...")
        _run_remote_mode(
            server_url=server_url,
            debug=debug,
            workspace=workspace,
            log_file=log_file,
        )
    else:
        # Local mode
        _run_local_mode(
            elpis_command=elpis_command,
            mnemosyne_command=mnemosyne_cmd,
            debug=debug,
            workspace=workspace,
            log_file=log_file,
            enable_consolidation=not no_memory,
        )


if __name__ == "__main__":
    main()
