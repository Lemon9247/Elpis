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
from psyche.client.psyche_client import LocalPsycheClient
from psyche.client.react_handler import ReactHandler, ReactConfig
from psyche.client.idle_handler import IdleHandler, IdleConfig
from psyche.core import PsycheCore, CoreConfig, ContextConfig, MemoryHandlerConfig
from psyche.mcp.client import ElpisClient, MnemosyneClient
from psyche.tools import ToolEngine
from psyche.tools.tool_engine import ToolSettings
from psyche.tools.tool_definitions import ToolDefinition, RecallMemoryInput, StoreMemoryInput
from psyche.tools.implementations.memory_tools import MemoryTools


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
    tool_engine.register_tool(ToolDefinition(
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
    ))

    # Register store_memory tool
    tool_engine.register_tool(ToolDefinition(
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
    ))

    logger.info("Memory tools registered: recall_memory, store_memory")


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

    # --- Create MCP clients ---
    elpis_client = ElpisClient(server_command=server_command)

    mnemosyne_client = None
    if mnemosyne_command and enable_consolidation:
        mnemosyne_client = MnemosyneClient(server_command=mnemosyne_command)
        logger.info(f"Mnemosyne client configured: {mnemosyne_command}")

    # --- Create PsycheCore (memory coordination layer) ---
    core_config = CoreConfig(
        context=ContextConfig(
            max_context_tokens=24000,
            reserve_tokens=4000,
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
    app = PsycheApp(
        client=client,
        react_handler=react_handler,
        idle_handler=idle_handler,
        elpis_client=elpis_client,
        mnemosyne_client=mnemosyne_client,
    )

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
            _run_async_cleanup(core)
            cleanup_done = True


if __name__ == "__main__":
    main()
