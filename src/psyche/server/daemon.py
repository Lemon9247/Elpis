"""
Psyche Daemon - Server lifecycle and connection management.

Manages:
- PsycheCore initialization with Elpis/Mnemosyne clients
- HTTP and MCP server lifecycle
- Client connection tracking
- Dream scheduling when no clients connected
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Optional, Set

import uvicorn
from loguru import logger

from psyche.core import CoreConfig, ContextConfig, MemoryHandlerConfig, PsycheCore
from psyche.mcp.client import ElpisClient, MnemosyneClient
from psyche.server.http import HTTPServerConfig, PsycheHTTPServer

if TYPE_CHECKING:
    from psyche.handlers.dream_handler import DreamHandler


@dataclass
class ServerConfig:
    """Configuration for Psyche server daemon."""

    # HTTP server
    http_host: str = "127.0.0.1"
    http_port: int = 8741

    # MCP server (future)
    mcp_enabled: bool = False

    # Elpis connection
    elpis_command: str = "elpis-server"

    # Mnemosyne connection (optional)
    mnemosyne_command: Optional[str] = "mnemosyne-server"

    # Core configuration
    core: CoreConfig = field(default_factory=CoreConfig)

    # Dreaming configuration
    dream_enabled: bool = True
    dream_delay_seconds: float = 60.0  # Wait before starting to dream

    # Model name for API
    model_name: str = "psyche"

    # Consolidation configuration
    consolidation_enabled: bool = True
    consolidation_interval: float = 300.0  # Check every 5 minutes
    consolidation_importance_threshold: float = 0.6
    consolidation_similarity_threshold: float = 0.85


class PsycheDaemon:
    """
    Psyche server daemon.

    Manages the full lifecycle of the Psyche substrate:
    - Initializes PsycheCore with Elpis and Mnemosyne
    - Runs HTTP server for external agents
    - Tracks client connections
    - Schedules dreaming when idle
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        """
        Initialize the daemon.

        Args:
            config: Server configuration
        """
        self.config = config or ServerConfig()

        # Components (initialized in start())
        self.elpis_client: Optional[ElpisClient] = None
        self.mnemosyne_client: Optional[MnemosyneClient] = None
        self.core: Optional[PsycheCore] = None
        self.http_server: Optional[PsycheHTTPServer] = None
        self.dream_handler: Optional[DreamHandler] = None

        # Connection tracking
        self._connections: Set[str] = set()
        self._dream_task: Optional[asyncio.Task] = None
        self._consolidation_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Server state
        self._running = False

    async def start(self) -> None:
        """Start the Psyche server."""
        logger.info("Starting Psyche daemon...")

        # Create MCP client objects
        self.elpis_client = ElpisClient(server_command=self.config.elpis_command)
        logger.info(f"Elpis client configured: {self.config.elpis_command}")

        if self.config.mnemosyne_command:
            self.mnemosyne_client = MnemosyneClient(
                server_command=self.config.mnemosyne_command
            )
            logger.info(f"Mnemosyne client configured: {self.config.mnemosyne_command}")

        try:
            # Use nested context managers to keep MCP servers running
            async with self._connect_elpis() as elpis:
                async with self._connect_mnemosyne() as mnemosyne:
                    # Initialize core with connected clients
                    await self._init_core_with_clients(elpis, mnemosyne)

                    # Create HTTP server
                    self._init_http_server()

                    # Create dream handler (if enabled)
                    if self.config.dream_enabled:
                        await self._init_dream_handler()

                    # Start consolidation loop (if enabled)
                    if self.config.consolidation_enabled and self.mnemosyne_client:
                        self._consolidation_task = asyncio.create_task(
                            self._consolidation_loop()
                        )
                        logger.info("Started server-side consolidation loop")

                    # Start serving
                    self._running = True
                    logger.info(
                        f"Psyche server running at http://{self.config.http_host}:{self.config.http_port}"
                    )

                    try:
                        await self._run_server()
                    finally:
                        # Shutdown BEFORE context managers exit (while Elpis/Mnemosyne still connected)
                        await self.shutdown()

        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")

    @asynccontextmanager
    async def _connect_elpis(self) -> AsyncIterator[ElpisClient]:
        """Connect to Elpis server."""
        if not self.elpis_client:
            raise RuntimeError("Elpis client not configured")

        async with self.elpis_client.connect() as client:
            yield client

    @asynccontextmanager
    async def _connect_mnemosyne(self) -> AsyncIterator[Optional[MnemosyneClient]]:
        """Connect to Mnemosyne server (optional)."""
        if not self.mnemosyne_client:
            yield None
            return

        try:
            async with self.mnemosyne_client.connect() as client:
                yield client
        except Exception as e:
            logger.warning(f"Failed to connect to Mnemosyne: {e}")
            logger.warning("Continuing without persistent memory")
            yield None

    async def _init_core_with_clients(
        self,
        elpis: ElpisClient,
        mnemosyne: Optional[MnemosyneClient],
    ) -> None:
        """Initialize PsycheCore with connected clients."""
        # Query Elpis for its capabilities to configure context correctly
        try:
            capabilities = await elpis.get_capabilities()
            elpis_context_length = capabilities.get("context_length", 4096)
            logger.info(f"Elpis context_length: {elpis_context_length}")

            # Configure Psyche's context to fit within Elpis's window
            # Leave room for response generation (reserve ~25% for output)
            max_context = int(elpis_context_length * 0.75)
            reserve = int(elpis_context_length * 0.20)

            # Update the config's context settings
            self.config.core.context.max_context_tokens = max_context
            self.config.core.context.reserve_tokens = reserve
            logger.info(
                f"Context configured: max_tokens={max_context}, reserve={reserve}"
            )
        except Exception as e:
            logger.warning(f"Failed to query Elpis capabilities: {e}")
            logger.warning("Using default context settings")

        self.core = PsycheCore(
            elpis_client=elpis,
            mnemosyne_client=mnemosyne,
            config=self.config.core,
        )
        self.core.initialize()
        logger.info("PsycheCore initialized")

    def _init_http_server(self) -> None:
        """Initialize HTTP server."""
        http_config = HTTPServerConfig(
            host=self.config.http_host,
            port=self.config.http_port,
            model_name=self.config.model_name,
        )

        self.http_server = PsycheHTTPServer(
            core=self.core,
            daemon=self,
            config=http_config,
        )
        logger.info("HTTP server initialized")

    async def _init_dream_handler(self) -> None:
        """Initialize dream handler."""
        try:
            from psyche.handlers.dream_handler import DreamHandler

            self.dream_handler = DreamHandler(core=self.core)
            logger.info("Dream handler initialized")
        except ImportError:
            logger.warning("DreamHandler not available yet - dreaming disabled")
            self.dream_handler = None

    async def _run_server(self) -> None:
        """Run the HTTP server."""
        config = uvicorn.Config(
            self.http_server.app,
            host=self.config.http_host,
            port=self.config.http_port,
            log_level="warning",
        )
        server = uvicorn.Server(config)

        # Run until shutdown requested
        await server.serve()

    async def _consolidation_loop(self) -> None:
        """
        Periodic memory consolidation loop.

        Runs server-side to consolidate memories independently of client connections.
        """
        logger.info(f"Consolidation loop started (interval: {self.config.consolidation_interval}s)")

        try:
            while self._running:
                await asyncio.sleep(self.config.consolidation_interval)

                if not self._running:
                    break

                await self._maybe_consolidate()

        except asyncio.CancelledError:
            logger.debug("Consolidation loop cancelled")
        except Exception as e:
            logger.error(f"Consolidation loop error: {e}")

    async def _maybe_consolidate(self) -> bool:
        """
        Check and run memory consolidation if needed.

        Returns:
            True if consolidation was run, False otherwise
        """
        if not self.mnemosyne_client or not self.mnemosyne_client.is_connected:
            return False

        try:
            # Check if consolidation is recommended
            should_consolidate, reason, short_term, long_term = (
                await self.mnemosyne_client.should_consolidate()
            )

            if not should_consolidate:
                logger.debug(f"Consolidation not needed: {reason}")
                return False

            logger.info(f"Starting memory consolidation: {reason}")

            # Run consolidation
            result = await self.mnemosyne_client.consolidate_memories(
                importance_threshold=self.config.consolidation_importance_threshold,
                similarity_threshold=self.config.consolidation_similarity_threshold,
            )

            logger.info(
                f"Consolidation complete: promoted {result.memories_promoted}, "
                f"archived {result.memories_archived}, "
                f"clusters formed: {result.clusters_formed}"
            )

            return True

        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            return False

    # --- Connection Tracking ---

    def on_client_connect(self, client_id: str) -> None:
        """Called when a client connects."""
        self._connections.add(client_id)
        logger.info(f"Client connected: {client_id} (total: {len(self._connections)})")

        # Cancel any pending dream
        self._cancel_dreaming()

    def on_client_disconnect(self, client_id: str) -> None:
        """Called when a client disconnects."""
        self._connections.discard(client_id)
        logger.info(f"Client disconnected: {client_id} (total: {len(self._connections)})")

        # Maybe start dreaming
        if not self._connections and self.config.dream_enabled:
            self._schedule_dreaming()

    def _schedule_dreaming(self) -> None:
        """Schedule dreaming after delay if still no clients."""
        if self._dream_task is not None:
            return  # Already scheduled

        async def delayed_dream():
            logger.debug(
                f"Dreaming scheduled in {self.config.dream_delay_seconds}s..."
            )
            await asyncio.sleep(self.config.dream_delay_seconds)

            if not self._connections and self.dream_handler:
                logger.info("No clients connected - entering dream state")
                await self.dream_handler.start_dreaming()

        self._dream_task = asyncio.create_task(delayed_dream())

    def _cancel_dreaming(self) -> None:
        """Cancel pending or active dreaming."""
        if self._dream_task:
            self._dream_task.cancel()
            self._dream_task = None

        if self.dream_handler:
            self.dream_handler.stop_dreaming()

    # --- Lifecycle ---

    @property
    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self._connections)

    @property
    def is_dreaming(self) -> bool:
        """Check if currently dreaming."""
        return self.dream_handler is not None and self.dream_handler.is_dreaming

    async def shutdown(self) -> None:
        """Graceful shutdown with memory consolidation."""
        if not self._running:
            return  # Already shut down

        print("Shutting down Psyche...")
        logger.info("Shutting down Psyche daemon...")
        self._running = False

        # Cancel dreaming
        self._cancel_dreaming()

        # Cancel consolidation loop
        if self._consolidation_task and not self._consolidation_task.done():
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
            self._consolidation_task = None

        # Shutdown core (consolidate memories)
        if self.core:
            print("  Consolidating memories...")
            try:
                await self.core.shutdown()
                print("  Memories consolidated.")
            except Exception as e:
                logger.error(f"Error during core shutdown: {e}")
                print(f"  Warning: Memory consolidation failed: {e}")

        # Signal shutdown
        self._shutdown_event.set()

        logger.info("Psyche daemon shutdown complete")
