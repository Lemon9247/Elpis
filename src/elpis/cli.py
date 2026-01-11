"""
CLI entry point for Elpis agent.

This module wires together all Phase 1 components:
- Settings loading and configuration
- Hardware detection and logging setup
- LLM inference engine initialization
- Tool engine setup
- Agent orchestrator creation
- REPL interface launch
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from elpis.agent.orchestrator import AgentOrchestrator
from elpis.agent.repl import ElpisREPL
from elpis.config.settings import Settings
from elpis.llm.inference import LlamaInference
from elpis.tools.tool_engine import ToolEngine
from elpis.utils.logging import configure_logging


@click.command()
@click.option(
    "--config",
    default="configs/config.default.toml",
    help="Path to configuration file",
    type=click.Path(exists=False),
)
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option(
    "--workspace",
    default=None,
    help="Override workspace directory path",
    type=click.Path(file_okay=False),
)
def main(config: str, debug: bool, workspace: Optional[str]) -> None:
    """
    Elpis - Emotional Coding Agent (Phase 1: Basic Agent Harness).

    An AI coding assistant with LLM-powered reasoning and tool execution.
    """
    try:
        # Load settings
        settings = Settings()

        # Override settings from CLI flags
        if debug:
            settings.logging.level = "DEBUG"

        if workspace:
            settings.tools.workspace_dir = workspace

        # Configure logging
        configure_logging(settings.logging)
        logger.info("=" * 60)
        logger.info("Starting Elpis - Emotional Coding Agent")
        logger.info("=" * 60)
        logger.info(f"Configuration: {config}")
        logger.info(f"Workspace: {settings.tools.workspace_dir}")
        logger.info(f"Log level: {settings.logging.level}")

        # Validate model file exists
        model_path = Path(settings.model.path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.info("Please download the model first:")
            logger.info("  python scripts/download_model.py")
            sys.exit(1)

        logger.info(f"Model file: {model_path}")

        # Initialize LLM inference engine
        logger.info("Loading LLM model (this may take a moment)...")
        llm = LlamaInference(settings.model)
        logger.info("LLM model loaded successfully")

        # Initialize tool engine
        logger.info("Initializing tool engine...")
        tools = ToolEngine(settings.tools.workspace_dir, settings)
        logger.info(f"Tool engine ready with {len(tools.tools)} tools")

        # Initialize agent orchestrator
        logger.info("Initializing agent orchestrator...")
        agent = AgentOrchestrator(llm, tools, settings)
        logger.info("Agent orchestrator ready")

        # Create and run REPL
        logger.info("Starting REPL interface...")
        repl = ElpisREPL(agent)

        # Run async REPL
        asyncio.run(repl.run())

    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
