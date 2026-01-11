"""Logging configuration using loguru."""

import sys
from pathlib import Path

from loguru import logger


def configure_logging(logging_settings) -> None:
    """Configure loguru logging with settings."""
    # Remove default handler
    logger.remove()

    # Add stdout handler with level
    logger.add(
        sys.stderr,
        level=logging_settings.level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # Add file handler if output file specified
    if logging_settings.output_file:
        log_file = Path(logging_settings.output_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        if logging_settings.format == "json":
            logger.add(
                log_file,
                level=logging_settings.level,
                format="{time} {level} {message}",
                serialize=True,
                rotation="10 MB",
                retention="1 week",
            )
        else:
            logger.add(
                log_file,
                level=logging_settings.level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="10 MB",
                retention="1 week",
            )
