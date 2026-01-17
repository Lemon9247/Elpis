"""User client components for Psyche.

DEPRECATED: This module is deprecated. Please use the new locations:
- TUI components: `hermes` package (e.g., `from hermes.app import Hermes`)
- Handlers: `psyche.handlers` package (e.g., `from psyche.handlers import ReactHandler`)

This module provides backward compatibility shims that will be removed in a future version.
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "The psyche.client module is deprecated. "
    "Use `hermes` for TUI components and `psyche.handlers` for handlers. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new locations for backward compatibility
from psyche.handlers import (
    ReactHandler,
    ReactConfig,
    IdleHandler,
    IdleConfig,
    ThoughtEvent,
    PsycheClient,
    LocalPsycheClient,
    RemotePsycheClient,
)

# Legacy TUI components - import from hermes
try:
    from hermes.repl import HermesREPL as PsycheREPL
    from hermes.display import DisplayManager
    from hermes.app import Hermes as PsycheApp
except ImportError:
    # Hermes may not be installed in some configurations
    PsycheREPL = None
    DisplayManager = None
    PsycheApp = None

__all__ = [
    # Handlers (now in psyche.handlers)
    "ReactHandler",
    "ReactConfig",
    "IdleHandler",
    "IdleConfig",
    "ThoughtEvent",
    "PsycheClient",
    "LocalPsycheClient",
    "RemotePsycheClient",
    # TUI (now in hermes)
    "PsycheREPL",
    "DisplayManager",
    "PsycheApp",
]
