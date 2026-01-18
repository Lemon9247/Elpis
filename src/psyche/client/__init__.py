"""User client components for Psyche.

DEPRECATED: This module is deprecated and will be removed in a future version.

Please migrate to the new locations:
- TUI components: `hermes` package
    from hermes.app import Hermes
    from hermes.repl import HermesREPL
    from hermes.display import DisplayManager

- Handlers: `psyche.handlers` package
    from psyche.handlers import ReactHandler, ReactConfig
    from psyche.handlers import IdleHandler, IdleConfig, ThoughtEvent
    from psyche.handlers import PsycheClient, LocalPsycheClient, RemotePsycheClient
"""

import warnings

warnings.warn(
    "The psyche.client module is deprecated. "
    "Use `hermes` for TUI components and `psyche.handlers` for handlers. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
