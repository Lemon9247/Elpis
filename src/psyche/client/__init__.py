"""User client components for Psyche.

DEPRECATED: This module is deprecated and will be removed in a future version.

Please migrate to the new locations:
- TUI components: `hermes` package
    from hermes.app import Hermes
    from hermes.repl import HermesREPL
    from hermes.display import DisplayManager

- Handlers: `hermes.handlers` package (moved from psyche.handlers)
    from hermes.handlers import ReactHandler, ReactConfig
    from hermes.handlers import IdleHandler, IdleConfig, ThoughtEvent
    from hermes.handlers import PsycheClient, LocalPsycheClient, RemotePsycheClient
"""

import warnings

warnings.warn(
    "The psyche.client module is deprecated. "
    "Use `hermes` for TUI components and `hermes.handlers` for handlers. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
