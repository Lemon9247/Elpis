"""Interactive REPL for Hermes.

DEPRECATED: This module is deprecated. Please use the Hermes TUI instead:

    from hermes import Hermes

    # Or launch via CLI:
    $ hermes

The REPL was designed for the legacy MemoryServer architecture. The new
architecture uses PsycheClient, ReactHandler, and IdleHandler with the
Textual-based TUI.
"""

import warnings

warnings.warn(
    "hermes.repl is deprecated. Use the Hermes TUI (`hermes` command) instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)


class HermesREPL:
    """Deprecated REPL class. Use Hermes TUI instead."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "HermesREPL is deprecated. Use the Hermes TUI instead:\n"
            "  from hermes import Hermes\n"
            "  # Or launch via CLI: hermes"
        )
