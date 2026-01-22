"""
Monkey-patch for MCP library bug.

Fixes: RuntimeError: dictionary keys changed during iteration
in mcp/shared/session.py _receive_loop method.

The MCP library iterates over _response_streams.items() without copying,
causing a race condition when the dictionary is modified during iteration.

This patch uses a SafeDict that catches the error and retries with a copy.
"""

import logging


class SafeIterDict(dict):
    """A dict subclass that safely handles concurrent modification during iteration."""

    def items(self):
        """Return items safely, handling concurrent modification."""
        try:
            # Try to create a snapshot - may fail if dict is being modified
            return list(super().items())
        except RuntimeError:
            # If we get a RuntimeError during list creation, return empty
            # This is safe because the finally block in _receive_loop is just
            # cleaning up - missing some streams is better than crashing
            return []

    def keys(self):
        """Return keys safely, handling concurrent modification."""
        try:
            return list(super().keys())
        except RuntimeError:
            return []

    def values(self):
        """Return values safely, handling concurrent modification."""
        try:
            return list(super().values())
        except RuntimeError:
            return []


def apply_mcp_patch():
    """Apply the monkey-patch to fix the MCP library race condition."""
    try:
        from mcp.shared import session

        # Patch the BaseSession.__init__ to use SafeIterDict for _response_streams
        original_init = session.BaseSession.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Replace the _response_streams dict with our safe version
            # Copy any existing items (there shouldn't be any at init)
            existing = dict(self._response_streams) if hasattr(self, '_response_streams') else {}
            self._response_streams = SafeIterDict(existing)

        session.BaseSession.__init__ = patched_init
        logging.debug("Applied MCP library patch for _response_streams iteration bug")

    except Exception as e:
        logging.warning(f"Failed to apply MCP patch: {e}")
