"""MCP server - backward compatibility wrapper.

This module re-exports from elpis_inference.server for backward compatibility.
All new code should import directly from elpis_inference.
"""

from elpis_inference.server import *  # noqa: F403, F401
