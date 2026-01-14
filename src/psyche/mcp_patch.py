"""
Monkey-patch for MCP library bug.

Fixes: RuntimeError: dictionary keys changed during iteration
in mcp/shared/session.py _receive_loop method.

The MCP library iterates over _response_streams.items() without copying,
causing a race condition when the dictionary is modified during iteration.

This patch replaces the buggy iteration with a safe copy.
"""

import logging


def apply_mcp_patch():
    """Apply the monkey-patch to fix the MCP library race condition."""
    try:
        from mcp.shared import session

        # Store original method
        original_receive_loop = session.BaseSession._receive_loop

        async def patched_receive_loop(self):
            """Patched _receive_loop that safely iterates over response streams."""
            from anyio import ClosedResourceError, EndOfStream
            from mcp.shared.exceptions import McpError
            from mcp.shared.session import CONNECTION_CLOSED
            from mcp.types import ErrorData, JSONRPCError

            try:
                async for message in self._read_stream:
                    # This part is the same as the original
                    await self._handle_incoming(message)
            except ClosedResourceError:
                logging.debug("Read stream closed by client")
            except EndOfStream:
                logging.debug("Read stream closed by client")
            except Exception as e:
                logging.exception(f"Unhandled exception in receive loop: {e}")
            finally:
                # FIX: Iterate over a copy to prevent "dictionary keys changed during iteration"
                for id, stream in list(self._response_streams.items()):
                    error = ErrorData(code=CONNECTION_CLOSED, message="Connection closed")
                    try:
                        await stream.send(JSONRPCError(jsonrpc="2.0", id=id, error=error))
                        await stream.aclose()
                    except Exception:
                        pass
                self._response_streams.clear()

        # Apply the patch
        session.BaseSession._receive_loop = patched_receive_loop
        logging.debug("Applied MCP library patch for _response_streams iteration bug")

    except Exception as e:
        logging.warning(f"Failed to apply MCP patch: {e}")
