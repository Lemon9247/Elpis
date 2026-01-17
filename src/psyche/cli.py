"""Command-line interface for Psyche headless server.

This is a placeholder for Phase 5 which will implement an HTTP/WebSocket server
for multi-client connections.

For the TUI client, use the `hermes` command instead.
"""

import sys


def main() -> None:
    """Main entry point for Psyche headless server (stub)."""
    print("Psyche Headless Server")
    print("=" * 40)
    print()
    print("The headless server is not yet implemented.")
    print("This will be added in Phase 5 to support:")
    print("  - HTTP/WebSocket connections")
    print("  - Multi-client access")
    print("  - Remote TUI connections")
    print()
    print("For now, please use the Hermes TUI client:")
    print("  $ hermes")
    print()
    sys.exit(0)


if __name__ == "__main__":
    main()
