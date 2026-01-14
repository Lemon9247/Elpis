"""CLI entry point for elpis-server command."""

import asyncio
import sys

from elpis_inference.server import initialize, run_server


def main() -> None:
    """Main entry point for elpis-server command."""
    try:
        initialize()
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\nServer stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
