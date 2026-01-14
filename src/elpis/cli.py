"""CLI entry point for elpis-server command."""

import os

# Force single-threaded OpenMP/BLAS BEFORE any libraries are imported
# This prevents SIGSEGV race conditions in ggml's multi-threaded CPU code
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Disable CUDA graph optimization to prevent crashes with multi-threaded access
# See: https://github.com/ggml-org/llama.cpp/issues/11804
# llama_context is not thread-safe; disabling CUDA graphs prevents race conditions
# when streaming uses a separate thread for inference
os.environ.setdefault("GGML_CUDA_DISABLE_GRAPHS", "1")

import asyncio
import sys

from elpis.server import initialize, run_server


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
