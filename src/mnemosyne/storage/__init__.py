"""Memory storage backends."""

try:
    from mnemosyne.storage.chroma_store import ChromaMemoryStore
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

__all__ = []

if CHROMA_AVAILABLE:
    __all__.append("ChromaMemoryStore")
