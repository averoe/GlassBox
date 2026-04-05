"""
GlassBox RAG — A transparent, modular RAG framework.

Public API:
    from glassbox_rag import GlassBoxEngine, GlassBoxConfig, load_config
    from glassbox_rag import Document, RetrievalResult, Chunk
"""

__version__ = "0.2.0"

from glassbox_rag.config import GlassBoxConfig, load_config
from glassbox_rag.core.engine import GlassBoxEngine
from glassbox_rag.core.retriever import Document, RetrievalResult
from glassbox_rag.core.writeback import WriteBackRequest, WriteBackResult, WriteBackMode
from glassbox_rag.core.chunker import Chunk, ChunkingStats, create_chunker
from glassbox_rag.core.encoder import (
    BaseEncoder,
    EncoderFactory,
    EncoderError,
    EncoderNotAvailableError,
)

__all__ = [
    # Core
    "GlassBoxEngine",
    "GlassBoxConfig",
    "load_config",
    # Retrieval
    "Document",
    "RetrievalResult",
    # Write-back
    "WriteBackRequest",
    "WriteBackResult",
    "WriteBackMode",
    # Chunking
    "Chunk",
    "ChunkingStats",
    "create_chunker",
    # Encoding
    "BaseEncoder",
    "EncoderFactory",
    "EncoderError",
    "EncoderNotAvailableError",
    # Version
    "__version__",
]
