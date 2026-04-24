"""
GlassBox RAG — A transparent, modular RAG framework.

Public API:
    from glassbox_rag import GlassBoxEngine, GlassBoxConfig, load_config
    from glassbox_rag import Document, RetrievalResult, Chunk
    from glassbox_rag import HookManager, HookPoint
    from glassbox_rag import DocumentDeduplicator
    from glassbox_rag import TokenCounter
    from glassbox_rag import LLMGenerator, StreamEvent
"""

__version__ = "1.0.0"

from glassbox_rag.config import GlassBoxConfig, load_config
from glassbox_rag.core.engine import GlassBoxEngine
from glassbox_rag.core.retriever import Document, RetrievalResult
from glassbox_rag.core.writeback import WriteBackRequest, WriteBackResult, WriteBackMode
from glassbox_rag.core.chunker import Chunk, ChunkingStats, create_chunker
from glassbox_rag.core.encoder import (
    BaseEncoder,
    EmbeddingCache,
    EncoderFactory,
    EncoderError,
    EncoderNotAvailableError,
)
from glassbox_rag.core.hooks import HookManager, HookPoint
from glassbox_rag.core.dedup import DocumentDeduplicator
from glassbox_rag.core.tokens import TokenCounter
from glassbox_rag.core.generator import StreamEvent, GenerationResult

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
    "EmbeddingCache",
    "EncoderFactory",
    "EncoderError",
    "EncoderNotAvailableError",
    # Hooks
    "HookManager",
    "HookPoint",
    # Dedup
    "DocumentDeduplicator",
    # Tokens
    "TokenCounter",
    # Generation
    "StreamEvent",
    "GenerationResult",
    # Version
    "__version__",
]
