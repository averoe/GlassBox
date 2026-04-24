"""Core components: engine, encoder, retriever, writeback, metrics, chunker, hooks, dedup, tokens, generator."""

from glassbox_rag.core.engine import GlassBoxEngine
from glassbox_rag.core.encoder import ModularEncodingLayer, BaseEncoder, EncoderFactory, EmbeddingCache
from glassbox_rag.core.retriever import AdaptiveRetriever, Document, RetrievalResult
from glassbox_rag.core.writeback import WriteBackManager, WriteBackRequest, WriteBackResult
from glassbox_rag.core.metrics import MetricsTracker, OperationType
from glassbox_rag.core.chunker import create_chunker, Chunk
from glassbox_rag.core.hooks import HookManager, HookPoint
from glassbox_rag.core.dedup import DocumentDeduplicator
from glassbox_rag.core.tokens import TokenCounter

__all__ = [
    "GlassBoxEngine",
    "ModularEncodingLayer",
    "BaseEncoder",
    "EncoderFactory",
    "EmbeddingCache",
    "AdaptiveRetriever",
    "Document",
    "RetrievalResult",
    "WriteBackManager",
    "WriteBackRequest",
    "WriteBackResult",
    "MetricsTracker",
    "OperationType",
    "create_chunker",
    "Chunk",
    "HookManager",
    "HookPoint",
    "DocumentDeduplicator",
    "TokenCounter",
]
