"""Core components: engine, encoder, retriever, writeback, metrics, chunker."""

from glassbox_rag.core.engine import GlassBoxEngine
from glassbox_rag.core.encoder import ModularEncodingLayer, BaseEncoder, EncoderFactory
from glassbox_rag.core.retriever import AdaptiveRetriever, Document, RetrievalResult
from glassbox_rag.core.writeback import WriteBackManager, WriteBackRequest, WriteBackResult
from glassbox_rag.core.metrics import MetricsTracker, OperationType
from glassbox_rag.core.chunker import create_chunker, Chunk

__all__ = [
    "GlassBoxEngine",
    "ModularEncodingLayer",
    "BaseEncoder",
    "EncoderFactory",
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
]
