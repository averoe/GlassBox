"""
GlassBox RAG - A transparent, modular RAG framework for AI/ML applications.

High-level API for working with the GlassBox engine.
"""

__version__ = "0.1.0"
__author__ = "GlassBox Team"

from glassbox_rag.core.engine import GlassBoxEngine
from glassbox_rag.trace.tracker import TraceTracker

__all__ = [
    "GlassBoxEngine",
    "TraceTracker",
]
