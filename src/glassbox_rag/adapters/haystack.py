"""
Haystack adapter for GlassBox RAG.

Provides a Haystack-compatible Retriever component that can be
used in Haystack pipelines.

Usage:
    from glassbox_rag.adapters.haystack import GlassBoxHaystackRetriever

    retriever = GlassBoxHaystackRetriever(engine=engine)
    result = await retriever.run(query="What is RAG?")

Requires: pip install haystack-ai
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


def _check_haystack() -> None:
    try:
        import haystack  # noqa: F401
    except ImportError:
        raise ImportError(
            "haystack-ai is required for the Haystack adapter. "
            "Install it with: pip install haystack-ai"
        )


def _maybe_component(cls: type) -> type:
    """Apply Haystack @component decorator if available."""
    try:
        from haystack import component
        return component(cls)
    except (ImportError, Exception):
        return cls


@_maybe_component
class GlassBoxHaystackRetriever:
    """
    Haystack-compatible retriever backed by GlassBox.

    Decorated with @component when haystack-ai is installed,
    allowing it to be used in Haystack Pipeline.add_component().
    """

    def __init__(
        self,
        engine: Any,
        top_k: int = 5,
        encoder: str | None = None,
    ) -> None:
        _check_haystack()
        self.engine = engine
        self.top_k = top_k
        self.encoder = encoder

    async def run(
        self,
        query: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Run the retriever and return Haystack-compatible output.

        Returns:
            Dict with 'documents' key containing Haystack Document objects.
        """
        from haystack import Document as HDocument

        result = await self.engine.retrieve(
            query=query,
            encoder=self.encoder,
            top_k=top_k or self.top_k,
        )

        h_docs = []
        for doc in result.documents:
            h_docs.append(HDocument(
                id=doc.id,
                content=doc.content,
                score=doc.score,
                meta={
                    **doc.metadata,
                    "strategy": result.strategy,
                    "trace_id": result.trace_id,
                },
            ))

        return {"documents": h_docs}

    def warm_up(self) -> None:
        """Haystack lifecycle: warm up (no-op for GlassBox)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize component for Haystack pipeline YAML export."""
        return {
            "type": "glassbox_rag.adapters.haystack.GlassBoxHaystackRetriever",
            "init_parameters": {
                "top_k": self.top_k,
                "encoder": self.encoder,
            },
        }
