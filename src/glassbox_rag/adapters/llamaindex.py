"""
LlamaIndex adapter for GlassBox RAG.

Provides LlamaIndex-compatible QueryEngine and Retriever wrappers.

Usage:
    from glassbox_rag.adapters.llamaindex import GlassBoxQueryEngine

    query_engine = GlassBoxQueryEngine(engine=engine)
    response = await query_engine.aquery("What is RAG?")

Requires: pip install llama-index-core
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


def _check_llamaindex() -> None:
    try:
        import llama_index.core  # noqa: F401
    except ImportError:
        raise ImportError(
            "llama-index-core is required for the LlamaIndex adapter. "
            "Install it with: pip install llama-index-core"
        )


class GlassBoxLlamaRetriever:
    """
    LlamaIndex-compatible retriever backed by GlassBox.

    Returns LlamaIndex NodeWithScore objects for seamless integration
    with LlamaIndex query pipelines.
    """

    def __init__(
        self,
        engine: Any,
        top_k: int = 5,
        encoder: str | None = None,
    ) -> None:
        _check_llamaindex()
        self.engine = engine
        self.top_k = top_k
        self.encoder = encoder

    async def aretrieve(self, query_str: str) -> list[Any]:
        """Async retrieval — returns LlamaIndex NodeWithScore objects."""
        from llama_index.core.schema import NodeWithScore, TextNode

        result = await self.engine.retrieve(
            query=query_str, encoder=self.encoder, top_k=self.top_k,
        )

        nodes = []
        for doc in result.documents:
            node = TextNode(
                text=doc.content,
                id_=doc.id,
                metadata={
                    **doc.metadata,
                    "strategy": result.strategy,
                    "trace_id": result.trace_id,
                },
            )
            nodes.append(NodeWithScore(node=node, score=doc.score or 0.0))

        return nodes

    def retrieve(self, query_str: str) -> list[Any]:
        """Sync wrapper."""
        import asyncio
        return asyncio.run(self.aretrieve(query_str))


class GlassBoxQueryEngine:
    """
    LlamaIndex-compatible query engine backed by GlassBox.

    Wraps GlassBox's retrieve+generate pipeline into the
    LlamaIndex QueryEngine interface.
    """

    def __init__(
        self,
        engine: Any,
        top_k: int = 5,
        encoder: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        _check_llamaindex()
        self.engine = engine
        self.top_k = top_k
        self.encoder = encoder
        self.system_prompt = system_prompt

    async def aquery(self, query_str: str) -> Any:
        """Async query — returns a LlamaIndex Response object."""
        from llama_index.core.response.schema import Response
        from llama_index.core.schema import NodeWithScore, TextNode

        result = await self.engine.generate(
            query_str,
            encoder=self.encoder,
            top_k=self.top_k,
            system_prompt=self.system_prompt,
        )

        # Build source nodes
        source_nodes = []
        for source in result.get("sources", []):
            node = TextNode(
                text=source.get("content", ""),
                id_=source.get("id", ""),
                metadata=source.get("metadata", {}),
            )
            source_nodes.append(
                NodeWithScore(node=node, score=source.get("score", 0.0))
            )

        return Response(
            response=result.get("answer", ""),
            source_nodes=source_nodes,
            metadata={
                "trace_id": result.get("trace_id", ""),
                "generation": result.get("generation", {}),
            },
        )

    def query(self, query_str: str) -> Any:
        """Sync wrapper."""
        import asyncio
        return asyncio.run(self.aquery(query_str))


# ── Runtime ABC registration ────────────────────────────────────

try:
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.base.base_query_engine import BaseQueryEngine

    BaseRetriever.register(GlassBoxLlamaRetriever)
    BaseQueryEngine.register(GlassBoxQueryEngine)
except (ImportError, AttributeError):
    pass
