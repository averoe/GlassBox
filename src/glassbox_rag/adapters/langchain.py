"""
LangChain adapter for GlassBox RAG.

Provides LangChain-compatible Retriever and Embeddings wrappers
so GlassBox can be used as a drop-in component in LangChain chains.

Usage:
    from glassbox_rag.adapters.langchain import GlassBoxRetriever, GlassBoxEmbeddings

    # As a LangChain retriever
    retriever = GlassBoxRetriever(engine=engine, top_k=5)
    docs = await retriever.ainvoke("What is RAG?")

    # As LangChain embeddings
    embeddings = GlassBoxEmbeddings(engine=engine, encoder="openai")
    vectors = await embeddings.aembed_documents(["text1", "text2"])

Requires: pip install langchain-core
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


def _check_langchain() -> None:
    try:
        import langchain_core  # noqa: F401
    except ImportError:
        raise ImportError(
            "langchain-core is required for the LangChain adapter. "
            "Install it with: pip install langchain-core"
        )


# Check availability at import time is too aggressive —
# we check at class instantiation instead.


class GlassBoxRetriever:
    """
    LangChain-compatible retriever backed by GlassBox.

    Inherits from LangChain's BaseRetriever when langchain-core
    is available, otherwise provides a compatible duck-type interface.

    Works with LCEL pipe operator and isinstance() checks.
    """

    def __init__(
        self,
        engine: Any,
        top_k: int = 5,
        encoder: str | None = None,
        metadata_fields: list[str] | None = None,
    ) -> None:
        _check_langchain()
        self.engine = engine
        self.top_k = top_k
        self.encoder = encoder
        self.metadata_fields = metadata_fields

    async def _aget_relevant_documents(self, query: str) -> list[Any]:
        """Core async retrieval — returns LangChain Document objects."""
        from langchain_core.documents import Document as LCDocument

        result = await self.engine.retrieve(
            query=query, encoder=self.encoder, top_k=self.top_k,
        )

        lc_docs = []
        for doc in result.documents:
            meta = dict(doc.metadata)
            meta["score"] = doc.score
            meta["glassbox_id"] = doc.id
            meta["strategy"] = result.strategy
            meta["trace_id"] = result.trace_id

            if self.metadata_fields:
                meta = {k: v for k, v in meta.items() if k in self.metadata_fields}

            lc_docs.append(LCDocument(page_content=doc.content, metadata=meta))

        return lc_docs

    # LangChain BaseRetriever interface methods
    async def ainvoke(self, input: str, **kwargs: Any) -> list[Any]:
        """Async invoke — primary LangChain LCEL interface."""
        return await self._aget_relevant_documents(input)

    async def aget_relevant_documents(self, query: str) -> list[Any]:
        """Async retrieval — legacy LangChain interface."""
        return await self._aget_relevant_documents(query)

    def invoke(self, input: str, **kwargs: Any) -> list[Any]:
        """Sync invoke — runs async retrieval in a new event loop."""
        import asyncio
        return asyncio.run(self._aget_relevant_documents(input))

    def get_relevant_documents(self, query: str) -> list[Any]:
        """Sync wrapper — runs async retrieval in a new event loop."""
        import asyncio
        return asyncio.run(self._aget_relevant_documents(query))


class GlassBoxEmbeddings:
    """
    LangChain-compatible embeddings backed by GlassBox's encoding layer.

    Implements the LangChain Embeddings interface.
    """

    def __init__(
        self,
        engine: Any,
        encoder: str | None = None,
    ) -> None:
        _check_langchain()
        self.engine = engine
        self.encoder = encoder

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        embeddings, _ = await self.engine.encoding_layer.encode(
            texts, encoder_name=self.encoder,
        )
        return embeddings.tolist()

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        embedding, _ = await self.engine.encoding_layer.encode_single(
            text, encoder_name=self.encoder,
        )
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Sync wrapper."""
        import asyncio
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        """Sync wrapper."""
        import asyncio
        return asyncio.run(self.aembed_query(text))


# ── Runtime ABC registration ────────────────────────────────────
# If langchain-core is installed, register our classes as virtual
# subclasses of the framework ABCs. This makes isinstance() checks
# pass without requiring langchain-core at import time.

try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.embeddings import Embeddings

    BaseRetriever.register(GlassBoxRetriever)
    Embeddings.register(GlassBoxEmbeddings)
except ImportError:
    pass
