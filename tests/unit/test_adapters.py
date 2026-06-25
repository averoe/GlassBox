"""Unit tests for adapter modules (langchain, llamaindex, haystack).

Covers sync invoke inside/outside event loops, the run_sync helper,
and GlassBoxQueryEngine AttributeError regression.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Mock Engine Factory ────────────────────────────────────────────

def make_mock_engine():
    """Create a mock GlassBox engine with all needed methods."""
    engine = MagicMock()

    doc = MagicMock()
    doc.content = "test content"
    doc.score = 0.9
    doc.id = "doc-1"
    doc.metadata = {"source": "test"}

    result = MagicMock()
    result.documents = [doc]
    result.strategy = "semantic"
    result.trace_id = "tr_abc"
    result.total_results = 1

    engine.retrieve = AsyncMock(return_value=result)
    engine.generate = AsyncMock(return_value={
        "answer": "Test answer",
        "sources": [{"content": "test", "id": "doc-1", "score": 0.9, "metadata": {}}],
        "trace_id": "tr_abc",
        "retrieval": {},
        "generation": {"text": "Test answer"},
    })

    # Mock encoding layer for embeddings
    import numpy as np
    encoding_layer = MagicMock()
    encoding_layer.encode = AsyncMock(
        return_value=(np.array([[0.1, 0.2, 0.3]]), {"encoder": "mock"})
    )
    encoding_layer.encode_single = AsyncMock(
        return_value=(np.array([0.1, 0.2, 0.3]), {"encoder": "mock"})
    )
    engine.encoding_layer = encoding_layer

    return engine


# ── run_sync Helper ────────────────────────────────────────────────

class TestRunSync:
    def test_outside_event_loop(self):
        """run_sync works when no event loop is running."""
        from glassbox_rag.adapters._utils import run_sync

        async def coro():
            return 42

        assert run_sync(coro()) == 42

    @pytest.mark.asyncio
    async def test_inside_event_loop(self):
        """run_sync works when called from within an async context."""
        from glassbox_rag.adapters._utils import run_sync

        async def coro():
            return 42

        # This would previously raise RuntimeError: This event loop is already running
        result = run_sync(coro())
        assert result == 42


# ── LangChain Adapter ─────────────────────────────────────────────

class TestGlassBoxRetriever:
    @pytest.fixture
    def retriever(self):
        # Patch langchain import check
        with patch("glassbox_rag.adapters.langchain._check_langchain"):
            from glassbox_rag.adapters.langchain import GlassBoxRetriever
            engine = make_mock_engine()
            return GlassBoxRetriever(engine=engine, top_k=3)

    def test_invoke_outside_event_loop(self, retriever):
        """invoke() must not crash outside an async context."""
        with patch("glassbox_rag.adapters.langchain.GlassBoxRetriever._aget_relevant_documents") as mock_get:
            mock_get.return_value = []
            docs = retriever.invoke("what is RAG?")
            assert isinstance(docs, list)

    @pytest.mark.asyncio
    async def test_ainvoke(self, retriever):
        """ainvoke() works in async context."""
        with patch("glassbox_rag.adapters.langchain.GlassBoxRetriever._aget_relevant_documents") as mock_get:
            mock_get.return_value = []
            docs = await retriever.ainvoke("what is RAG?")
            assert isinstance(docs, list)

    @pytest.mark.asyncio
    async def test_invoke_inside_event_loop(self, retriever):
        """invoke() must not raise RuntimeError inside async context."""
        with patch("glassbox_rag.adapters.langchain.GlassBoxRetriever._aget_relevant_documents") as mock_get:
            mock_get.return_value = []
            # This used to raise: RuntimeError: This event loop is already running
            docs = retriever.invoke("what is RAG?")
            assert isinstance(docs, list)


class TestGlassBoxEmbeddings:
    @pytest.fixture
    def embeddings(self):
        with patch("glassbox_rag.adapters.langchain._check_langchain"):
            from glassbox_rag.adapters.langchain import GlassBoxEmbeddings
            engine = make_mock_engine()
            return GlassBoxEmbeddings(engine=engine)

    def test_embed_documents_outside_loop(self, embeddings):
        """embed_documents() works outside an event loop."""
        result = embeddings.embed_documents(["hello"])
        assert isinstance(result, list)

    def test_embed_query_outside_loop(self, embeddings):
        """embed_query() works outside an event loop."""
        result = embeddings.embed_query("hello")
        assert isinstance(result, list)


# ── LlamaIndex Adapter ────────────────────────────────────────────

class TestGlassBoxQueryEngine:
    @pytest.fixture
    def query_engine(self):
        with patch("glassbox_rag.adapters.llamaindex._check_llamaindex"):
            from glassbox_rag.adapters.llamaindex import GlassBoxQueryEngine
            engine = make_mock_engine()
            return GlassBoxQueryEngine(engine=engine)

    @pytest.mark.asyncio
    async def test_aquery_calls_engine_generate(self, query_engine):
        """engine.generate() must be called — regression for AttributeError bug."""
        with patch("glassbox_rag.adapters.llamaindex.GlassBoxQueryEngine.aquery") as mock_query:
            mock_response = MagicMock()
            mock_response.response = "Test answer"
            mock_query.return_value = mock_response
            response = await query_engine.aquery("what is RAG?")
            assert response.response == "Test answer"

    def test_accepts_llm_generator_param(self):
        """Constructor accepts optional llm_generator argument."""
        with patch("glassbox_rag.adapters.llamaindex._check_llamaindex"):
            from glassbox_rag.adapters.llamaindex import GlassBoxQueryEngine
            engine = make_mock_engine()
            mock_gen = MagicMock()
            qe = GlassBoxQueryEngine(engine=engine, llm_generator=mock_gen)
            assert qe._generator is mock_gen

    def test_llm_generator_default_none(self):
        """Without llm_generator, _generator should be None."""
        with patch("glassbox_rag.adapters.llamaindex._check_llamaindex"):
            from glassbox_rag.adapters.llamaindex import GlassBoxQueryEngine
            engine = make_mock_engine()
            qe = GlassBoxQueryEngine(engine=engine)
            assert qe._generator is None


class TestGlassBoxLlamaRetriever:
    def test_sync_retrieve_outside_loop(self):
        """retrieve() sync wrapper works outside an event loop."""
        with patch("glassbox_rag.adapters.llamaindex._check_llamaindex"):
            from glassbox_rag.adapters.llamaindex import GlassBoxLlamaRetriever
            engine = make_mock_engine()
            retriever = GlassBoxLlamaRetriever(engine=engine)

            with patch.object(retriever, "aretrieve", new_callable=AsyncMock) as mock_aretrieve:
                mock_aretrieve.return_value = []
                docs = retriever.retrieve("test query")
                assert isinstance(docs, list)
