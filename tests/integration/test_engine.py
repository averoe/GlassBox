"""Integration tests for GlassBox RAG."""

import asyncio
import pytest
from glassbox_rag import GlassBoxEngine
from glassbox_rag.config import GlassBoxConfig


class TestGlassBoxEngine:
    """Integration tests for the main engine."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        config = GlassBoxConfig()
        return GlassBoxEngine(config)

    @pytest.mark.asyncio
    async def test_ingest_documents(self, engine):
        """Test ingesting documents."""
        documents = [
            {"content": "Test document 1"},
            {"content": "Test document 2"},
        ]
        
        result = await engine.ingest(documents)
        
        assert result["success"] is True
        assert result["documents_ingested"] == 2

    @pytest.mark.asyncio
    async def test_retrieve_documents(self, engine):
        """Test document retrieval."""
        # First ingest
        await engine.ingest([
            {"content": "RAG combines retrieval with generation"},
            {"content": "Vector stores enable semantic search"},
        ])
        
        # Then retrieve
        result, trace = await engine.retrieve("What is RAG?", top_k=2)
        
        assert result.query == "What is RAG?"
        assert trace is not None
        assert "trace_id" in trace

    @pytest.mark.asyncio
    async def test_update_with_protection(self, engine):
        """Test protected write-back."""
        result = await engine.update(
            document_id="doc_001",
            content="Updated content",
            confidence_score=0.95,
        )
        
        assert result is not None
        assert hasattr(result, "success")

    def test_trace_retrieval(self, engine):
        """Test getting traces."""
        trace = engine.get_trace("nonexistent")
        assert trace is None
        
        traces = engine.list_traces()
        assert isinstance(traces, list)

    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.encoding_layer is not None
        assert engine.retriever is not None
        assert engine.metrics_tracker is not None
        assert engine.trace_tracker is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])