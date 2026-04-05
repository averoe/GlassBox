"""Integration tests for GlassBox RAG engine."""

import pytest
from glassbox_rag import GlassBoxEngine, GlassBoxConfig


class TestGlassBoxEngine:
    """Integration tests for the main engine."""

    @pytest.fixture
    async def engine(self):
        """Create and initialize a test engine with defaults."""
        config = GlassBoxConfig(
            # Use SQLite for local testing
            database={"type": "sqlite", "sqlite": {"path": ":memory:"}},
            # Disable vector store for unit-level integration tests
            vector_store={"type": "chroma", "chroma": None},
        )
        eng = GlassBoxEngine(config)
        await eng.initialize()
        yield eng
        await eng.shutdown()

    def test_engine_construction(self):
        """Engine can be constructed without async init."""
        config = GlassBoxConfig()
        engine = GlassBoxEngine(config)
        assert engine.encoding_layer is not None
        assert engine.metrics_tracker is not None
        assert engine.trace_tracker is not None
        assert engine.chunker is not None
        assert engine._initialized is False

    @pytest.mark.asyncio
    async def test_engine_lifecycle(self):
        """Engine initializes and shuts down cleanly."""
        config = GlassBoxConfig()
        engine = GlassBoxEngine(config)
        await engine.initialize()
        assert engine._initialized is True
        await engine.shutdown()
        assert engine._initialized is False

    @pytest.mark.asyncio
    async def test_retrieve_requires_init(self):
        """Calling retrieve before initialize raises EngineError."""
        from glassbox_rag.core.engine import EngineError

        config = GlassBoxConfig()
        engine = GlassBoxEngine(config)

        with pytest.raises(EngineError, match="not initialized"):
            await engine.retrieve("test query")

    @pytest.mark.asyncio
    async def test_ingest_validates_documents(self):
        """Ingest rejects empty documents or missing content."""
        from glassbox_rag.core.engine import EngineError

        config = GlassBoxConfig()
        engine = GlassBoxEngine(config)
        await engine.initialize()

        with pytest.raises(EngineError, match="No documents"):
            await engine.ingest([])

        with pytest.raises(EngineError, match="missing 'content'"):
            await engine.ingest([{"metadata": {"key": "value"}}])

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_update_with_protection(self):
        """Test protected write-back."""
        config = GlassBoxConfig(writeback={"mode": "full"})
        engine = GlassBoxEngine(config)
        await engine.initialize()

        result = await engine.update(
            document_id="doc_001",
            content="Updated content",
            confidence_score=0.95,
        )

        assert result is not None
        assert result.success is True

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_update_rejected_low_confidence(self):
        """Test that low confidence write-back is rejected in protected mode."""
        config = GlassBoxConfig(
            writeback={
                "mode": "protected",
                "protected": {"confidence_threshold": 0.8},
            }
        )
        engine = GlassBoxEngine(config)
        await engine.initialize()

        result = await engine.update(
            document_id="doc_001",
            content="Dubious content",
            confidence_score=0.3,
        )

        assert result.success is False
        assert "threshold" in result.message

        await engine.shutdown()

    def test_trace_retrieval(self):
        """Trace retrieval returns None for nonexistent traces."""
        config = GlassBoxConfig()
        engine = GlassBoxEngine(config)
        assert engine.get_trace("nonexistent") is None
        assert engine.list_traces() == []

    @pytest.mark.asyncio
    async def test_metrics_summary(self):
        """Metrics summary works after initialization."""
        config = GlassBoxConfig()
        engine = GlassBoxEngine(config)
        await engine.initialize()

        summary = engine.get_metrics_summary()
        assert "total_requests" in summary

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_chunk_report(self):
        """Chunk report works."""
        config = GlassBoxConfig()
        engine = GlassBoxEngine(config)
        await engine.initialize()

        report = engine.get_chunk_report()
        assert "total_documents" in report

        await engine.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])