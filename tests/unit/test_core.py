"""Comprehensive unit tests for GlassBox RAG core components."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from glassbox_rag.config import (
    GlassBoxConfig,
    load_config,
    _replace_env_vars,
    ChunkingConfig,
)
from glassbox_rag.core.encoder import (
    EncoderFactory,
    ONNXEncoder,
    OllamaEncoder,
    OpenAIEncoder,
    EncoderError,
    EncoderNotAvailableError,
)
from glassbox_rag.core.metrics import (
    MetricsTracker,
    OperationType,
    LatencyHistogram,
    OperationMetrics,
)
from glassbox_rag.core.chunker import (
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    ChunkSizeMonitor,
    create_chunker,
)
from glassbox_rag.core.retriever import Document, RetrievalResult
from glassbox_rag.core.writeback import (
    WriteBackRequest,
    WriteBackResult,
    WriteBackMode,
    WriteBackManager,
)
from glassbox_rag.trace.tracker import TraceTracker, TraceLevel
from glassbox_rag.trace.visualizer import VisualDebugger


# ═══════════════════════════════════════════════════════════════════
#  Config Tests
# ═══════════════════════════════════════════════════════════════════

class TestConfig:
    """Test configuration loading and validation."""

    def test_default_config(self):
        config = GlassBoxConfig()
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 8000
        assert config.trace.enabled is True
        assert config.writeback.enabled is True
        assert config.chunking.strategy == "recursive"
        assert config.chunking.chunk_size == 512

    def test_config_override(self):
        config = GlassBoxConfig(
            server={"port": 9000},
            encoding={"default_encoder": "ollama"},
            chunking={"strategy": "sentence", "chunk_size": 1024},
        )
        assert config.server.port == 9000
        assert config.encoding.default_encoder == "ollama"
        assert config.chunking.strategy == "sentence"
        assert config.chunking.chunk_size == 1024

    def test_port_validation(self):
        with pytest.raises(Exception):
            GlassBoxConfig(server={"port": 99999})

    def test_invalid_vector_store_type(self):
        with pytest.raises(Exception):
            GlassBoxConfig(vector_store={"type": "invalid"})

    def test_invalid_writeback_mode(self):
        with pytest.raises(Exception):
            GlassBoxConfig(writeback={"mode": "invalid"})

    def test_invalid_log_level(self):
        with pytest.raises(Exception):
            GlassBoxConfig(logging={"level": "VERBOSE"})

    def test_sample_rate_bounds(self):
        with pytest.raises(Exception):
            GlassBoxConfig(trace={"sample_rate": 1.5})

    def test_env_var_replacement_simple(self):
        os.environ["TEST_VAR"] = "hello"
        result = _replace_env_vars("${TEST_VAR}")
        assert result == "hello"
        del os.environ["TEST_VAR"]

    def test_env_var_replacement_inline(self):
        os.environ["TEST_HOST"] = "myhost"
        result = _replace_env_vars("http://${TEST_HOST}:8080")
        assert result == "http://myhost:8080"
        del os.environ["TEST_HOST"]

    def test_env_var_replacement_with_default(self):
        result = _replace_env_vars("${NONEXISTENT_VAR:fallback}")
        assert result == "fallback"

    def test_env_var_replacement_nested(self):
        os.environ["DB_PASS"] = "secret"
        data = {"db": {"password": "${DB_PASS}"}}
        result = _replace_env_vars(data)
        assert result["db"]["password"] == "secret"
        del os.environ["DB_PASS"]

    def test_chunking_config_validation(self):
        with pytest.raises(Exception):
            ChunkingConfig(strategy="unknown")
        with pytest.raises(Exception):
            ChunkingConfig(chunk_size=10)  # Below minimum 64


# ═══════════════════════════════════════════════════════════════════
#  Encoder Tests
# ═══════════════════════════════════════════════════════════════════

class TestEncoderFactory:
    def test_create_onnx_encoder(self):
        encoder = EncoderFactory.create_encoder("onnx", {"model_path": "test"})
        assert isinstance(encoder, ONNXEncoder)

    def test_create_ollama_encoder(self):
        encoder = EncoderFactory.create_encoder("ollama", {"base_url": "http://localhost:11434"})
        assert isinstance(encoder, OllamaEncoder)

    def test_create_openai_encoder(self):
        encoder = EncoderFactory.create_encoder("openai", {"api_key": "test"})
        assert isinstance(encoder, OpenAIEncoder)

    def test_unknown_encoder_raises(self):
        with pytest.raises(ValueError, match="Unknown encoder type"):
            EncoderFactory.create_encoder("nonexistent", {})

    def test_register_custom_encoder(self):
        from glassbox_rag.core.encoder import BaseEncoder
        import numpy as np

        class CustomEncoder(BaseEncoder):
            @property
            def embedding_dim(self): return 128
            async def initialize(self): pass
            async def encode(self, texts): return np.zeros((len(texts), 128))
            async def encode_single(self, text): return np.zeros(128)

        EncoderFactory.register_encoder("custom", CustomEncoder)
        encoder = EncoderFactory.create_encoder("custom", {})
        assert isinstance(encoder, CustomEncoder)
        assert encoder.embedding_dim == 128

    def test_available_encoders(self):
        available = EncoderFactory.available_encoders()
        assert "openai" in available
        assert "ollama" in available
        assert "onnx" in available

    def test_openai_encoder_no_key(self):
        encoder = OpenAIEncoder({"api_key": None})
        with pytest.raises(EncoderError, match="API key"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(encoder.initialize())


# ═══════════════════════════════════════════════════════════════════
#  Chunker Tests
# ═══════════════════════════════════════════════════════════════════

class TestChunking:
    def _make_config(self, **kwargs):
        defaults = {"strategy": "fixed", "chunk_size": 100, "chunk_overlap": 20}
        defaults.update(kwargs)
        return ChunkingConfig(**defaults)

    def test_fixed_size_basic(self):
        config = self._make_config(chunk_size=50, chunk_overlap=0)
        chunker = FixedSizeChunker(config)
        text = "A" * 120
        chunks = chunker.chunk(text)
        assert len(chunks) == 3
        assert chunks[0].size == 50
        assert chunks[1].size == 50
        assert chunks[2].size == 20

    def test_fixed_size_with_overlap(self):
        config = self._make_config(chunk_size=100, chunk_overlap=20)
        chunker = FixedSizeChunker(config)
        text = "A" * 200
        chunks = chunker.chunk(text)
        assert len(chunks) > 2  # Overlap causes more chunks

    def test_fixed_size_empty(self):
        config = self._make_config()
        chunker = FixedSizeChunker(config)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_sentence_chunker(self):
        config = self._make_config(strategy="sentence", chunk_size=100, chunk_overlap=0)
        chunker = SentenceChunker(config)
        text = "First sentence. Second sentence. Third sentence here. Fourth sentence too."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert all(c.text.strip() for c in chunks)

    def test_recursive_chunker(self):
        config = self._make_config(strategy="recursive", chunk_size=100, chunk_overlap=0)
        chunker = RecursiveChunker(config)
        text = "Paragraph one content.\n\nParagraph two content.\n\nParagraph three is a bit longer to test splitting."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert all(c.text.strip() for c in chunks)

    def test_chunk_metadata(self):
        config = self._make_config(chunk_size=200, chunk_overlap=0)
        chunker = FixedSizeChunker(config)
        text = "Some content here."
        metadata = {"source": "test.txt"}
        chunks = chunker.chunk(text, metadata=metadata)
        assert chunks[0].metadata["source"] == "test.txt"

    def test_chunk_with_stats(self):
        config = self._make_config(chunk_size=50, chunk_overlap=0)
        chunker = FixedSizeChunker(config)
        text = "A" * 120
        chunks, stats = chunker.chunk_with_stats(text)
        assert stats.total_chunks == 3
        assert stats.total_chars == 120
        assert stats.min_chunk_size == 20
        assert stats.max_chunk_size == 50

    def test_create_chunker_factory(self):
        config = ChunkingConfig(strategy="recursive")
        chunker = create_chunker(config)
        assert isinstance(chunker, RecursiveChunker)

    def test_chunk_monitor(self):
        from glassbox_rag.core.chunker import Chunk

        monitor = ChunkSizeMonitor(target_size=100)
        chunks = [
            Chunk(text="A" * 90, index=0, start_char=0, end_char=90),
            Chunk(text="B" * 110, index=1, start_char=90, end_char=200),
        ]
        monitor.record(chunks)
        report = monitor.get_report()
        assert report["total_chunks"] == 2
        assert report["total_documents"] == 1


# ═══════════════════════════════════════════════════════════════════
#  Metrics Tests
# ═══════════════════════════════════════════════════════════════════

class TestMetrics:
    def test_start_end_request(self):
        config = GlassBoxConfig()
        tracker = MetricsTracker(config)
        metrics = tracker.start_request("req_001")
        assert metrics.request_id == "req_001"

        result = tracker.end_request("req_001")
        assert result.request_id == "req_001"
        assert result.end_time is not None
        assert result.total_latency_ms >= 0

    def test_operation_tracking(self):
        config = GlassBoxConfig()
        tracker = MetricsTracker(config)
        tracker.start_request("req_001")

        op = tracker.start_operation(OperationType.ENCODE)
        tracker.end_operation(op, input_tokens=100, output_tokens=50)

        assert op.total_tokens == 150
        assert op.end_time is not None
        assert op.get_latency_ms() >= 0

    def test_latency_histogram(self):
        hist = LatencyHistogram()
        for i in range(100):
            hist.record(float(i))

        assert hist.count == 100
        assert hist.mean > 0
        assert hist.p50 > 0
        assert hist.p95 > hist.p50
        assert hist.p99 >= hist.p95

    def test_metrics_summary(self):
        config = GlassBoxConfig()
        tracker = MetricsTracker(config)
        tracker.start_request("req_001")
        op = tracker.start_operation(OperationType.ENCODE)
        tracker.end_operation(op)
        tracker.end_request("req_001")

        summary = tracker.get_summary()
        assert summary["total_requests"] == 1

    def test_disabled_metrics(self):
        config = GlassBoxConfig(metrics={"enabled": False})
        tracker = MetricsTracker(config)
        metrics = tracker.start_request("req_001")
        assert metrics.request_id == "req_001"  # Still returns an object


# ═══════════════════════════════════════════════════════════════════
#  Trace Tests
# ═══════════════════════════════════════════════════════════════════

class TestTrace:
    def test_start_end_trace(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        trace = tracker.start_trace("req_001")
        assert trace.request_id == "req_001"

        ended = tracker.end_trace(trace.trace_id)
        assert ended.end_time is not None

    def test_trace_steps(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        tracker.start_trace("req_001")

        step = tracker.start_step("encode", input_data={"query": "test"})
        assert step.name == "encode"
        assert step.level == TraceLevel.INFO

        tracker.end_step(step.step_id, output_data={"dim": 384})
        assert step.end_time is not None
        assert step.output_data["dim"] == 384

    def test_nested_steps(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        trace = tracker.start_trace("req_001")

        root = tracker.start_step("root")
        child = tracker.start_step("child")
        tracker.end_step(child.step_id)
        tracker.end_step(root.step_id)
        tracker.end_trace(trace.trace_id)

        assert len(root.children) == 1
        assert root.children[0].name == "child"

    def test_trace_retrieval(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        trace = tracker.start_trace("req_001")
        tracker.end_trace(trace.trace_id)

        retrieved = tracker.get_trace(trace.trace_id)
        assert retrieved is not None
        assert retrieved.trace_id == trace.trace_id

    def test_trace_not_found(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        assert tracker.get_trace("nonexistent") is None

    def test_trace_step_error(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        tracker.start_trace("req_001")
        step = tracker.start_step("failing")
        tracker.end_step(step.step_id, error="Something broke")
        assert step.error == "Something broke"
        assert step.level == TraceLevel.ERROR

    def test_trace_visualization(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        trace = tracker.start_trace("req_001")
        root = tracker.start_step("root")
        tracker.end_step(root.step_id)
        tracker.end_trace(trace.trace_id)

        viz = trace.visualize()
        assert "root" in viz
        assert "Trace" in viz

    def test_list_traces(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        for i in range(5):
            t = tracker.start_trace(f"req_{i}")
            tracker.end_trace(t.trace_id)

        traces = tracker.list_traces(limit=3)
        assert len(traces) == 3

    def test_trace_stats(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        t = tracker.start_trace("req_001")
        tracker.end_trace(t.trace_id)

        stats = tracker.get_stats()
        assert stats["total_traces"] == 1

    def test_disabled_trace(self):
        config = GlassBoxConfig(trace={"enabled": False})
        tracker = TraceTracker(config)
        trace = tracker.start_trace("req_001")
        assert trace.trace_id == ""  # Disabled returns empty


# ═══════════════════════════════════════════════════════════════════
#  WriteBack Tests
# ═══════════════════════════════════════════════════════════════════

class TestWriteBack:
    def test_valid_request(self):
        req = WriteBackRequest(
            operation="update",
            document_id="doc_001",
            content="Hello",
            confidence_score=0.8,
        )
        assert req.operation == "update"
        assert req.confidence_score == 0.8

    def test_invalid_operation(self):
        with pytest.raises(ValueError, match="Invalid operation"):
            WriteBackRequest(
                operation="invalid",
                document_id="doc_001",
                content="Hello",
            )

    def test_empty_document_id(self):
        with pytest.raises(ValueError, match="document_id must not be empty"):
            WriteBackRequest(
                operation="update",
                document_id="",
                content="Hello",
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(ValueError, match="confidence_score must be"):
            WriteBackRequest(
                operation="update",
                document_id="doc_001",
                content="Hello",
                confidence_score=1.5,
            )

    @pytest.mark.asyncio
    async def test_read_only_rejects(self):
        config = GlassBoxConfig(writeback={"mode": "read-only"})
        manager = WriteBackManager(config)
        req = WriteBackRequest(
            operation="update", document_id="doc_001",
            content="Hello", confidence_score=1.0,
        )
        result = await manager.execute_write(req)
        assert result.success is False
        assert "read-only" in result.message

    @pytest.mark.asyncio
    async def test_disabled_rejects(self):
        config = GlassBoxConfig(writeback={"enabled": False})
        manager = WriteBackManager(config)
        req = WriteBackRequest(
            operation="update", document_id="doc_001",
            content="Hello", confidence_score=1.0,
        )
        result = await manager.execute_write(req)
        assert result.success is False
        assert "disabled" in result.message

    @pytest.mark.asyncio
    async def test_protected_low_confidence(self):
        config = GlassBoxConfig(
            writeback={"mode": "protected", "protected": {"confidence_threshold": 0.8}}
        )
        manager = WriteBackManager(config)
        req = WriteBackRequest(
            operation="update", document_id="doc_001",
            content="Hello", confidence_score=0.5,
        )
        result = await manager.execute_write(req)
        assert result.success is False
        assert "threshold" in result.message

    @pytest.mark.asyncio
    async def test_full_mode_succeeds(self):
        config = GlassBoxConfig(writeback={"mode": "full"})
        manager = WriteBackManager(config)
        req = WriteBackRequest(
            operation="update", document_id="doc_001",
            content="Hello", confidence_score=1.0,
        )
        result = await manager.execute_write(req)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_protected_review_flow(self):
        config = GlassBoxConfig(
            writeback={
                "mode": "protected",
                "protected": {"confidence_threshold": 0.8, "human_review": True}
            }
        )
        manager = WriteBackManager(config)
        req = WriteBackRequest(
            operation="update", document_id="doc_001",
            content="Hello", confidence_score=0.5,
        )
        result = await manager.execute_write(req)
        assert result.requires_review is True
        assert result.success is False

        # Approve the review
        approved = await manager.approve_review(result.operation_id)
        assert approved.success is True


# ═══════════════════════════════════════════════════════════════════
#  Visualizer Tests
# ═══════════════════════════════════════════════════════════════════

class TestVisualDebugger:
    def _make_trace(self, tracker):
        trace = tracker.start_trace("req_001")
        root = tracker.start_step("retrieve")
        child1 = tracker.start_step("encode_query")
        tracker.end_step(child1.step_id)
        child2 = tracker.start_step("search")
        tracker.end_step(child2.step_id)
        tracker.end_step(root.step_id)
        tracker.end_trace(trace.trace_id)
        return trace

    def test_waterfall(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        trace = self._make_trace(tracker)

        waterfall = VisualDebugger.format_waterfall(trace)
        assert "Waterfall" in waterfall
        assert "retrieve" in waterfall

    def test_cost_breakdown(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        trace = self._make_trace(tracker)

        breakdown = VisualDebugger.format_cost_breakdown(trace)
        assert "Cost" in breakdown

    def test_inspect_step(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        trace = self._make_trace(tracker)

        result = VisualDebugger.inspect_step(trace, "encode_query")
        assert result is not None
        assert result["name"] == "encode_query"

    def test_inspect_missing_step(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        trace = self._make_trace(tracker)

        result = VisualDebugger.inspect_step(trace, "nonexistent")
        assert result is None

    def test_filter_slow_traces(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        trace = self._make_trace(tracker)

        # Should include our trace (it has some duration)
        results = VisualDebugger.filter_slow_traces([trace], min_duration_ms=0.0)
        assert len(results) == 1

    def test_filter_error_traces(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)

        # Create trace with error
        trace = tracker.start_trace("req_001")
        step = tracker.start_step("failing")
        tracker.end_step(step.step_id, error="boom")
        tracker.end_trace(trace.trace_id)

        results = VisualDebugger.filter_error_traces([trace])
        assert len(results) == 1

    def test_trace_summary(self):
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        trace = self._make_trace(tracker)

        summary = VisualDebugger.get_trace_summary(trace)
        assert summary["root_step"] == "retrieve"
        assert summary["total_steps"] >= 3


# ═══════════════════════════════════════════════════════════════════
#  Document / RetrievalResult Tests
# ═══════════════════════════════════════════════════════════════════

class TestModels:
    def test_document_to_dict(self):
        doc = Document(id="d1", content="hello", score=0.9, metadata={"k": "v"})
        d = doc.to_dict()
        assert d["id"] == "d1"
        assert d["score"] == 0.9

    def test_retrieval_result(self):
        result = RetrievalResult(
            documents=[],
            query="test",
            strategy="semantic",
            execution_time_ms=12.5,
            total_results=0,
        )
        d = result.to_dict()
        assert d["strategy"] == "semantic"
        assert d["execution_time_ms"] == 12.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
