"""Unit tests for GlassBox RAG core components."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from glassbox_rag.config import GlassBoxConfig
from glassbox_rag.core.encoder import EncoderFactory, ONNXEncoder, OllamaEncoder, OpenAIEncoder
from glassbox_rag.core.metrics import MetricsTracker, OperationType
from glassbox_rag.trace.tracker import TraceTracker, TraceLevel


class TestEncoderFactory:
    """Test encoder factory."""

    def test_create_onnx_encoder(self):
        """Test creating ONNX encoder."""
        config = {"model_path": "models/test"}
        encoder = EncoderFactory.create_encoder("onnx", config)
        assert isinstance(encoder, ONNXEncoder)

    def test_create_ollama_encoder(self):
        """Test creating Ollama encoder."""
        config = {"base_url": "http://localhost:11434"}
        encoder = EncoderFactory.create_encoder("ollama", config)
        assert isinstance(encoder, OllamaEncoder)

    def test_create_openai_encoder(self):
        """Test creating OpenAI encoder."""
        config = {"api_key": "test-key"}
        encoder = EncoderFactory.create_encoder("openai", config)
        assert isinstance(encoder, OpenAIEncoder)

    def test_unknown_encoder(self):
        """Test error on unknown encoder."""
        with pytest.raises(ValueError):
            EncoderFactory.create_encoder("unknown", {})


class TestMetricsTracker:
    """Test metrics tracking."""

    def test_start_request(self):
        """Test starting request tracking."""
        config = GlassBoxConfig()
        tracker = MetricsTracker(config)
        
        metrics = tracker.start_request("req_001")
        assert metrics.request_id == "req_001"
        assert tracker.current_request_metrics is metrics

    def test_end_request(self):
        """Test ending request tracking."""
        config = GlassBoxConfig()
        tracker = MetricsTracker(config)
        
        tracker.start_request("req_001")
        metrics = tracker.end_request("req_001")
        
        assert metrics.request_id == "req_001"
        assert metrics.end_time is not None
        assert "req_001" in tracker.request_history

    def test_operation_tracking(self):
        """Test operation tracking."""
        config = GlassBoxConfig()
        tracker = MetricsTracker(config)
        
        tracker.start_request("req_001")
        operation = tracker.start_operation(OperationType.ENCODE)
        
        # Simulate operation
        tracker.end_operation(operation, input_tokens=100, output_tokens=50)
        
        assert operation.total_tokens == 150
        assert operation.end_time is not None


class TestTraceTracker:
    """Test trace tracking."""

    def test_start_trace(self):
        """Test starting a trace."""
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        
        trace = tracker.start_trace("req_001")
        assert trace.request_id == "req_001"
        assert tracker.current_trace is trace

    def test_trace_steps(self):
        """Test trace step management."""
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        
        tracker.start_trace("req_001")
        step = tracker.start_step("test_operation")
        
        assert step.name == "test_operation"
        assert step.level == TraceLevel.INFO
        
        tracker.end_step(step.step_id)
        assert step.end_time is not None

    def test_trace_retrieval(self):
        """Test retrieving traces."""
        config = GlassBoxConfig()
        tracker = TraceTracker(config)
        
        trace = tracker.start_trace("req_001")
        tracker.end_trace(trace.trace_id)
        
        retrieved = tracker.get_trace(trace.trace_id)
        assert retrieved is not None
        assert retrieved.trace_id == trace.trace_id


class TestConfig:
    """Test configuration loading."""

    def test_default_config(self):
        """Test default configuration."""
        config = GlassBoxConfig()
        
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 8000
        assert config.trace.enabled is True
        assert config.writeback.enabled is True

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = GlassBoxConfig(
            server={"port": 9000},
            encoding={"default_encoder": "openai"},
        )
        assert config.server.port == 9000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
