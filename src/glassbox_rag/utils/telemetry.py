"""
OpenTelemetry + Prometheus metrics export.

Bridges GlassBox's internal tracing and metrics with standard
observability backends. Integrates with the VisualDebugger to
provide a unified observability dashboard.
"""

from typing import Any, Dict, List, Optional
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)

# ── Lazy imports for optional OTel deps ──────────────────────────
_otel_available = False
_prom_available = False

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import StatusCode

    _otel_available = True
except ImportError:
    pass

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    _prom_available = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════
#  OpenTelemetry Integration
# ═══════════════════════════════════════════════════════════════════


class OTelExporter:
    """
    Bridges GlassBox traces → OpenTelemetry spans.

    Converts the internal Trace/TraceStep hierarchy into OTel spans
    so they appear in Jaeger, Zipkin, Datadog, or any OTel-compatible
    observability backend.
    """

    def __init__(
        self,
        service_name: str = "glassbox-rag",
        endpoint: Optional[str] = None,
        exporter_type: str = "console",
    ):
        self.service_name = service_name
        self.endpoint = endpoint
        self.exporter_type = exporter_type
        self._tracer = None
        self._provider = None

    def initialize(self) -> bool:
        """Initialize OTel tracer provider and exporter."""
        if not _otel_available:
            logger.warning(
                "OpenTelemetry not installed. Run: pip install glassbox-rag[telemetry]"
            )
            return False

        resource = Resource.create({"service.name": self.service_name})
        self._provider = TracerProvider(resource=resource)

        if self.exporter_type == "otlp" and self.endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                exporter = OTLPSpanExporter(endpoint=self.endpoint)
            except ImportError:
                logger.warning("OTLP exporter not installed, falling back to console")
                exporter = ConsoleSpanExporter()
        elif self.exporter_type == "jaeger" and self.endpoint:
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter

                exporter = JaegerExporter(
                    agent_host_name=self.endpoint.split(":")[0],
                    agent_port=int(self.endpoint.split(":")[-1]),
                )
            except ImportError:
                logger.warning("Jaeger exporter not installed, falling back to console")
                exporter = ConsoleSpanExporter()
        else:
            exporter = ConsoleSpanExporter()

        self._provider.add_span_processor(BatchSpanProcessor(exporter))
        otel_trace.set_tracer_provider(self._provider)
        self._tracer = otel_trace.get_tracer("glassbox-rag")

        logger.info(
            "OpenTelemetry initialized: exporter=%s, endpoint=%s",
            self.exporter_type,
            self.endpoint or "console",
        )
        return True

    def export_trace(self, trace: Any) -> None:
        """
        Export a GlassBox Trace as OTel spans.

        Converts the step tree into a hierarchy of OTel spans
        with proper parent-child relationships.
        """
        if self._tracer is None or trace.root_step is None:
            return

        self._export_step(
            step=trace.root_step,
            trace_id=trace.trace_id,
            request_id=trace.request_id,
            parent_context=None,
        )

    def _export_step(
        self,
        step: Any,
        trace_id: str,
        request_id: str,
        parent_context: Any,
    ) -> None:
        """Recursively export steps as OTel spans."""
        if self._tracer is None:
            return

        context_mgr = self._tracer.start_as_current_span(
            name=step.name,
            context=parent_context,
        )

        with context_mgr as span:
            span.set_attribute("glassbox.trace_id", trace_id)
            span.set_attribute("glassbox.request_id", request_id)
            span.set_attribute("glassbox.step_id", step.step_id)
            span.set_attribute("glassbox.duration_ms", step.get_duration_ms())

            # Add input/output as events
            if step.input_data:
                span.add_event("input", attributes=_safe_attrs(step.input_data))
            if step.output_data:
                span.add_event("output", attributes=_safe_attrs(step.output_data))

            if step.error:
                span.set_status(StatusCode.ERROR, step.error)
                span.record_exception(Exception(step.error))
            else:
                span.set_status(StatusCode.OK)

            # Export children
            current_context = otel_trace.context.get_current()
            for child in step.children:
                self._export_step(child, trace_id, request_id, current_context)

    def shutdown(self) -> None:
        """Shutdown OTel provider."""
        if self._provider:
            self._provider.shutdown()


# ═══════════════════════════════════════════════════════════════════
#  Prometheus Metrics
# ═══════════════════════════════════════════════════════════════════


class PrometheusMetrics:
    """
    Exports GlassBox metrics as Prometheus-scrapable gauges/counters/histograms.

    Exposes:
    - glassbox_requests_total  (Counter)
    - glassbox_request_duration_seconds (Histogram)
    - glassbox_tokens_total (Counter)
    - glassbox_cost_usd_total (Counter)
    - glassbox_operation_duration_seconds (Histogram per operation type)
    - glassbox_vectors_total (Gauge)
    - glassbox_chunks_total (Gauge)
    """

    def __init__(self) -> None:
        self._initialized = False

        if not _prom_available:
            logger.warning(
                "prometheus_client not installed. Run: pip install glassbox-rag[telemetry]"
            )
            return

        self.info = Info("glassbox", "GlassBox RAG framework info")
        self.info.info({"version": "0.2.0"})

        self.requests_total = Counter(
            "glassbox_requests_total",
            "Total number of requests processed",
            ["operation"],
        )
        self.request_duration = Histogram(
            "glassbox_request_duration_seconds",
            "Request duration in seconds",
            ["operation"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
        self.tokens_total = Counter(
            "glassbox_tokens_total",
            "Total tokens processed",
            ["operation"],
        )
        self.cost_total = Counter(
            "glassbox_cost_usd_total",
            "Total cost in USD",
        )
        self.operation_duration = Histogram(
            "glassbox_operation_duration_seconds",
            "Operation duration in seconds",
            ["operation_type"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        )
        self.vectors_total = Gauge(
            "glassbox_vectors_total",
            "Total vectors in store",
        )
        self.chunks_total = Gauge(
            "glassbox_chunks_created_total",
            "Total chunks created",
        )
        self.active_requests = Gauge(
            "glassbox_active_requests",
            "Currently active requests",
        )
        self._initialized = True
        logger.info("Prometheus metrics initialized")

    def record_request(
        self,
        operation: str,
        duration_seconds: float,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record a completed request."""
        if not self._initialized:
            return

        self.requests_total.labels(operation=operation).inc()
        self.request_duration.labels(operation=operation).observe(duration_seconds)

        if tokens > 0:
            self.tokens_total.labels(operation=operation).inc(tokens)
        if cost_usd > 0:
            self.cost_total.inc(cost_usd)

    def record_operation(
        self,
        operation_type: str,
        duration_seconds: float,
    ) -> None:
        """Record a single operation's duration."""
        if not self._initialized:
            return
        self.operation_duration.labels(operation_type=operation_type).observe(
            duration_seconds
        )

    def set_vectors_count(self, count: int) -> None:
        if self._initialized:
            self.vectors_total.set(count)

    def set_chunks_count(self, count: int) -> None:
        if self._initialized:
            self.chunks_total.set(count)

    def inc_active(self) -> None:
        if self._initialized:
            self.active_requests.inc()

    def dec_active(self) -> None:
        if self._initialized:
            self.active_requests.dec()

    def generate_metrics(self) -> bytes:
        """Generate Prometheus-format metrics output."""
        if not _prom_available:
            return b"# prometheus_client not installed\n"
        return generate_latest()

    @property
    def content_type(self) -> str:
        if _prom_available:
            return CONTENT_TYPE_LATEST
        return "text/plain"


# ═══════════════════════════════════════════════════════════════════
#  Integrated Telemetry Hub
# ═══════════════════════════════════════════════════════════════════


class TelemetryHub:
    """
    Central telemetry hub that bridges internal observability
    with external systems.

    Integrates:
    - GlassBox TraceTracker → OTel spans
    - GlassBox MetricsTracker → Prometheus metrics
    - VisualDebugger reads from both internal + OTel data
    """

    def __init__(
        self,
        otel_enabled: bool = False,
        otel_endpoint: Optional[str] = None,
        otel_exporter: str = "console",
        prometheus_enabled: bool = False,
        service_name: str = "glassbox-rag",
    ):
        self.otel_exporter: Optional[OTelExporter] = None
        self.prometheus: Optional[PrometheusMetrics] = None

        if otel_enabled:
            self.otel_exporter = OTelExporter(
                service_name=service_name,
                endpoint=otel_endpoint,
                exporter_type=otel_exporter,
            )
            self.otel_exporter.initialize()

        if prometheus_enabled:
            self.prometheus = PrometheusMetrics()

    def on_trace_complete(self, trace: Any) -> None:
        """Called when a GlassBox trace completes. Exports to OTel."""
        if self.otel_exporter:
            try:
                self.otel_exporter.export_trace(trace)
            except Exception as e:
                logger.error("Failed to export trace to OTel: %s", e)

    def on_request_complete(
        self,
        operation: str,
        duration_seconds: float,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Called when a request completes. Updates Prometheus."""
        if self.prometheus:
            self.prometheus.record_request(
                operation=operation,
                duration_seconds=duration_seconds,
                tokens=tokens,
                cost_usd=cost_usd,
            )

    def on_operation_complete(
        self,
        operation_type: str,
        duration_seconds: float,
    ) -> None:
        """Called when an operation completes."""
        if self.prometheus:
            self.prometheus.record_operation(operation_type, duration_seconds)

    def get_prometheus_metrics(self) -> bytes:
        """Get Prometheus-format metrics."""
        if self.prometheus:
            return self.prometheus.generate_metrics()
        return b"# Prometheus not enabled\n"

    def get_prometheus_content_type(self) -> str:
        if self.prometheus:
            return self.prometheus.content_type
        return "text/plain"

    def get_telemetry_status(self) -> Dict:
        """Get telemetry system status for the visual debugger."""
        return {
            "otel_enabled": self.otel_exporter is not None,
            "otel_backend": (
                self.otel_exporter.exporter_type if self.otel_exporter else None
            ),
            "prometheus_enabled": self.prometheus is not None,
            "prometheus_initialized": (
                self.prometheus._initialized if self.prometheus else False
            ),
        }

    def shutdown(self) -> None:
        if self.otel_exporter:
            self.otel_exporter.shutdown()


def _safe_attrs(data: Dict[str, Any]) -> Dict[str, str]:
    """Convert dict values to OTel-safe attribute types (str)."""
    result = {}
    for k, v in data.items():
        if isinstance(v, (str, int, float, bool)):
            result[k] = str(v)
        else:
            result[k] = str(v)[:256]  # Truncate large values
    return result
