"""
Metrics and Cost Tracking - concurrency-safe.

Tracks token usage, latency, and costs per stage of the RAG pipeline.
Uses contextvars for per-request isolation and adds percentile tracking.
"""

from __future__ import annotations

import contextvars
import statistics
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Deque, Dict, List, Optional

from glassbox_rag.config import GlassBoxConfig
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)

# ── Per-request context ──────────────────────────────────────────
_current_request: contextvars.ContextVar[Optional["RequestMetrics"]] = contextvars.ContextVar(
    "current_request_metrics", default=None
)


class OperationType(Enum):
    """Types of operations to track."""

    ENCODE = "encode"
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    GENERATE = "generate"
    WRITEBACK = "writeback"
    CHUNK = "chunk"
    INGEST = "ingest"


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""

    operation_type: OperationType
    start_time: datetime
    end_time: datetime | None = None

    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Cost tracking
    cost_usd: float = 0.0

    # Additional metadata
    metadata: Dict = field(default_factory=dict)

    def get_latency_ms(self) -> float:
        """Get operation latency in milliseconds."""
        if self.end_time:
            delta = (self.end_time - self.start_time).total_seconds()
            return round(delta * 1000, 3)
        return 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "operation_type": self.operation_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "latency_ms": self.get_latency_ms(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata,
        }


@dataclass
class RequestMetrics:
    """Aggregated metrics for an entire request."""

    request_id: str
    start_time: datetime
    end_time: datetime | None = None

    operations: list[OperationMetrics] = field(default_factory=list)

    # Aggregated values
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def add_operation(self, operation: OperationMetrics) -> None:
        """Add operation metrics."""
        self.operations.append(operation)

    def finalize(self) -> None:
        """Finalize metrics and compute aggregates."""
        if not self.end_time:
            self.end_time = datetime.now(timezone.utc)

        self.total_latency_ms = round(
            (self.end_time - self.start_time).total_seconds() * 1000, 3
        )

        self.total_tokens = 0
        self.total_cost_usd = 0.0
        for operation in self.operations:
            self.total_tokens += operation.total_tokens
            self.total_cost_usd += operation.cost_usd

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "operations": [op.to_dict() for op in self.operations],
        }


class LatencyHistogram:
    """Tracks latency percentiles for an operation type using rolling window."""

    def __init__(self, max_samples: int = 1000):
        self._samples: Deque[float] = deque(maxlen=max_samples)

    def record(self, latency_ms: float) -> None:
        """Record a latency sample."""
        self._samples.append(latency_ms)

    def percentile(self, p: float) -> float:
        """Get percentile (0-100). Returns 0 if no samples."""
        if not self._samples:
            return 0.0
        sorted_samples = sorted(self._samples)
        k = (len(sorted_samples) - 1) * (p / 100)
        f = int(k)
        c = f + 1
        if c >= len(sorted_samples):
            return sorted_samples[f]
        d = k - f
        return sorted_samples[f] + d * (sorted_samples[c] - sorted_samples[f])

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def mean(self) -> float:
        return statistics.mean(self._samples) if self._samples else 0.0

    @property
    def count(self) -> int:
        return len(self._samples)

    def to_dict(self) -> Dict:
        return {
            "count": self.count,
            "mean_ms": round(self.mean, 3),
            "p50_ms": round(self.p50, 3),
            "p95_ms": round(self.p95, 3),
            "p99_ms": round(self.p99, 3),
        }


class MetricsTracker:
    """
    Concurrency-safe metrics tracker.

    Uses contextvars for per-request state and maintains global
    latency histograms per operation type.
    """

    def __init__(self, config: GlassBoxConfig):
        """Initialize metrics tracker."""
        self.config = config
        self.enabled = config.metrics.enabled
        self.track_tokens = config.metrics.track_tokens
        self.track_latency = config.metrics.track_latency
        self.track_cost = config.metrics.track_cost
        self.costs = config.metrics.costs or {}

        # Global request history (bounded, ordered by insertion for O(1) eviction)
        self._request_history: Ordereddict[str, RequestMetrics] = OrderedDict()
        self._max_history = 5000

        # Per-operation-type latency histograms
        self._histograms: dict[OperationType, LatencyHistogram] = {
            op: LatencyHistogram() for op in OperationType
        }

        # Global counters
        self.total_requests: int = 0
        self.total_tokens_used: int = 0
        self.total_cost_usd: float = 0.0

    def start_request(self, request_id: str) -> RequestMetrics:
        """Start tracking a new request (stored in context)."""
        metrics = RequestMetrics(
            request_id=request_id,
            start_time=datetime.now(timezone.utc),
        )

        if self.enabled:
            _current_request.set(metrics)
            self.total_requests += 1

        return metrics

    def end_request(self, request_id: str) -> RequestMetrics:
        """End tracking for a request."""
        metrics = _current_request.get()

        if not self.enabled or metrics is None:
            return RequestMetrics(
                request_id=request_id,
                start_time=datetime.now(timezone.utc),
            )

        metrics.finalize()

        # Update globals
        self.total_tokens_used += metrics.total_tokens
        self.total_cost_usd += metrics.total_cost_usd

        # Store in history (O(1) eviction via OrderedDict)
        self._request_history[request_id] = metrics
        if len(self._request_history) > self._max_history:
            self._request_history.popitem(last=False)  # Remove oldest entry

        # Clear context
        _current_request.set(None)

        logger.debug(
            "Request %s completed: %.2fms, %d tokens, $%.6f",
            request_id[:8],
            metrics.total_latency_ms,
            metrics.total_tokens,
            metrics.total_cost_usd,
        )
        return metrics

    def start_operation(
        self,
        operation_type: OperationType,
        metadata: Dict | None = None,
    ) -> OperationMetrics:
        """Start tracking an operation."""
        return OperationMetrics(
            operation_type=operation_type,
            start_time=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

    def end_operation(
        self,
        operation: OperationMetrics,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """End tracking for an operation and calculate costs."""
        if not self.enabled:
            return

        operation.end_time = datetime.now(timezone.utc)
        latency = operation.get_latency_ms()

        if self.track_tokens:
            operation.input_tokens = input_tokens
            operation.output_tokens = output_tokens
            operation.total_tokens = input_tokens + output_tokens

        if self.track_cost:
            operation.cost_usd = self._calculate_cost(operation)

        if self.track_latency:
            self._histograms[operation.operation_type].record(latency)

        # Attach to current request
        metrics = _current_request.get()
        if metrics is not None:
            metrics.add_operation(operation)

    def _calculate_cost(self, operation: OperationMetrics) -> float:
        """Calculate cost for an operation."""
        if not self.track_cost or not self.costs:
            return 0.0

        total_cost = 0.0

        if operation.total_tokens > 0:
            # Check for operation-specific token cost
            op_token_key = f"{operation.operation_type.value}_1k_tokens"
            if op_token_key in self.costs:
                total_cost += (operation.total_tokens / 1000) * self.costs[op_token_key]
            else:
                # Fallback to generic token costs
                for cost_key, cost_value in self.costs.items():
                    if "1k_tokens" in cost_key:
                        total_cost += (operation.total_tokens / 1000) * cost_value
                        break

        # Per-call cost
        op_call_key = f"{operation.operation_type.value}_cost"
        if op_call_key in self.costs:
            total_cost += self.costs[op_call_key]

        return total_cost

    def get_request_metrics(self, request_id: str) -> RequestMetrics | None:
        """Get metrics for a specific request."""
        return self._request_history.get(request_id)

    def get_histogram(self, operation_type: OperationType) -> LatencyHistogram:
        """Get the latency histogram for an operation type."""
        return self._histograms[operation_type]

    def get_summary(self) -> Dict:
        """Get a global metrics summary."""
        return {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "histograms": {
                op.value: hist.to_dict()
                for op, hist in self._histograms.items()
                if hist.count > 0
            },
        }
