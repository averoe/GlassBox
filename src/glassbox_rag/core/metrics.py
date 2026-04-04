"""
Metrics and Cost Tracking - detailed tracking of operations.

Tracks token usage, latency, and costs per stage of the RAG pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from glassbox_rag.config import GlassBoxConfig


class OperationType(Enum):
    """Types of operations to track."""

    ENCODE = "encode"
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    GENERATE = "generate"
    WRITEBACK = "writeback"


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""

    operation_type: OperationType
    start_time: datetime
    end_time: Optional[datetime] = None
    latency_ms: float = 0.0
    
    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Cost tracking
    cost_usd: float = 0.0
    
    # Additional metadata
    metadata: Dict = field(default_factory=dict)

    def get_latency(self) -> float:
        """Get operation latency in milliseconds."""
        if self.end_time:
            delta = (self.end_time - self.start_time).total_seconds()
            return delta * 1000
        return 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "operation_type": self.operation_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "latency_ms": self.get_latency(),
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
    end_time: Optional[datetime] = None
    
    operations: List[OperationMetrics] = field(default_factory=list)
    
    # Aggregated values
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def add_operation(self, operation: OperationMetrics):
        """Add operation metrics."""
        self.operations.append(operation)

    def finalize(self):
        """Finalize metrics and compute aggregates."""
        if not self.end_time:
            self.end_time = datetime.now()

        self.total_latency_ms = (
            (self.end_time - self.start_time).total_seconds() * 1000
        )

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
            "total_cost_usd": self.total_cost_usd,
            "operations": [op.to_dict() for op in self.operations],
        }


class MetricsTracker:
    """
    Tracks metrics and costs across the RAG pipeline.

    Records token usage, latency, and costs for each operation stage.
    """

    def __init__(self, config: GlassBoxConfig):
        """Initialize metrics tracker."""
        self.config = config
        self.enabled = config.metrics.enabled
        self.track_tokens = config.metrics.track_tokens
        self.track_latency = config.metrics.track_latency
        self.track_cost = config.metrics.track_cost
        self.costs = config.metrics.costs or {}
        
        # Storage for request metrics
        self.current_request_metrics: Optional[RequestMetrics] = None
        self.request_history: Dict[str, RequestMetrics] = {}

    def start_request(self, request_id: str) -> RequestMetrics:
        """Start tracking a new request."""
        if not self.enabled:
            return RequestMetrics(request_id=request_id, start_time=datetime.now())

        metrics = RequestMetrics(request_id=request_id, start_time=datetime.now())
        self.current_request_metrics = metrics
        return metrics

    def end_request(self, request_id: str) -> RequestMetrics:
        """End tracking for a request."""
        if not self.enabled or self.current_request_metrics is None:
            return RequestMetrics(request_id=request_id, start_time=datetime.now())

        self.current_request_metrics.finalize()
        self.request_history[request_id] = self.current_request_metrics
        metrics = self.current_request_metrics
        self.current_request_metrics = None
        return metrics

    def start_operation(
        self,
        operation_type: OperationType,
        metadata: Optional[Dict] = None,
    ) -> OperationMetrics:
        """Start tracking an operation."""
        if not self.enabled:
            return OperationMetrics(
                operation_type=operation_type,
                start_time=datetime.now(),
            )

        operation = OperationMetrics(
            operation_type=operation_type,
            start_time=datetime.now(),
            metadata=metadata or {},
        )
        return operation

    def end_operation(
        self,
        operation: OperationMetrics,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """End tracking for an operation and calculate costs."""
        if not self.enabled:
            return

        operation.end_time = datetime.now()

        if self.track_tokens:
            operation.input_tokens = input_tokens
            operation.output_tokens = output_tokens
            operation.total_tokens = input_tokens + output_tokens

        if self.track_cost:
            operation.cost_usd = self._calculate_cost(operation)

        if self.current_request_metrics:
            self.current_request_metrics.add_operation(operation)

    def _calculate_cost(self, operation: OperationMetrics) -> float:
        """Calculate cost for an operation."""
        if not self.track_cost or not self.costs:
            return 0.0

        total_cost = 0.0

        # Cost based on tokens
        if operation.total_tokens > 0:
            # Assume costs are per 1k tokens
            for cost_key, cost_value in self.costs.items():
                if "token" in cost_key.lower():
                    # Rough calculation: cost per 1k tokens
                    total_cost += (operation.total_tokens / 1000) * cost_value

        # Cost based on operation type
        op_type_key = f"{operation.operation_type.value}_cost"
        if op_type_key in self.costs:
            total_cost += self.costs[op_type_key]

        return total_cost

    def get_request_metrics(self, request_id: str) -> Optional[RequestMetrics]:
        """Get metrics for a specific request."""
        return self.request_history.get(request_id)
