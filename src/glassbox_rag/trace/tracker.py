"""
Trace Tracker - core observability system (concurrency-safe).

Records detailed step-by-step execution traces for every operation
in the RAG pipeline. Uses contextvars so each async request gets
its own trace context — no cross-request corruption.
"""

import contextvars
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from glassbox_rag.config import GlassBoxConfig
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)

# ── Context Variables (one per async task / request) ──────────────
_current_trace: contextvars.ContextVar[Optional["Trace"]] = contextvars.ContextVar(
    "current_trace", default=None
)
_current_step_stack: contextvars.ContextVar[List["TraceStep"]] = contextvars.ContextVar(
    "current_step_stack", default=[]
)


class TraceLevel(Enum):
    """Trace verbosity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class TraceStep:
    """A single step in the execution trace."""

    step_id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    level: TraceLevel = TraceLevel.INFO

    # Input/Output
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)

    # Nested steps
    children: List["TraceStep"] = field(default_factory=list)

    # Error information
    error: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_duration_ms(self) -> float:
        """Get step duration in milliseconds."""
        if self.end_time:
            delta = (self.end_time - self.start_time).total_seconds()
            return round(delta * 1000, 3)
        return 0.0

    def add_child(self, step: "TraceStep") -> None:
        """Add a child step."""
        self.children.append(step)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.get_duration_ms(),
            "level": self.level.value,
            "input_data": _safe_serialize(self.input_data),
            "output_data": _safe_serialize(self.output_data),
            "children": [child.to_dict() for child in self.children],
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class Trace:
    """Complete execution trace for a request."""

    trace_id: str
    request_id: str
    start_time: datetime
    end_time: Optional[datetime] = None

    # Root step
    root_step: Optional[TraceStep] = None

    # Flat list for quick lookup
    steps: Dict[str, TraceStep] = field(default_factory=dict)

    # Request metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_duration_ms(self) -> float:
        """Get total trace duration."""
        if self.end_time:
            delta = (self.end_time - self.start_time).total_seconds()
            return round(delta * 1000, 3)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.get_duration_ms(),
            "root_step": self.root_step.to_dict() if self.root_step else None,
            "num_steps": len(self.steps),
            "metadata": self.metadata,
        }

    def visualize(self) -> str:
        """Generate ASCII visualization of the trace."""
        if not self.root_step:
            return "No trace data"

        lines: List[str] = []
        lines.append(f"╔══ Trace {self.trace_id[:12]}…")
        lines.append(f"║  Request: {self.request_id[:12]}…")
        lines.append(f"║  Duration: {self.get_duration_ms():.2f}ms")
        lines.append(f"║  Steps: {len(self.steps)}")
        lines.append("╠══════════════════════════════════════")

        self._visualize_step(self.root_step, lines, 0)

        lines.append("╚══════════════════════════════════════")
        return "\n".join(lines)

    def _visualize_step(self, step: TraceStep, lines: List[str], depth: int) -> None:
        """Recursively visualize steps."""
        indent = "║  " + "  " * depth
        is_last = depth == 0

        duration = f" ({step.get_duration_ms():.2f}ms)"
        status = " ❌" if step.error else " ✓"

        connector = "├─ " if not is_last else "┌─ "
        lines.append(f"{indent}{connector}{step.name}{duration}{status}")

        if step.error:
            lines.append(f"{indent}│  └─ Error: {step.error}")

        for child in step.children:
            self._visualize_step(child, lines, depth + 1)


class TraceTracker:
    """
    Concurrency-safe trace tracking system with pluggable backends.

    Uses contextvars so each async task (request) gets isolated
    trace state. Supports memory, Redis, and PostgreSQL backends
    for persistent trace storage.
    """

    def __init__(self, config: GlassBoxConfig):
        """Initialize trace tracker."""
        self.config = config
        self.enabled = config.trace.enabled
        self.backend_type = config.trace.backend
        self.retention_days = config.trace.retention_days
        self.sample_rate = config.trace.sample_rate

        # In-memory cache (always present for fast access)
        self._traces: Dict[str, Trace] = {}
        self._max_stored = 10000

        # Persistent backend (initialized async via init_backend)
        self._backend = None

    async def init_backend(self) -> None:
        """Initialize the persistent trace backend (call during engine startup)."""
        if self.backend_type == "memory":
            return  # In-memory only — nothing to initialize

        try:
            from glassbox_rag.trace.backends import create_trace_backend
            self._backend = await create_trace_backend(
                self.backend_type,
                self.config.model_extra or {},
            )
            logger.info("Trace backend initialized: %s", self.backend_type)
        except Exception as e:
            logger.warning("Trace backend '%s' failed, using memory: %s", self.backend_type, e)
            self._backend = None

    def start_trace(self, request_id: str) -> Trace:
        """Start a new trace for the current async context."""
        if not self.enabled:
            return Trace(
                trace_id="",
                request_id=request_id,
                start_time=datetime.now(timezone.utc),
            )

        trace_id = str(uuid4())
        trace = Trace(
            trace_id=trace_id,
            request_id=request_id,
            start_time=datetime.now(timezone.utc),
        )

        _current_trace.set(trace)
        _current_step_stack.set([])

        logger.debug("Started trace %s for request %s", trace_id[:8], request_id[:8])
        return trace

    def end_trace(self, trace_id: str) -> Trace:
        """End and store a trace (in-memory + persistent backend)."""
        trace = _current_trace.get()

        if not self.enabled or trace is None:
            return Trace(
                trace_id=trace_id,
                request_id="",
                start_time=datetime.now(timezone.utc),
            )

        trace.end_time = datetime.now(timezone.utc)

        # Store in memory cache
        self._traces[trace_id] = trace
        if len(self._traces) > self._max_stored:
            oldest_key = min(self._traces, key=lambda k: self._traces[k].start_time)
            del self._traces[oldest_key]

        # Persist to backend (fire-and-forget)
        if self._backend is not None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._persist_trace(trace))
                else:
                    asyncio.run(self._persist_trace(trace))
            except RuntimeError:
                pass  # No event loop — skip persistence

        _current_trace.set(None)
        _current_step_stack.set([])

        logger.debug(
            "Ended trace %s (%.2fms, %d steps)",
            trace_id[:8], trace.get_duration_ms(), len(trace.steps),
        )
        return trace

    async def _persist_trace(self, trace: Trace) -> None:
        """Persist trace to backend storage."""
        if self._backend:
            try:
                await self._backend.store_trace(trace.trace_id, trace.to_dict())
            except Exception as e:
                logger.debug("Failed to persist trace %s: %s", trace.trace_id[:8], e)

    def start_step(
        self,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceStep:
        """Start a new trace step in the current context."""
        if not self.enabled:
            return TraceStep(
                step_id="",
                name=name,
                start_time=datetime.now(timezone.utc),
            )

        step_id = str(uuid4())
        step = TraceStep(
            step_id=step_id,
            name=name,
            start_time=datetime.now(timezone.utc),
            input_data=input_data or {},
            metadata=metadata or {},
        )

        trace = _current_trace.get()
        stack = _current_step_stack.get()

        if trace is not None:
            trace.steps[step_id] = step
            if stack:
                parent = stack[-1]
                parent.add_child(step)
            else:
                trace.root_step = step

        new_stack = list(stack)
        new_stack.append(step)
        _current_step_stack.set(new_stack)

        return step

    def end_step(
        self,
        step_id: str,
        output_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """End a trace step in the current context."""
        if not self.enabled:
            return

        stack = _current_step_stack.get()
        if not stack:
            return

        step = stack[-1]
        step.end_time = datetime.now(timezone.utc)

        if output_data:
            step.output_data = output_data

        if error:
            step.error = error
            step.level = TraceLevel.ERROR
            logger.warning("Step '%s' failed: %s", step.name, error)

        new_stack = list(stack[:-1])
        _current_step_stack.set(new_stack)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Retrieve a stored trace (from memory cache)."""
        return self._traces.get(trace_id)

    async def get_trace_persistent(self, trace_id: str) -> Optional[Dict]:
        """Retrieve from persistent backend (if not in memory)."""
        trace = self._traces.get(trace_id)
        if trace:
            return trace.to_dict()
        if self._backend:
            return await self._backend.get_trace(trace_id)
        return None

    def list_traces(self, limit: int = 100) -> List[Trace]:
        """List stored traces from memory, most recent first."""
        sorted_traces = sorted(
            self._traces.values(),
            key=lambda t: t.start_time,
            reverse=True,
        )
        return sorted_traces[:limit]

    def clear_traces(self) -> None:
        """Clear all stored traces."""
        self._traces.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        traces = list(self._traces.values())
        if not traces:
            return {"total_traces": 0, "backend": self.backend_type}

        durations = [t.get_duration_ms() for t in traces if t.end_time]
        return {
            "total_traces": len(traces),
            "backend": self.backend_type,
            "persistent_backend": self._backend is not None,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
        }

    async def shutdown(self) -> None:
        """Shutdown the persistent backend."""
        if self._backend:
            await self._backend.shutdown()
            self._backend = None


def _safe_serialize(data: Dict[str, Any]) -> Dict[str, Any]:
    """Safely serialize data dict, converting non-serializable values."""
    result = {}
    for key, value in data.items():
        try:
            # Test if value is JSON-serializable
            import json
            json.dumps(value)
            result[key] = value
        except (TypeError, ValueError):
            result[key] = str(value)
    return result
