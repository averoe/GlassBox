"""
Trace Tracker - core observability system.

Records detailed step-by-step execution traces for every operation
in the RAG pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from glassbox_rag.config import GlassBoxConfig


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
            return delta * 1000
        return 0.0

    def add_child(self, step: "TraceStep"):
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
            "input_data": self.input_data,
            "output_data": self.output_data,
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
            return delta * 1000
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
            "metadata": self.metadata,
        }

    def visualize(self) -> str:
        """Generate ASCII visualization of the trace."""
        if not self.root_step:
            return "No trace data"

        lines = []
        lines.append(f"Trace {self.trace_id}")
        lines.append(f"Duration: {self.get_duration_ms():.2f}ms")
        lines.append("")
        
        self._visualize_step(self.root_step, lines, 0)
        
        return "\n".join(lines)

    def _visualize_step(self, step: TraceStep, lines: List[str], depth: int):
        """Recursively visualize steps."""
        indent = "  " * depth
        prefix = "├─ " if depth > 0 else ""
        
        duration = f" ({step.get_duration_ms():.2f}ms)"
        error_mark = " ❌" if step.error else ""
        
        lines.append(f"{indent}{prefix}{step.name}{duration}{error_mark}")
        
        if step.error:
            lines.append(f"{indent}  └─ Error: {step.error}")
        
        for child in step.children:
            self._visualize_step(child, lines, depth + 1)


class TraceTracker:
    """
    Main trace tracking system.

    Records detailed step-by-step execution for the entire RAG pipeline.
    """

    def __init__(self, config: GlassBoxConfig):
        """Initialize trace tracker."""
        self.config = config
        self.enabled = config.trace.enabled
        self.backend = config.trace.backend
        self.retention_days = config.trace.retention_days
        self.sample_rate = config.trace.sample_rate
        
        # In-memory trace storage
        self.traces: Dict[str, Trace] = {}
        self.current_trace: Optional[Trace] = None
        self.current_step_stack: List[TraceStep] = []

    def start_trace(self, request_id: str) -> Trace:
        """Start a new trace."""
        if not self.enabled:
            return Trace(
                trace_id="",
                request_id=request_id,
                start_time=datetime.now(),
            )

        trace_id = str(uuid4())
        trace = Trace(
            trace_id=trace_id,
            request_id=request_id,
            start_time=datetime.now(),
        )
        
        self.current_trace = trace
        self.current_step_stack = []
        
        return trace

    def end_trace(self, trace_id: str) -> Trace:
        """End and store a trace."""
        if not self.enabled or self.current_trace is None:
            return Trace(
                trace_id=trace_id,
                request_id="",
                start_time=datetime.now(),
            )

        self.current_trace.end_time = datetime.now()
        self.traces[trace_id] = self.current_trace
        
        trace = self.current_trace
        self.current_trace = None
        self.current_step_stack = []
        
        return trace

    def start_step(
        self,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceStep:
        """Start a new trace step."""
        if not self.enabled:
            return TraceStep(
                step_id="",
                name=name,
                start_time=datetime.now(),
            )

        step_id = str(uuid4())
        step = TraceStep(
            step_id=step_id,
            name=name,
            start_time=datetime.now(),
            input_data=input_data or {},
            metadata=metadata or {},
        )
        
        # Add to current trace
        if self.current_trace:
            self.current_trace.steps[step_id] = step
            
            # Connect to parent or set as root
            if self.current_step_stack:
                parent = self.current_step_stack[-1]
                parent.add_child(step)
            else:
                self.current_trace.root_step = step
        
        self.current_step_stack.append(step)
        return step

    def end_step(
        self,
        step_id: str,
        output_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        """End a trace step."""
        if not self.enabled or not self.current_step_stack:
            return

        step = self.current_step_stack.pop()
        step.end_time = datetime.now()
        
        if output_data:
            step.output_data = output_data
        
        if error:
            step.error = error
            step.level = TraceLevel.ERROR

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Retrieve a stored trace."""
        return self.traces.get(trace_id)

    def list_traces(self, limit: int = 100) -> List[Trace]:
        """List all stored traces."""
        # Return most recent traces
        sorted_traces = sorted(
            self.traces.values(),
            key=lambda t: t.start_time,
            reverse=True,
        )
        return sorted_traces[:limit]

    def clear_traces(self):
        """Clear all stored traces."""
        self.traces.clear()
