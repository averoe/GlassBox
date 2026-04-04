"""
Visual Debugger Backend - provides visualization for traces.

Supports both ASCII and structured data formats for the web UI.
"""

from typing import Dict, List, Optional

from glassbox_rag.trace.tracker import Trace, TraceStep


class VisualDebugger:
    """
    Provides visualization and debugging utilities for traces.

    Generates ASCII art, structured data, and supports filtering/searching
    through execution traces.
    """

    @staticmethod
    def format_trace_ascii(trace: Trace) -> str:
        """Generate ASCII visualization of a trace."""
        return trace.visualize()

    @staticmethod
    def format_trace_json(trace: Trace) -> Dict:
        """Generate JSON representation of a trace."""
        return trace.to_dict()

    @staticmethod
    def filter_by_name(traces: List[Trace], name_pattern: str) -> List[Trace]:
        """Filter traces by step name pattern."""
        filtered = []
        for trace in traces:
            if trace.root_step and VisualDebugger._search_step(trace.root_step, name_pattern):
                filtered.append(trace)
        return filtered

    @staticmethod
    def _search_step(step: TraceStep, pattern: str) -> bool:
        """Recursively search for pattern in steps."""
        if pattern.lower() in step.name.lower():
            return True
        
        for child in step.children:
            if VisualDebugger._search_step(child, pattern):
                return True
        
        return False

    @staticmethod
    def get_trace_summary(trace: Trace) -> Dict:
        """Get a summary of the trace."""
        return {
            "trace_id": trace.trace_id,
            "request_id": trace.request_id,
            "duration_ms": trace.get_duration_ms(),
            "start_time": trace.start_time.isoformat(),
            "end_time": trace.end_time.isoformat() if trace.end_time else None,
            "total_steps": len(trace.steps),
            "root_step": trace.root_step.name if trace.root_step else None,
        }

    @staticmethod
    def get_step_tree(step: TraceStep, max_depth: Optional[int] = None, depth: int = 0) -> Dict:
        """Convert step to nested tree structure."""
        if max_depth and depth >= max_depth:
            return {}

        tree = {
            "id": step.step_id,
            "name": step.name,
            "duration_ms": step.get_duration_ms(),
            "level": step.level.value,
            "error": step.error,
            "children": [],
        }

        for child in step.children:
            tree["children"].append(
                VisualDebugger.get_step_tree(child, max_depth, depth + 1)
            )

        return tree
