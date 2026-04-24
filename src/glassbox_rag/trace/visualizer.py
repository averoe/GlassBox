"""
Visual Debugger — enhanced trace visualization and analysis.

Provides ASCII waterfalls, cost breakdowns, per-step inspection,
trace comparison, and simple anomaly detection.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from glassbox_rag.trace.tracker import Trace, TraceStep
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class VisualDebugger:
    """
    Provides visualization and debugging utilities for traces.

    Features:
    - ASCII trace tree
    - Latency waterfall visualization
    - Cost / token breakdown per step
    - Per-step input/output inspection
    - Trace comparison (diff two traces)
    - Anomaly detection (flag slow steps)
    """

    # ── ASCII Trace Tree ──────────────────────────────────────────

    @staticmethod
    def format_trace_ascii(trace: Trace) -> str:
        """Generate ASCII visualization of a trace."""
        return trace.visualize()

    @staticmethod
    def format_trace_json(trace: Trace) -> Dict:
        """Generate JSON representation of a trace."""
        return trace.to_dict()

    # ── Latency Waterfall ─────────────────────────────────────────

    @staticmethod
    def format_waterfall(trace: Trace, width: int = 60) -> str:
        """
        Generate a latency waterfall showing relative timing of each step.

        Example output:
        ╔══ Latency Waterfall ══════════════════════════════════════
        ║ retrieve        |████████████████████████████████| 45.2ms
        ║  encode_query   |██████████░░░░░░░░░░░░░░░░░░░░░| 12.3ms
        ║  adaptive_retr  |░░░░░░░░░░████████████████████░░| 28.1ms
        ╚══════════════════════════════════════════════════════════
        """
        if not trace.root_step or not trace.end_time:
            return "No timing data available"

        total_ms = trace.get_duration_ms()
        if total_ms == 0:
            return "Trace duration is zero"

        lines: list[str] = []
        lines.append(f"╔══ Latency Waterfall (total: {total_ms:.1f}ms) " + "═" * max(0, width - 30))

        steps = _flatten_steps(trace.root_step)

        for step, depth in steps:
            step_ms = step.get_duration_ms()
            if total_ms > 0:
                start_offset = (step.start_time - trace.start_time).total_seconds() * 1000
                start_pct = start_offset / total_ms
                width_pct = step_ms / total_ms
            else:
                start_pct = 0
                width_pct = 1

            bar = _render_bar(start_pct, width_pct, bar_width=width - 30)
            indent = "  " * depth
            # Cap indent + name width to keep columns aligned regardless of depth
            available = max(4, 18 - len(indent))
            truncated_name = (step.name[:available - 2] + "..") if len(step.name) > available else step.name
            name = truncated_name.ljust(available)
            status = "❌" if step.error else ""

            lines.append(
                f"║ {indent}{name} |{bar}| {step_ms:>7.1f}ms {status}"
            )

        lines.append("╚" + "═" * (width + 5))
        return "\n".join(lines)

    # ── Cost & Token Breakdown ────────────────────────────────────

    @staticmethod
    def format_cost_breakdown(trace: Trace) -> str:
        """
        Generate a cost / token breakdown per step.

        Reads operation metadata from trace steps.
        """
        if not trace.root_step:
            return "No trace data"

        lines: list[str] = []
        lines.append("╔══ Cost & Token Breakdown ═══════════════════════")

        steps = _flatten_steps(trace.root_step)
        total_tokens = 0
        total_cost = 0.0

        for step, depth in steps:
            tokens = step.output_data.get("tokens", 0)
            cost = step.output_data.get("cost_usd", 0.0)
            total_tokens += tokens
            total_cost += cost

            indent = "  " * depth
            name = step.name[:20].ljust(20)

            token_str = f"{tokens:>6} tok" if tokens else "       -"
            cost_str = f"${cost:.6f}" if cost else "       -"

            lines.append(f"║ {indent}{name} {token_str}  {cost_str}")

        lines.append("╠════════════════════════════════════════════════")
        lines.append(f"║ {'TOTAL':22} {total_tokens:>6} tok  ${total_cost:.6f}")
        lines.append("╚════════════════════════════════════════════════")

        return "\n".join(lines)

    # ── Per-Step Inspection ───────────────────────────────────────

    @staticmethod
    def inspect_step(trace: Trace, step_name: str) -> Dict | None:
        """
        Get detailed info for a specific step by name.

        Returns the first step matching the name, with full
        input/output data and timing.
        """
        if not trace.root_step:
            return None

        step = _find_step_by_name(trace.root_step, step_name)
        if step is None:
            return None

        return {
            "step_id": step.step_id,
            "name": step.name,
            "duration_ms": step.get_duration_ms(),
            "level": step.level.value,
            "start_time": step.start_time.isoformat() if step.start_time else None,
            "end_time": step.end_time.isoformat() if step.end_time else None,
            "input_data": step.input_data,
            "output_data": step.output_data,
            "error": step.error,
            "num_children": len(step.children),
            "children_names": [c.name for c in step.children],
            "metadata": step.metadata,
        }

    # ── Trace Comparison ──────────────────────────────────────────

    @staticmethod
    def compare_traces(trace_a: Trace, trace_b: Trace) -> Dict:
        """
        Compare two traces side-by-side.

        Returns per-step latency differences and an overall summary.
        """
        steps_a = {s.name: s for s, _ in _flatten_steps(trace_a.root_step)} if trace_a.root_step else {}
        steps_b = {s.name: s for s, _ in _flatten_steps(trace_b.root_step)} if trace_b.root_step else {}

        all_names = sorted(set(list(steps_a.keys()) + list(steps_b.keys())))

        comparisons: list[Dict] = []
        for name in all_names:
            a_ms = steps_a[name].get_duration_ms() if name in steps_a else None
            b_ms = steps_b[name].get_duration_ms() if name in steps_b else None

            diff = None
            if a_ms is not None and b_ms is not None:
                diff = round(b_ms - a_ms, 3)

            comparisons.append({
                "step": name,
                "trace_a_ms": a_ms,
                "trace_b_ms": b_ms,
                "diff_ms": diff,
                "faster": "A" if diff and diff > 0 else ("B" if diff and diff < 0 else "equal"),
                "slower": "B" if diff and diff > 0 else ("A" if diff and diff < 0 else "equal"),
            })

        return {
            "trace_a_id": trace_a.trace_id,
            "trace_b_id": trace_b.trace_id,
            "trace_a_total_ms": trace_a.get_duration_ms(),
            "trace_b_total_ms": trace_b.get_duration_ms(),
            "total_diff_ms": round(trace_b.get_duration_ms() - trace_a.get_duration_ms(), 3),
            "steps": comparisons,
        }

    # ── Anomaly Detection ─────────────────────────────────────────

    @staticmethod
    def detect_anomalies(
        traces: list[Trace],
        threshold_multiplier: float = 2.0,
    ) -> list[Dict]:
        """
        Detect anomalous steps across a list of traces.

        A step is anomalous if its duration exceeds the mean + threshold * stddev
        across all traces containing that step.

        Args:
            traces: List of traces to analyze.
            threshold_multiplier: How many standard deviations above mean = anomaly.

        Returns:
            List of anomaly dicts with trace_id, step, duration, expected, etc.
        """
        from collections import defaultdict
        import math

        # Collect per-step durations
        step_durations: dict[str, list[tuple]] = defaultdict(list)
        for trace in traces:
            if not trace.root_step:
                continue
            for step, _ in _flatten_steps(trace.root_step):
                ms = step.get_duration_ms()
                if ms > 0:
                    step_durations[step.name].append((trace.trace_id, ms))

        anomalies: list[Dict] = []
        for step_name, entries in step_durations.items():
            if len(entries) < 2:
                continue  # Need at least a baseline to compare against

            if len(entries) < 5:
                logger.debug(
                    "Step '%s': only %d samples, anomaly detection less reliable",
                    step_name, len(entries),
                )

            durations = [d for _, d in entries]
            mean = sum(durations) / len(durations)

            # For 2-sample case, use ratio instead of stddev (stddev would be trivial)
            if len(entries) == 2:
                for trace_id, ms in entries:
                    if mean > 0 and ms / mean > threshold_multiplier:
                        anomalies.append({
                            "trace_id": trace_id,
                            "step": step_name,
                            "duration_ms": round(ms, 3),
                            "mean_ms": round(mean, 3),
                            "stddev_ms": 0,
                            "threshold_ms": round(mean * threshold_multiplier, 3),
                            "times_slower": round(ms / mean, 2),
                        })
                continue

            variance = sum((d - mean) ** 2 for d in durations) / len(durations)
            stddev = math.sqrt(variance) if variance > 0 else 0

            threshold = mean + threshold_multiplier * stddev

            for trace_id, ms in entries:
                if ms > threshold:
                    anomalies.append({
                        "trace_id": trace_id,
                        "step": step_name,
                        "duration_ms": round(ms, 3),
                        "mean_ms": round(mean, 3),
                        "stddev_ms": round(stddev, 3),
                        "threshold_ms": round(threshold, 3),
                        "times_slower": round(ms / mean, 2) if mean > 0 else 0,
                    })

        return sorted(anomalies, key=lambda a: a["times_slower"], reverse=True)

    # ── Filtering / Search ────────────────────────────────────────

    @staticmethod
    def filter_by_name(traces: list[Trace], name_pattern: str) -> list[Trace]:
        """Filter traces containing a step matching the pattern."""
        filtered = []
        for trace in traces:
            if trace.root_step and _search_step(trace.root_step, name_pattern):
                filtered.append(trace)
        return filtered

    @staticmethod
    def filter_slow_traces(traces: list[Trace], min_duration_ms: float) -> list[Trace]:
        """Filter traces slower than the given threshold."""
        return [t for t in traces if t.get_duration_ms() >= min_duration_ms]

    @staticmethod
    def filter_error_traces(traces: list[Trace]) -> list[Trace]:
        """Filter traces that contain errors."""
        result = []
        for trace in traces:
            if trace.root_step and _has_error(trace.root_step):
                result.append(trace)
        return result

    # ── Summary / Tree ────────────────────────────────────────────

    @staticmethod
    def get_trace_summary(trace: Trace) -> Dict:
        """Get a summary of the trace."""
        has_errors = False
        if trace.root_step:
            has_errors = _has_error(trace.root_step)

        return {
            "trace_id": trace.trace_id,
            "request_id": trace.request_id,
            "duration_ms": trace.get_duration_ms(),
            "start_time": trace.start_time.isoformat(),
            "end_time": trace.end_time.isoformat() if trace.end_time else None,
            "total_steps": len(trace.steps),
            "root_step": trace.root_step.name if trace.root_step else None,
            "has_errors": has_errors,
        }

    @staticmethod
    def get_step_tree(
        step: TraceStep,
        max_depth: int | None = None,
        depth: int = 0,
    ) -> Dict:
        """Convert step to nested tree structure."""
        if max_depth is not None and depth >= max_depth:
            return {}

        tree = {
            "id": step.step_id,
            "name": step.name,
            "duration_ms": step.get_duration_ms(),
            "level": step.level.value,
            "error": step.error,
            "input_keys": list(step.input_data.keys()),
            "output_keys": list(step.output_data.keys()),
            "children": [],
        }

        for child in step.children:
            child_tree = VisualDebugger.get_step_tree(child, max_depth, depth + 1)
            if child_tree:
                tree["children"].append(child_tree)

        return tree

    # ── Pipeline Visualization ────────────────────────────────────

    @staticmethod
    def format_pipeline_diagram(trace: "Trace") -> str:
        """
        Generate an ASCII pipeline diagram of the RAG flow.

        Shows the data flowing through each stage with timing.
        Example:
        ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
        │  Ingest  │───▶│  Chunk   │───▶│  Encode  │───▶│  Store   │
        │  45.2ms  │    │  12.3ms  │    │  28.1ms  │    │   4.8ms  │
        └──────────┘    └──────────┘    └──────────┘    └──────────┘
        """
        if not trace.root_step:
            return "No pipeline data available"

        steps = _flatten_steps(trace.root_step)
        # Only show top-level steps (depth == 1) for the pipeline view
        pipeline_steps = [
            (s, d) for s, d in steps if d <= 1
        ]
        if not pipeline_steps:
            return "No pipeline steps"

        lines_top = []
        lines_mid = []
        lines_bot = []
        lines_dur = []

        for i, (step, _) in enumerate(pipeline_steps):
            raw = step.name
            name = (raw[:8] + "..") if len(raw) > 10 else raw
            name = name.center(10)
            ms = step.get_duration_ms()
            dur = f"{ms:.1f}ms".center(10)
            status = "❌" if step.error else ""

            lines_top.append("┌──────────┐")
            lines_mid.append(f"│{name}│")
            lines_dur.append(f"│{dur}│")
            lines_bot.append(f"└──────────┘")

            if i < len(pipeline_steps) - 1:
                lines_top.append("    ")
                lines_mid.append("───▶")
                lines_dur.append("    ")
                lines_bot.append("    ")

        return "\n".join([
            "".join(lines_top),
            "".join(lines_mid),
            "".join(lines_dur),
            "".join(lines_bot),
        ])

    # ── Telemetry Dashboard ───────────────────────────────────────

    @staticmethod
    def format_telemetry_dashboard(
        trace: "Trace",
        telemetry_status: Dict | None = None,
        metrics_summary: Dict | None = None,
        as_html: bool = False,
    ) -> str:
        """
        Generate a combined telemetry dashboard.

        Integrates trace waterfall + Prometheus/OTel status
        into a single view for debugging and monitoring.

        Args:
            as_html: If True, wrap the ASCII output in a minimal HTML template
                     suitable for HTMLResponse from a web server.
        """
        lines: list = []
        lines.append("╔══════════════════════════════════════════════════════════════")
        lines.append("║  GlassBox RAG — Telemetry Dashboard")
        lines.append("╠══════════════════════════════════════════════════════════════")

        # Trace info
        lines.append(f"║  Trace ID:     {trace.trace_id}")
        lines.append(f"║  Request:      {trace.request_id}")
        lines.append(f"║  Duration:     {trace.get_duration_ms():.1f}ms")
        lines.append(f"║  Steps:        {len(trace.steps)}")
        lines.append("║")

        # Telemetry status
        if telemetry_status:
            otel = "✓ active" if telemetry_status.get("otel_enabled") else "✗ disabled"
            prom = "✓ active" if telemetry_status.get("prometheus_enabled") else "✗ disabled"
            backend = telemetry_status.get("otel_backend", "none")
            lines.append(f"║  OpenTelemetry: {otel} ({backend})")
            lines.append(f"║  Prometheus:    {prom}")
            lines.append("║")

        # Metrics summary
        if metrics_summary:
            lines.append(f"║  Total Requests:    {metrics_summary.get('total_requests', 0)}")
            lines.append(f"║  Total Tokens:      {metrics_summary.get('total_tokens', 0)}")
            lines.append(f"║  Total Cost:        ${metrics_summary.get('total_cost_usd', 0):.6f}")
            latency = metrics_summary.get("latency", {})
            if latency:
                lines.append(f"║  p50 Latency:       {latency.get('p50', 0):.1f}ms")
                lines.append(f"║  p95 Latency:       {latency.get('p95', 0):.1f}ms")
                lines.append(f"║  p99 Latency:       {latency.get('p99', 0):.1f}ms")
            lines.append("║")

        # Mini waterfall
        if trace.root_step:
            lines.append("║  ── Step Breakdown ──")
            flat = _flatten_steps(trace.root_step)
            total = trace.get_duration_ms()
            for step, depth in flat:
                indent = "  " * depth
                ms = step.get_duration_ms()
                pct = (ms / total * 100) if total > 0 else 0
                status = "❌" if step.error else "✓"
                lines.append(
                    f"║    {indent}{status} {step.name:<18} {ms:>7.1f}ms ({pct:.0f}%)"
                )
            lines.append("║")

        lines.append("╚══════════════════════════════════════════════════════════════")
        ascii_out = "\n".join(lines)

        if not as_html:
            return ascii_out

        # Wrap in a minimal HTML template for browser rendering
        import html as html_mod
        escaped = html_mod.escape(ascii_out)
        return (
            f'<!DOCTYPE html>\n'
            f'<html><head><meta charset="utf-8">\n'
            f'<title>GlassBox Trace {trace.trace_id[:8]}</title>\n'
            f'<style>body{{background:#0f1117;color:#cdd6f4;font-family:monospace;padding:2rem}}\n'
            f'pre{{white-space:pre-wrap;line-height:1.6}}</style></head>\n'
            f'<body><pre>{escaped}</pre></body></html>'
        )

    @staticmethod
    def get_enriched_summary(
        trace: "Trace",
        telemetry_status: Dict | None = None,
        metrics_summary: Dict | None = None,
    ) -> Dict:
        """
        Get trace summary enriched with telemetry context.

        Combines internal trace data with OTel/Prometheus status
        for a complete observability picture.
        """
        base = VisualDebugger.get_trace_summary(trace)
        base["telemetry"] = telemetry_status or {}
        base["metrics"] = metrics_summary or {}

        # Add step-level breakdown
        if trace.root_step:
            steps_detail = []
            for step, depth in _flatten_steps(trace.root_step):
                total = trace.get_duration_ms()
                ms = step.get_duration_ms()
                steps_detail.append({
                    "name": step.name,
                    "depth": depth,
                    "duration_ms": ms,
                    "pct_of_total": round((ms / total * 100) if total > 0 else 0, 1),
                    "error": step.error,
                    "input_keys": list(step.input_data.keys()),
                    "output_keys": list(step.output_data.keys()),
                })
            base["steps_detail"] = steps_detail

        return base


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _flatten_steps(
    step: TraceStep,
    depth: int = 0,
) -> list[tuple]:
    """Flatten a step tree to list of (step, depth) tuples."""
    result = [(step, depth)]
    for child in step.children:
        result.extend(_flatten_steps(child, depth + 1))
    return result


def _find_step_by_name(step: TraceStep, name: str) -> TraceStep | None:
    """Find first step matching name."""
    if step.name == name:
        return step
    for child in step.children:
        found = _find_step_by_name(child, name)
        if found:
            return found
    return None


def _search_step(step: TraceStep, pattern: str) -> bool:
    """Recursively search for pattern in step names."""
    if pattern.lower() in step.name.lower():
        return True
    for child in step.children:
        if _search_step(child, pattern):
            return True
    return False


def _has_error(step: TraceStep) -> bool:
    """Check if step or any descendant has an error."""
    if step.error:
        return True
    for child in step.children:
        if _has_error(child):
            return True
    return False


def _render_bar(start_pct: float, width_pct: float, bar_width: int = 30) -> str:
    """Render a single bar for the waterfall chart."""
    start_pct = max(0.0, min(1.0, start_pct))
    width_pct = max(0.0, min(1.0 - start_pct, width_pct))

    start_pos = int(start_pct * bar_width)
    end_pos = int((start_pct + width_pct) * bar_width)
    end_pos = max(end_pos, start_pos + 1)  # At least 1 char wide

    bar = "░" * start_pos + "█" * (end_pos - start_pos) + "░" * (bar_width - end_pos)
    return bar[:bar_width]
