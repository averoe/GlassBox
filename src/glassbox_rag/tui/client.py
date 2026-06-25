"""
GlassBox TUI — Direct Engine Client.

Wraps the GlassBoxEngine for TUI consumption, handling async
initialisation and providing a clean interface for all screens.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from glassbox_rag.config import GlassBoxConfig, load_config
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HealthSnapshot:
    """Point-in-time health reading."""
    status: str = "unknown"
    version: str = ""
    components: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class MetricsSnapshot:
    """Point-in-time metrics reading."""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    histograms: Dict[str, Dict] = field(default_factory=dict)


class EngineClient:
    """
    Direct wrapper around GlassBoxEngine for TUI usage.

    Instead of HTTP, the TUI instantiates the engine in-process
    and calls Python methods directly.  This class adds a
    convenience layer so screens don't need to know engine internals.
    """

    def __init__(self, config: GlassBoxConfig | None = None, config_path: str | None = None):
        self._config_path = config_path
        self._config = config
        self._engine = None  # type: ignore[assignment]
        self._initialized = False
        self._init_error: str | None = None

        # Rolling history for sparklines (up to 60 samples)
        self._request_history: list[int] = []
        self._latency_history: list[float] = []
        self._token_history: list[int] = []
        self._cost_history: list[float] = []

    # ── Lifecycle ─────────────────────────────────────────────────

    async def initialize(self) -> bool:
        """Start the engine.  Returns True on success."""
        try:
            if self._config is None:
                if self._config_path:
                    from pathlib import Path
                    self._config = load_config(self._config_path)
                else:
                    self._config = GlassBoxConfig()

            from glassbox_rag.core.engine import GlassBoxEngine
            self._engine = GlassBoxEngine(self._config)
            await self._engine.initialize()
            self._initialized = True
            self._init_error = None
            logger.info("EngineClient: engine initialised successfully")
            return True
        except Exception as exc:
            self._init_error = str(exc)
            logger.error("EngineClient: init failed — %s", exc)
            return False

    async def shutdown(self) -> None:
        if self._engine and self._initialized:
            await self._engine.shutdown()
            self._initialized = False

    @property
    def ready(self) -> bool:
        return self._initialized

    @property
    def error(self) -> str | None:
        return self._init_error

    @property
    def config(self) -> GlassBoxConfig:
        if self._config is None:
            self._config = GlassBoxConfig()
        return self._config

    # ── Health ────────────────────────────────────────────────────

    async def get_health(self) -> HealthSnapshot:
        if not self._initialized or self._engine is None:
            return HealthSnapshot(status="unavailable", warnings=[self._init_error or "Engine not initialised"])

        from glassbox_rag import __version__
        components: Dict[str, str] = {
            "encoder": "operational" if self._engine.encoding_layer.encoders else "no_encoders",
            "retriever": "operational",
            "writeback": "operational",
            "trace": "operational",
            "reranker": "operational" if self._engine.reranker else "disabled",
        }
        if self._engine.vector_store:
            try:
                ok = await self._engine.vector_store.health_check()
                components["vector_store"] = "operational" if ok else "unhealthy"
            except Exception:
                components["vector_store"] = "unhealthy"
        else:
            components["vector_store"] = "not_configured"

        if self._engine.database:
            try:
                ok = await self._engine.database.health_check()
                components["database"] = "operational" if ok else "unhealthy"
            except Exception:
                components["database"] = "unhealthy"
        else:
            components["database"] = "not_configured"

        all_ok = all(
            v in ("operational", "not_configured", "enabled", "disabled")
            for v in components.values()
        )
        warnings: list[str] = []
        cfg = self.config
        if cfg.server.workers > 1 and cfg.security.rate_limit_backend == "memory":
            warnings.append(
                f"Memory-based rate limiter is per-process — with {cfg.server.workers} workers, "
                f"effective rate limit is {cfg.server.workers}x the configured "
                f"{cfg.security.rate_limit_rpm} RPM."
            )
        otel = "enabled" if cfg.telemetry.otel_enabled else "disabled"
        components["telemetry_otel"] = otel
        prom = "enabled" if cfg.telemetry.prometheus_enabled else "disabled"
        components["telemetry_prometheus"] = prom

        return HealthSnapshot(
            status="healthy" if all_ok else "degraded",
            version=__version__,
            components=components,
            warnings=warnings,
        )

    # ── Metrics ───────────────────────────────────────────────────

    async def get_metrics(self) -> MetricsSnapshot:
        if not self._initialized or self._engine is None:
            return MetricsSnapshot()

        summary = self._engine.get_metrics_summary()

        total_req = summary.get("total_requests", 0)
        total_tokens = summary.get("total_tokens_used", 0)
        total_cost = summary.get("total_cost_usd", 0.0)
        histograms = summary.get("histograms", {})

        # Compute avg latency from histograms
        avg_latency = 0.0
        total_ms = 0.0
        total_count = 0
        for op_hist in histograms.values():
            total_ms += (op_hist.get("mean_ms", 0) or 0) * (op_hist.get("count", 0) or 0)
            total_count += op_hist.get("count", 0) or 0
        if total_count > 0:
            avg_latency = total_ms / total_count

        snap = MetricsSnapshot(
            total_requests=total_req,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            avg_latency_ms=round(avg_latency, 1),
            histograms=histograms,
        )

        # Record history for sparklines
        self._request_history.append(total_req)
        self._latency_history.append(avg_latency)
        self._token_history.append(total_tokens)
        self._cost_history.append(total_cost)
        for h in (self._request_history, self._latency_history, self._token_history, self._cost_history):
            if len(h) > 60:
                h.pop(0)

        return snap

    def get_sparkline_data(self, metric: str) -> list[float]:
        mapping = {
            "requests": self._request_history,
            "latency": self._latency_history,
            "tokens": self._token_history,
            "cost": self._cost_history,
        }
        data = mapping.get(metric, [])
        return [float(v) for v in data] if data else [0.0]

    # ── Traces ────────────────────────────────────────────────────

    async def get_traces(self, limit: int = 50) -> list[Dict[str, Any]]:
        if not self._initialized or self._engine is None:
            return []
        raw = self._engine.list_traces(limit)
        traces = []
        for t in raw:
            flat_steps: list[dict] = []
            if t.get("root_step"):
                self._flatten_steps(t["root_step"], flat_steps)
            traces.append({
                **t,
                "id": t.get("trace_id", t.get("id", "")),
                "steps": flat_steps,
            })
        return traces

    async def get_trace(self, trace_id: str) -> Dict[str, Any] | None:
        if not self._initialized or self._engine is None:
            return None
        return self._engine.get_trace(trace_id)

    async def get_trace_visualization(self, trace_id: str) -> Dict[str, Any] | None:
        if not self._initialized or self._engine is None:
            return None
        trace_obj = self._engine.trace_tracker.get_trace(trace_id)
        if not trace_obj:
            return None
        from glassbox_rag.trace.visualizer import VisualDebugger
        return {
            "trace_id": trace_id,
            "ascii": VisualDebugger.format_trace_ascii(trace_obj),
            "waterfall": VisualDebugger.format_waterfall(trace_obj),
            "cost_breakdown": VisualDebugger.format_cost_breakdown(trace_obj),
            "summary": VisualDebugger.get_trace_summary(trace_obj),
            "telemetry": self._engine.get_telemetry_status(),
        }

    async def get_anomalies(self, threshold: float = 2.0, limit: int = 100) -> Dict[str, Any]:
        if not self._initialized or self._engine is None:
            return {"anomalies": [], "traces_analyzed": 0}
        from glassbox_rag.trace.visualizer import VisualDebugger
        traces = self._engine.trace_tracker.list_traces(limit)
        anomalies = VisualDebugger.detect_anomalies(traces, threshold)
        return {"anomalies": anomalies, "traces_analyzed": len(traces)}

    async def compare_traces(self, id_a: str, id_b: str) -> Dict[str, Any] | None:
        if not self._initialized or self._engine is None:
            return None
        a = self._engine.trace_tracker.get_trace(id_a)
        b = self._engine.trace_tracker.get_trace(id_b)
        if not a or not b:
            return None
        from glassbox_rag.trace.visualizer import VisualDebugger
        return VisualDebugger.compare_traces(a, b)

    # ── Operations ────────────────────────────────────────────────

    async def run_retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        if not self._initialized or self._engine is None:
            return {"error": "Engine not initialised"}
        try:
            result = await self._engine.retrieve(query=query, top_k=top_k)
            return {
                "success": True,
                "query": query,
                "num_results": result.total_results,
                "strategy": result.strategy,
                "execution_time_ms": result.execution_time_ms,
                "documents": [
                    {"id": d.id, "content": d.content[:200], "score": d.score or 0.0}
                    for d in result.documents
                ],
                "trace_id": result.trace_id,
            }
        except Exception as exc:
            return {"error": str(exc)}

    async def run_ingest(self, documents: list[Dict[str, Any]]) -> Dict[str, Any]:
        if not self._initialized or self._engine is None:
            return {"error": "Engine not initialised"}
        try:
            result = await self._engine.ingest(documents=documents)
            return result
        except Exception as exc:
            return {"error": str(exc)}

    # ── Observability ─────────────────────────────────────────────

    def get_telemetry_status(self) -> Dict[str, Any]:
        if not self._initialized or self._engine is None:
            return {}
        return self._engine.get_telemetry_status()

    def get_cache_stats(self) -> Dict[str, Any]:
        if not self._initialized or self._engine is None:
            return {}
        return self._engine.get_cache_stats()

    def get_chunk_report(self) -> Dict[str, Any]:
        if not self._initialized or self._engine is None:
            return {}
        return self._engine.get_chunk_report()

    def get_hook_info(self) -> Dict[str, Any]:
        if not self._initialized or self._engine is None:
            return {}
        return self._engine.get_hook_info()

    # ── Evaluation & Experiments ───────────────────────────────────

    async def get_evaluations(self, limit: int = 20) -> list[Dict[str, Any]]:
        """Load evaluation reports from .glassbox/evaluations/."""
        import json as _json
        from pathlib import Path

        eval_dir = Path(".glassbox/evaluations")
        if not eval_dir.exists():
            return []
        files = sorted(eval_dir.glob("eval_*.json"), reverse=True)[:limit]
        reports: list[Dict[str, Any]] = []
        for f in files:
            try:
                data = _json.loads(f.read_text(encoding="utf-8"))
                reports.append(data)
            except Exception:
                pass
        return reports

    async def get_experiments(self, limit: int = 20) -> list[Dict[str, Any]]:
        """Load experiment records from .glassbox/experiments/."""
        import json as _json
        from pathlib import Path

        exp_dir = Path(".glassbox/experiments")
        if not exp_dir.exists():
            return []
        files = sorted(exp_dir.glob("exp_*.json"), reverse=True)[:limit]
        experiments: list[Dict[str, Any]] = []
        for f in files:
            try:
                data = _json.loads(f.read_text(encoding="utf-8"))
                experiments.append(data)
            except Exception:
                pass
        return experiments

    async def compare_experiments(self, id_a: str, id_b: str) -> Dict[str, Any] | None:
        """Compare two experiments using ExperimentTracker."""
        try:
            from glassbox_rag.experiments import ExperimentTracker
            tracker = ExperimentTracker.from_workspace()
            report = tracker.compare(id_a, id_b)
            return report.to_dict()
        except Exception:
            return None

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _flatten_steps(step: dict, out: list) -> None:
        out.append({
            "name": step.get("name", ""),
            "duration_ms": step.get("duration_ms", 0),
            "error": step.get("error"),
        })
        for child in step.get("children", []):
            EngineClient._flatten_steps(child, out)
