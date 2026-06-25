"""
Experiment Tracker — records and compares pipeline runs.

Every pipeline run is an experiment. Experiments link pipeline config,
dataset, trace, and evaluation artifacts together to make runs comparable.

Stored in `.glassbox/experiments/<timestamp>.json`.

Usage:
    from glassbox_rag import ExperimentTracker

    tracker = ExperimentTracker.from_workspace()
    tracker.record(config_version=2, dataset_version="golden_v1",
                   trace_id="abc", eval_results={"faithfulness": 0.92})
    experiments = tracker.list()
    tracker.compare("exp_20260624_143201", "exp_20260624_150000")
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentRecord:
    """A single experiment record."""

    experiment_id: str
    timestamp: str
    pipeline_version: int
    dataset_version: str = ""
    git_commit: str | None = None
    snapshot_id: str | None = None
    trace_id: str = ""
    evaluation_results: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ComparisonReport:
    """Result of comparing two experiments."""

    experiment_a: str
    experiment_b: str
    metric_changes: dict[str, dict[str, float]] = field(default_factory=dict)
    config_version_change: tuple[int, int] = (0, 0)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_a": self.experiment_a,
            "experiment_b": self.experiment_b,
            "metric_changes": self.metric_changes,
            "config_version_change": list(self.config_version_change),
            "summary": self.summary,
        }


class ExperimentTracker:
    """
    Tracks pipeline experiments in `.glassbox/experiments/`.

    Each experiment records config version, dataset, trace reference,
    evaluation results, and optional git commit hash.
    """

    def __init__(self, workspace_path: Path) -> None:
        self.workspace = workspace_path
        self.experiments_dir = workspace_path / ".glassbox" / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_workspace(cls, path: str = ".") -> "ExperimentTracker":
        """Create an ExperimentTracker for the given workspace directory."""
        return cls(Path(path).resolve())

    def record(
        self,
        config_version: int,
        dataset_version: str = "",
        trace_id: str = "",
        eval_results: dict[str, Any] | None = None,
        git_commit: str | None = None,
        snapshot_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        """
        Record a new experiment.

        Returns the created ExperimentRecord.
        """
        now = datetime.now(timezone.utc)
        exp_id = f"exp_{now.strftime('%Y%m%d_%H%M%S')}"

        # Auto-detect git commit if not provided
        if git_commit is None:
            git_commit = self._get_git_commit()

        experiment = ExperimentRecord(
            experiment_id=exp_id,
            timestamp=now.isoformat(),
            pipeline_version=config_version,
            dataset_version=dataset_version,
            git_commit=git_commit,
            snapshot_id=snapshot_id,
            trace_id=trace_id,
            evaluation_results=eval_results or {},
            metadata=metadata or {},
        )

        # Write to file
        exp_file = self.experiments_dir / f"{exp_id}.json"
        exp_file.write_text(
            json.dumps(experiment.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        logger.info(
            "Experiment recorded: %s (pipeline v%d, trace %s)",
            exp_id, config_version, trace_id[:12] if trace_id else "none",
        )
        return experiment

    def list(self, limit: int = 100) -> list[ExperimentRecord]:
        """List experiments, most recent first."""
        experiments = []
        files = sorted(self.experiments_dir.glob("exp_*.json"), reverse=True)
        for f in files[:limit]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                experiments.append(ExperimentRecord.from_dict(data))
            except Exception as e:
                logger.warning("Failed to load experiment %s: %s", f.name, e)
        return experiments

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        """Get a specific experiment by ID."""
        exp_file = self.experiments_dir / f"{experiment_id}.json"
        if not exp_file.exists():
            return None
        try:
            data = json.loads(exp_file.read_text(encoding="utf-8"))
            return ExperimentRecord.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load experiment %s: %s", experiment_id, e)
            return None

    def compare(self, exp_id_a: str, exp_id_b: str) -> ComparisonReport:
        """
        Compare two experiments side-by-side.

        Returns a ComparisonReport with metric changes and summary.
        """
        exp_a = self.get(exp_id_a)
        exp_b = self.get(exp_id_b)

        if exp_a is None:
            raise ValueError(f"Experiment '{exp_id_a}' not found")
        if exp_b is None:
            raise ValueError(f"Experiment '{exp_id_b}' not found")

        # Compare metrics
        metric_changes: dict[str, dict[str, float]] = {}
        all_metrics = set(exp_a.evaluation_results.keys()) | set(exp_b.evaluation_results.keys())
        for metric in all_metrics:
            val_a = exp_a.evaluation_results.get(metric)
            val_b = exp_b.evaluation_results.get(metric)
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                delta = val_b - val_a
                metric_changes[metric] = {
                    "before": float(val_a),
                    "after": float(val_b),
                    "delta": round(delta, 4),
                    "improved": delta > 0,
                }

        # Summary
        improved = sum(1 for m in metric_changes.values() if m.get("improved"))
        degraded = len(metric_changes) - improved
        summary = (
            f"Comparing {exp_id_a} (v{exp_a.pipeline_version}) → "
            f"{exp_id_b} (v{exp_b.pipeline_version}): "
            f"{improved} improved, {degraded} degraded out of {len(metric_changes)} metrics"
        )

        return ComparisonReport(
            experiment_a=exp_id_a,
            experiment_b=exp_id_b,
            metric_changes=metric_changes,
            config_version_change=(exp_a.pipeline_version, exp_b.pipeline_version),
            summary=summary,
        )

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(self.workspace),
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None
