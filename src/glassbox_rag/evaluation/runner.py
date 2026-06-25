"""
Evaluation Runner — runs batch evaluation and produces structured reports.

Accepts a pipeline, a dataset, and a metric list. Returns a structured
EvaluationReport with per-query scores and aggregate scores. Writes results
to `.glassbox/evaluations/` with timestamp and pipeline version.

Uses asyncio.gather with Semaphore for concurrent query execution.

Usage:
    runner = EvaluationRunner(pipeline=engine)
    report = await runner.run(dataset, metrics=["recall@5", "faithfulness", "ndcg"])
    print(report.summary())
    regression = report.compare(other_report)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from glassbox_rag.evaluation.datasets import GoldenDataset, GoldenEntry
from glassbox_rag.evaluation import retrieval as retrieval_metrics
from glassbox_rag.evaluation import generation as generation_metrics
from glassbox_rag.utils.logging import get_logger

if TYPE_CHECKING:
    from glassbox_rag.config import EvaluationConfig

logger = get_logger(__name__)


@dataclass
class QueryScore:
    """Evaluation scores for a single query."""

    query: str
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionReport:
    """Result of comparing two evaluation reports."""

    passed: bool
    threshold: float
    per_metric: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "threshold": self.threshold,
            "per_metric": self.per_metric,
        }

    def summary(self) -> str:
        lines = ["Regression Report:"]
        lines.append(f"  Overall: {'PASS ✓' if self.passed else 'FAIL ✗'}")
        lines.append(f"  Threshold: {self.threshold}")
        for metric, info in self.per_metric.items():
            status = "✓" if info.get("passed") else "✗"
            delta = info.get("delta", 0)
            lines.append(f"  {status} {metric}: {delta:+.4f}")
        return "\n".join(lines)


@dataclass
class EvaluationReport:
    """Structured evaluation report with per-query and aggregate scores."""

    timestamp: str = ""
    pipeline_version: int = 0
    dataset_name: str = ""
    metrics_requested: list[str] = field(default_factory=list)
    per_query_scores: list[QueryScore] = field(default_factory=list)
    aggregate_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Format a human-readable summary table."""
        lines = [
            f"Evaluation Report — {self.timestamp}",
            f"Pipeline Version: v{self.pipeline_version}",
            f"Queries Evaluated: {len(self.per_query_scores)}",
            f"Metrics: {', '.join(self.metrics_requested)}",
            "",
            "Aggregate Scores:",
        ]
        for metric, score in sorted(self.aggregate_scores.items()):
            lines.append(f"  {metric:<25} {score:.4f}")

        if self.per_query_scores:
            lines.append("")
            lines.append("Per-Query Breakdown:")
            for qs in self.per_query_scores[:10]:
                query_preview = qs.query[:50] + ("…" if len(qs.query) > 50 else "")
                scores_str = ", ".join(f"{k}={v:.3f}" for k, v in qs.scores.items())
                lines.append(f"  '{query_preview}' → {scores_str}")
            if len(self.per_query_scores) > 10:
                lines.append(f"  … and {len(self.per_query_scores) - 10} more")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "pipeline_version": self.pipeline_version,
            "dataset_name": self.dataset_name,
            "metrics_requested": self.metrics_requested,
            "per_query_scores": [
                {"query": qs.query, "scores": qs.scores, "metadata": qs.metadata}
                for qs in self.per_query_scores
            ],
            "aggregate_scores": self.aggregate_scores,
            "metadata": self.metadata,
        }

    def compare(
        self,
        other: "EvaluationReport",
        threshold: float = 0.05,
    ) -> RegressionReport:
        """
        Compare two reports and flag metric regressions.

        A metric "regresses" if it drops by more than `threshold`.
        """
        per_metric: dict[str, dict[str, Any]] = {}
        all_passed = True

        all_metrics = set(self.aggregate_scores.keys()) | set(other.aggregate_scores.keys())
        for metric in sorted(all_metrics):
            score_self = self.aggregate_scores.get(metric, 0.0)
            score_other = other.aggregate_scores.get(metric, 0.0)
            delta = score_other - score_self
            passed = delta >= -threshold
            if not passed:
                all_passed = False
            per_metric[metric] = {
                "before": score_self,
                "after": score_other,
                "delta": round(delta, 4),
                "passed": passed,
            }

        return RegressionReport(
            passed=all_passed,
            threshold=threshold,
            per_metric=per_metric,
        )


def _parse_metric_name(metric_str: str) -> tuple[str, int | None]:
    """Parse metric string like 'recall@5' into ('recall', 5)."""
    match = re.match(r"^(\w+)@(\d+)$", metric_str)
    if match:
        return match.group(1), int(match.group(2))
    return metric_str, None


class EvaluationRunner:
    """
    Runs batch evaluation with concurrent query processing.

    Uses asyncio.Semaphore for concurrency limiting (default: 5).
    """

    def __init__(
        self,
        pipeline: Any,
        concurrency_limit: int = 5,
        workspace_path: str = ".",
        eval_config: "EvaluationConfig | None" = None,
    ) -> None:
        """
        Args:
            pipeline: GlassBoxEngine instance.
            concurrency_limit: Max concurrent query evaluations.
            workspace_path: Path for writing results.
            eval_config: Optional EvaluationConfig — overrides concurrency_limit
                         and provides an evaluation-specific LLM generator.
        """
        self.pipeline = pipeline
        self.workspace = Path(workspace_path).resolve()
        self._eval_dir = self.workspace / ".glassbox" / "evaluations"
        self._eval_dir.mkdir(parents=True, exist_ok=True)

        # Wire concurrency_limit from EvaluationConfig if provided
        if eval_config is not None:
            self.concurrency_limit = eval_config.concurrency_limit
        else:
            self.concurrency_limit = concurrency_limit

        # Evaluation-specific generator (fallback to pipeline generator)
        self._eval_generator = None
        self._eval_config = eval_config

    async def run(
        self,
        dataset: GoldenDataset,
        metrics: list[str] | None = None,
    ) -> EvaluationReport:
        """
        Run batch evaluation across a golden dataset.

        Uses asyncio.gather with Semaphore for concurrent execution.

        Args:
            dataset: Golden dataset with query/answer/doc_id entries.
            metrics: List of metrics to compute (e.g. ["recall@5", "faithfulness"]).

        Returns:
            EvaluationReport with per-query and aggregate scores.
        """
        if metrics is None:
            metrics = ["recall@5", "precision@5", "faithfulness"]

        now = datetime.now(timezone.utc)
        pipeline_version = self._get_pipeline_version()

        # Parse metrics into retrieval vs generation
        retrieval_metric_list = []
        generation_metric_list = []
        for m in metrics:
            name, k = _parse_metric_name(m)
            if name in ("recall", "precision", "mrr", "ndcg", "hit_rate"):
                retrieval_metric_list.append((m, name, k or 5))
            else:
                generation_metric_list.append((m, name))

        # Run evaluations concurrently
        sem = asyncio.Semaphore(self.concurrency_limit)
        tasks = [
            self._evaluate_entry(entry, retrieval_metric_list, generation_metric_list, sem)
            for entry in dataset
        ]
        query_scores = await asyncio.gather(*tasks)

        # Compute aggregates
        aggregate: dict[str, list[float]] = {}
        for qs in query_scores:
            for metric_name, score in qs.scores.items():
                aggregate.setdefault(metric_name, []).append(score)

        aggregate_scores = {
            metric: sum(scores) / len(scores) if scores else 0.0
            for metric, scores in aggregate.items()
        }

        report = EvaluationReport(
            timestamp=now.isoformat(),
            pipeline_version=pipeline_version,
            dataset_name=str(dataset),
            metrics_requested=metrics,
            per_query_scores=query_scores,
            aggregate_scores=aggregate_scores,
        )

        # Write to disk
        self._save_report(report)

        logger.info(
            "Evaluation complete: %d queries, %d metrics, pipeline v%d",
            len(query_scores), len(metrics), pipeline_version,
        )

        return report

    async def _ensure_eval_generator(self) -> Any:
        """Return the eval-specific generator, or fall back to pipeline generator."""
        if self._eval_generator is not None:
            return self._eval_generator

        cfg = self._eval_config
        if cfg and cfg.backend and cfg.api_key:
            # Build a dedicated eval LLM generator
            from glassbox_rag.core.generator import LLMGenerator, GenerationConfig

            backend_config: dict[str, Any] = {
                "api_key": cfg.api_key.get_secret_value(),
                "model": cfg.model or "gpt-4o-mini",
            }
            if cfg.base_url:
                backend_config["base_url"] = cfg.base_url

            gen_config = GenerationConfig(model=cfg.model or "gpt-4o-mini")
            self._eval_generator = LLMGenerator(
                backend_type=cfg.backend,
                backend_config=backend_config,
                generation_config=gen_config,
            )
            await self._eval_generator.initialize()
            logger.info(
                "Evaluation-specific LLM generator initialised: %s/%s",
                cfg.backend, cfg.model,
            )
            return self._eval_generator

        # Fallback: use the pipeline's generator
        return getattr(self.pipeline, "_generator", None)

    async def _evaluate_entry(
        self,
        entry: GoldenEntry,
        retrieval_metrics_list: list[tuple[str, str, int]],
        generation_metrics_list: list[tuple[str, str]],
        sem: asyncio.Semaphore,
    ) -> QueryScore:
        """Evaluate a single query entry with semaphore-limited concurrency."""
        async with sem:
            scores: dict[str, float] = {}
            metadata: dict[str, Any] = {}

            try:
                # Run retrieval
                retrieval_result = await self.pipeline.retrieve(
                    query=entry.query, top_k=10,
                )
                retrieved_ids = [doc.id for doc in retrieval_result.documents]
                context_docs = [doc.content for doc in retrieval_result.documents]

                # Retrieval metrics
                for metric_str, name, k in retrieval_metrics_list:
                    if name == "recall":
                        scores[metric_str] = retrieval_metrics.recall_at_k(
                            retrieved_ids, entry.relevant_doc_ids, k
                        )
                    elif name == "precision":
                        scores[metric_str] = retrieval_metrics.precision_at_k(
                            retrieved_ids, entry.relevant_doc_ids, k
                        )
                    elif name == "ndcg":
                        scores[metric_str] = retrieval_metrics.ndcg(
                            retrieved_ids, entry.relevant_doc_ids, k
                        )
                    elif name == "hit_rate":
                        scores[metric_str] = retrieval_metrics.hit_rate(
                            retrieved_ids, entry.relevant_doc_ids, k
                        )

                # Generation metrics (require LLM)
                if generation_metrics_list:
                    try:
                        gen_result = await self.pipeline.generate(query=entry.query)
                        answer = gen_result.get("answer", "") if isinstance(gen_result, dict) else gen_result.answer

                        generator = await self._ensure_eval_generator()
                        if generator:
                            for metric_str, name in generation_metrics_list:
                                if name == "faithfulness":
                                    scores[metric_str] = await generation_metrics.faithfulness(
                                        answer, context_docs, generator,
                                    )
                                elif name == "hallucination_rate":
                                    scores[metric_str] = await generation_metrics.hallucination_rate(
                                        answer, context_docs, generator,
                                    )
                                elif name == "groundedness":
                                    scores[metric_str] = await generation_metrics.groundedness(
                                        answer, context_docs, generator,
                                    )
                                elif name == "context_relevance":
                                    scores[metric_str] = await generation_metrics.context_relevance(
                                        entry.query, context_docs, generator,
                                    )
                                elif name == "llm_as_judge":
                                    scores[metric_str] = await generation_metrics.llm_as_judge(
                                        entry.query, answer, context_docs, generator,
                                    )
                    except Exception as e:
                        logger.warning("Generation metrics failed for '%s': %s", entry.query[:50], e)

            except Exception as e:
                logger.warning("Evaluation failed for '%s': %s", entry.query[:50], e)
                metadata["error"] = str(e)

            return QueryScore(query=entry.query, scores=scores, metadata=metadata)

    def _get_pipeline_version(self) -> int:
        """Get current pipeline version from .glassbox/versions/."""
        current_file = self.workspace / ".glassbox" / "versions" / "current.json"
        if current_file.exists():
            try:
                data = json.loads(current_file.read_text(encoding="utf-8"))
                return data.get("version", 0)
            except Exception:
                pass
        return 0

    def _save_report(self, report: EvaluationReport) -> None:
        """Save evaluation report to .glassbox/evaluations/."""
        filename = f"eval_{report.timestamp.replace(':', '-').replace('+', '_')}.json"
        filepath = self._eval_dir / filename
        filepath.write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Evaluation report saved: %s", filepath)
