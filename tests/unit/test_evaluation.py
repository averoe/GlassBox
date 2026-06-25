"""Unit tests for the evaluation system (evaluation/runner.py, evaluation/datasets.py).

Covers EvaluationRunner config wiring, EvaluationReport comparison,
GoldenDataset serialization, and EvaluationConfig integration.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from glassbox_rag.evaluation.datasets import GoldenDataset, GoldenEntry
from glassbox_rag.evaluation.runner import (
    EvaluationReport,
    EvaluationRunner,
    QueryScore,
    RegressionReport,
    _parse_metric_name,
)


# ── GoldenDataset ────────────────────────────────────────────────


class TestGoldenDataset:
    def test_create_empty(self):
        ds = GoldenDataset()
        assert len(ds) == 0

    def test_add_entries(self):
        ds = GoldenDataset()
        ds.add("What is RAG?", expected_answer="Retrieval-Augmented Generation")
        ds.add("How does chunking work?", relevant_doc_ids=["doc1"])
        assert len(ds) == 2
        assert ds[0].query == "What is RAG?"

    def test_round_trip_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_dataset.yaml"

            ds = GoldenDataset()
            ds.add("Query 1", expected_answer="Answer 1", relevant_doc_ids=["d1"])
            ds.add("Query 2", expected_answer="Answer 2")
            ds.to_yaml(yaml_path)

            loaded = GoldenDataset.from_yaml(yaml_path)
            assert len(loaded) == 2
            assert loaded[0].query == "Query 1"
            assert loaded[0].expected_answer == "Answer 1"
            assert loaded[0].relevant_doc_ids == ["d1"]

    def test_from_yaml_missing_file(self):
        with pytest.raises(FileNotFoundError):
            GoldenDataset.from_yaml("/nonexistent/path.yaml")

    def test_iteration(self):
        ds = GoldenDataset([
            GoldenEntry(query="q1"),
            GoldenEntry(query="q2"),
        ])
        queries = [e.query for e in ds]
        assert queries == ["q1", "q2"]


# ── EvaluationReport ─────────────────────────────────────────────


class TestEvaluationReport:
    def test_summary(self):
        report = EvaluationReport(
            aggregate_scores={"recall@5": 0.85, "faithfulness": 0.92},
            per_query_scores=[
                QueryScore(query="q1", scores={"recall@5": 0.8}),
            ],
        )
        text = report.summary()
        assert "recall@5" in text
        assert "0.8500" in text

    def test_to_dict(self):
        report = EvaluationReport(
            aggregate_scores={"recall@5": 0.85},
            per_query_scores=[
                QueryScore(query="q1", scores={"recall@5": 0.8}),
            ],
            pipeline_version=3,
        )
        d = report.to_dict()
        assert d["pipeline_version"] == 3
        assert d["aggregate_scores"]["recall@5"] == 0.85

    def test_compare_no_regression(self):
        report_a = EvaluationReport(
            aggregate_scores={"recall@5": 0.80},
        )
        report_b = EvaluationReport(
            aggregate_scores={"recall@5": 0.82},
        )
        regression = report_a.compare(report_b, threshold=0.05)
        assert regression.passed is True
        assert regression.per_metric["recall@5"]["delta"] == pytest.approx(0.02, abs=0.001)

    def test_compare_with_regression(self):
        report_a = EvaluationReport(
            aggregate_scores={"recall@5": 0.90, "faithfulness": 0.95},
        )
        report_b = EvaluationReport(
            aggregate_scores={"recall@5": 0.80, "faithfulness": 0.96},
        )
        regression = report_a.compare(report_b, threshold=0.05)
        assert regression.passed is False
        assert regression.per_metric["recall@5"]["passed"] is False
        assert regression.per_metric["faithfulness"]["passed"] is True


# ── RegressionReport ─────────────────────────────────────────────


class TestRegressionReport:
    def test_summary(self):
        rr = RegressionReport(
            passed=False,
            threshold=0.05,
            per_metric={
                "recall@5": {"before": 0.9, "after": 0.8, "delta": -0.1, "passed": False},
            },
        )
        text = rr.summary()
        assert "FAIL" in text or "fail" in text.lower()

    def test_to_dict(self):
        rr = RegressionReport(
            passed=True,
            threshold=0.05,
            per_metric={},
        )
        d = rr.to_dict()
        assert d["passed"] is True
        assert d["threshold"] == 0.05


# ── Metric parsing ───────────────────────────────────────────────


class TestMetricParsing:
    def test_parse_with_k(self):
        name, k = _parse_metric_name("recall@5")
        assert name == "recall"
        assert k == 5

    def test_parse_without_k(self):
        name, k = _parse_metric_name("faithfulness")
        assert name == "faithfulness"
        assert k is None

    def test_parse_precision(self):
        name, k = _parse_metric_name("precision@10")
        assert name == "precision"
        assert k == 10


# ── EvaluationRunner ─────────────────────────────────────────────


class TestEvaluationRunner:
    def test_default_concurrency(self):
        mock_pipeline = MagicMock()
        runner = EvaluationRunner(pipeline=mock_pipeline)
        assert runner.concurrency_limit == 5

    def test_concurrency_from_eval_config(self):
        """EvaluationConfig.concurrency_limit overrides the default."""
        from glassbox_rag.config import EvaluationConfig

        mock_pipeline = MagicMock()
        eval_config = EvaluationConfig(concurrency_limit=10)

        runner = EvaluationRunner(
            pipeline=mock_pipeline,
            eval_config=eval_config,
        )
        assert runner.concurrency_limit == 10

    def test_explicit_concurrency_when_no_config(self):
        mock_pipeline = MagicMock()
        runner = EvaluationRunner(pipeline=mock_pipeline, concurrency_limit=3)
        assert runner.concurrency_limit == 3

    @pytest.mark.asyncio
    async def test_ensure_eval_generator_fallback(self):
        """Without eval config, falls back to pipeline's _generator."""
        mock_pipeline = MagicMock()
        mock_pipeline._generator = MagicMock()

        runner = EvaluationRunner(pipeline=mock_pipeline)
        gen = await runner._ensure_eval_generator()
        assert gen is mock_pipeline._generator

    @pytest.mark.asyncio
    async def test_ensure_eval_generator_none_when_no_pipeline_gen(self):
        """Returns None if neither eval config nor pipeline generator exist."""
        mock_pipeline = MagicMock(spec=[])  # no _generator attribute

        runner = EvaluationRunner(pipeline=mock_pipeline)
        gen = await runner._ensure_eval_generator()
        assert gen is None

    def test_eval_dir_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_pipeline = MagicMock()
            runner = EvaluationRunner(
                pipeline=mock_pipeline,
                workspace_path=tmpdir,
            )
            assert (Path(tmpdir) / ".glassbox" / "evaluations").exists()
