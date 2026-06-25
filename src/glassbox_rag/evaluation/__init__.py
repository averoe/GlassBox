"""
GlassBox RAG Evaluation System.

Provides retrieval metrics, generation metrics, golden datasets,
evaluation runner, and regression testing.

Usage:
    from glassbox_rag.evaluation import EvaluationRunner, GoldenDataset, EvaluationReport

    dataset = GoldenDataset.from_yaml(".glassbox/datasets/golden.yaml")
    runner = EvaluationRunner(pipeline=engine)
    report = await runner.run(dataset, metrics=["recall@5", "faithfulness"])
    print(report.summary())
"""

from glassbox_rag.evaluation.datasets import GoldenDataset
from glassbox_rag.evaluation.runner import EvaluationRunner, EvaluationReport, RegressionReport

__all__ = [
    "GoldenDataset",
    "EvaluationRunner",
    "EvaluationReport",
    "RegressionReport",
]
