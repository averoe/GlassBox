"""Unit tests for indexing extras (core/indexing_extras.py).

Covers LineageEnricher, DriftDetector, and FreshnessValidator
with both enabled and disabled states.
"""

import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from glassbox_rag.core.indexing_extras import (
    DriftDetector,
    FreshnessValidator,
    LineageEnricher,
)


# ── Helper fixtures ──────────────────────────────────────────────


@dataclass
class MockChunk:
    """Minimal chunk for testing lineage enrichment."""
    text: str = ""
    size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── LineageEnricher ──────────────────────────────────────────────


class TestLineageEnricher:
    def test_enrich_adds_lineage_fields(self):
        enricher = LineageEnricher(enabled=True)
        chunks = [
            MockChunk(text="hello world", size=11, metadata={"source": "doc.txt"}),
            MockChunk(text="second chunk", size=12, metadata={"source": "doc.txt"}),
        ]

        enricher.enrich(
            chunks,
            source_content="hello world second chunk",
            parent_doc_id="doc.txt",
        )

        for chunk in chunks:
            lineage = chunk.metadata.get("lineage")
            assert lineage is not None, "lineage key missing from metadata"
            assert lineage["parent_doc_id"] == "doc.txt"
            assert "chunk_hash" in lineage
            assert "source_hash" in lineage
            assert "ingest_timestamp" in lineage

    def test_enrich_disabled_is_noop(self):
        enricher = LineageEnricher(enabled=False)
        chunks = [
            MockChunk(text="hello", size=5, metadata={}),
        ]

        enricher.enrich(chunks, source_content="hello", parent_doc_id="doc.txt")

        assert "lineage_parent_doc" not in chunks[0].metadata

    def test_enrich_empty_chunks(self):
        enricher = LineageEnricher(enabled=True)
        # Should not raise
        enricher.enrich([], source_content="", parent_doc_id="doc.txt")


# ── DriftDetector ────────────────────────────────────────────────


class TestDriftDetector:
    @pytest.mark.asyncio
    async def test_callable_interface(self):
        """DriftDetector is callable as a hook fn (receives ctx dict, returns ctx dict)."""
        detector = DriftDetector(threshold=0.5, enabled=True)
        embeddings = np.random.randn(10, 128).astype(np.float32)

        ctx = {
            "embeddings": embeddings,
            "chunk_count": 10,
            "document_ids": [f"doc_{i}" for i in range(10)],
        }

        result = await detector(ctx)
        assert isinstance(result, dict)
        assert "embeddings" in result

    @pytest.mark.asyncio
    async def test_disabled_returns_context_unchanged(self):
        detector = DriftDetector(threshold=0.5, enabled=False)
        ctx = {"embeddings": np.zeros((5, 64)), "chunk_count": 5, "document_ids": []}

        result = await detector(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_drift_detection_with_uniform_embeddings(self):
        """Uniform embeddings should show low/no drift."""
        detector = DriftDetector(threshold=0.5, enabled=True)
        # Identical embeddings → zero variance → no drift
        embeddings = np.ones((10, 64), dtype=np.float32)
        ctx = {
            "embeddings": embeddings,
            "chunk_count": 10,
            "document_ids": [f"d{i}" for i in range(10)],
        }

        result = await detector(ctx)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_drift_detection_with_diverse_embeddings(self):
        """Diverse embeddings should trigger drift warning (logged, not raised)."""
        detector = DriftDetector(threshold=0.01, enabled=True)
        # Random embeddings → high variance
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((20, 128)).astype(np.float32)
        ctx = {
            "embeddings": embeddings,
            "chunk_count": 20,
            "document_ids": [f"d{i}" for i in range(20)],
        }

        result = await detector(ctx)
        assert isinstance(result, dict)

    def test_has_callable_interface(self):
        """DriftDetector must be callable (used as hook fn)."""
        detector = DriftDetector(threshold=0.5)
        assert callable(detector)


# ── FreshnessValidator ───────────────────────────────────────────


class TestFreshnessValidator:
    def test_first_run_is_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = FreshnessValidator(state_dir=tmpdir)
            report = validator.check(
                encoder_name="openai",
                encoder_model="text-embedding-3-small",
                embedding_dim=1536,
            )
            assert report["fresh"] is True
            assert "First run" in report["recommendation"]

    def test_same_config_is_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = FreshnessValidator(state_dir=tmpdir)
            # First run
            validator.check(
                encoder_name="openai",
                encoder_model="text-embedding-3-small",
                embedding_dim=1536,
            )
            # Second run with same config
            report = validator.check(
                encoder_name="openai",
                encoder_model="text-embedding-3-small",
                embedding_dim=1536,
            )
            assert report["fresh"] is True
            assert report["changed_fields"] == []

    def test_changed_config_is_stale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = FreshnessValidator(state_dir=tmpdir)
            # First run
            validator.check(
                encoder_name="openai",
                encoder_model="text-embedding-3-small",
                embedding_dim=1536,
            )
            # Second run with DIFFERENT model
            report = validator.check(
                encoder_name="openai",
                encoder_model="text-embedding-3-large",
                embedding_dim=3072,
            )
            assert report["fresh"] is False
            assert "re-index" in report["recommendation"].lower()

    def test_default_state_dir(self):
        """FreshnessValidator can use default .glassbox directory."""
        validator = FreshnessValidator()
        assert validator._state_file is not None
