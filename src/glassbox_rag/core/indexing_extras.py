"""
Indexing Extras — separated concerns for lineage, drift, and freshness.

These responsibilities are decoupled from engine.ingest():
  - LineageEnricher: runs at the chunker/indexing layer, enriching chunk metadata.
  - DriftDetector: runs as a POST_INGEST callback hook, not inline.
  - FreshnessValidator: triggered on encoder config change, not on every ingest call.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Lineage Enricher — runs at chunker/indexing layer
# ═══════════════════════════════════════════════════════════════════


class LineageEnricher:
    """
    Enriches chunk metadata with lineage information.

    Called at the chunker/indexing layer BEFORE chunks reach the engine.
    Adds source hashes, ingest timestamps, and parent document IDs.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def enrich(
        self,
        chunks: list[Any],
        source_content: str,
        parent_doc_id: str = "",
    ) -> list[Any]:
        """
        Add lineage metadata to each chunk.

        Args:
            chunks: List of Chunk objects to enrich.
            source_content: Original document content for hashing.
            parent_doc_id: Optional parent document ID.

        Returns:
            The same chunks with enriched metadata.
        """
        if not self.enabled:
            return chunks

        source_hash = hashlib.sha256(source_content.encode()).hexdigest()[:16]
        ingest_ts = datetime.now(timezone.utc).isoformat()

        for chunk in chunks:
            chunk.metadata["lineage"] = {
                "source_hash": source_hash,
                "ingest_timestamp": ingest_ts,
                "parent_doc_id": parent_doc_id,
                "chunk_hash": hashlib.sha256(chunk.text.encode()).hexdigest()[:16],
            }

        return chunks


# ═══════════════════════════════════════════════════════════════════
#  Drift Detector — registered as POST_INGEST hook
# ═══════════════════════════════════════════════════════════════════


class DriftDetector:
    """
    Detects embedding distribution drift between ingestion batches.

    Algorithm: cosine similarity of embedding centroid + document count ratio.

    Registered as a POST_INGEST hook. Does NOT run inline in engine.ingest().
    """

    def __init__(
        self,
        threshold: float = 0.15,
        enabled: bool = True,
    ) -> None:
        self.threshold = threshold
        self.enabled = enabled
        self.__name__ = "DriftDetector"  # Required by HookManager logging
        self._baseline_centroid: Optional[np.ndarray] = None
        self._baseline_count: int = 0
        self._drift_events: list[dict[str, Any]] = []

    async def __call__(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Post-ingest hook: check for drift in the newly ingested batch.

        Expected context keys:
            - embeddings: np.ndarray of shape (n, dim)
            - chunk_count: int

        Adds drift_report to context if drift detected.
        """
        if not self.enabled:
            return ctx

        embeddings = ctx.get("embeddings")
        chunk_count = ctx.get("chunk_count", 0)

        if embeddings is None or len(embeddings) == 0:
            return ctx

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        # Compute centroid of new batch
        new_centroid = np.mean(embeddings, axis=0)

        if self._baseline_centroid is None:
            # First ingest — set baseline
            self._baseline_centroid = new_centroid
            self._baseline_count = chunk_count
            return ctx

        # Cosine similarity between centroids
        cos_sim = self._cosine_similarity(self._baseline_centroid, new_centroid)
        centroid_drift = 1.0 - cos_sim

        # Document count ratio
        count_ratio = abs(chunk_count - self._baseline_count) / max(self._baseline_count, 1)

        # Combined drift score (weighted average)
        drift_score = 0.7 * centroid_drift + 0.3 * min(count_ratio, 1.0)

        drift_detected = drift_score > self.threshold

        drift_report = {
            "drift_detected": drift_detected,
            "drift_score": round(drift_score, 4),
            "centroid_drift": round(centroid_drift, 4),
            "count_ratio": round(count_ratio, 4),
            "threshold": self.threshold,
            "baseline_count": self._baseline_count,
            "new_count": chunk_count,
        }

        if drift_detected:
            self._drift_events.append(drift_report)
            logger.warning(
                "Embedding drift detected: score=%.4f (threshold=%.4f)",
                drift_score, self.threshold,
            )

        # Update baseline (rolling)
        alpha = 0.3
        self._baseline_centroid = (
            alpha * new_centroid + (1 - alpha) * self._baseline_centroid
        )
        self._baseline_count = chunk_count

        ctx["drift_report"] = drift_report
        return ctx

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @property
    def drift_events(self) -> list[dict[str, Any]]:
        """Return recorded drift events."""
        return list(self._drift_events)


# ═══════════════════════════════════════════════════════════════════
#  Freshness Validator — triggered on encoder config change
# ═══════════════════════════════════════════════════════════════════


class FreshnessValidator:
    """
    Validates index freshness when encoder configuration changes.

    Detects when the encoder model/dim changes between runs and warns
    that re-indexing may be needed. Triggered at engine initialization,
    NOT on every ingest call.
    """

    def __init__(self, state_dir: str = ".glassbox") -> None:
        from pathlib import Path
        self._state_file = Path(state_dir) / "encoder_state.json"

    def check(
        self,
        encoder_name: str,
        encoder_model: str,
        embedding_dim: int,
    ) -> dict[str, Any]:
        """
        Check if encoder config has changed since last run.

        Returns a freshness report dict with:
            - fresh: bool — True if config unchanged
            - changed_fields: list of fields that changed
            - recommendation: str — action to take
        """
        import json
        from pathlib import Path

        current = {
            "encoder_name": encoder_name,
            "encoder_model": encoder_model,
            "embedding_dim": embedding_dim,
        }

        report: dict[str, Any] = {
            "fresh": True,
            "changed_fields": [],
            "recommendation": "",
            "current": current,
        }

        if not self._state_file.exists():
            # First run — save state
            self._save_state(current)
            report["recommendation"] = "First run — baseline state saved."
            return report

        try:
            previous = json.loads(self._state_file.read_text(encoding="utf-8"))
        except Exception:
            self._save_state(current)
            return report

        # Compare fields
        changed = []
        for key in ("encoder_name", "encoder_model", "embedding_dim"):
            if previous.get(key) != current.get(key):
                changed.append(key)

        if changed:
            report["fresh"] = False
            report["changed_fields"] = changed
            report["previous"] = previous
            report["recommendation"] = (
                f"Encoder config changed ({', '.join(changed)}). "
                "Existing embeddings may be incompatible. "
                "Consider re-indexing your document collection."
            )
            logger.warning(
                "Encoder freshness check: config changed (%s). "
                "Re-indexing recommended.",
                ", ".join(changed),
            )

        # Always update state
        self._save_state(current)
        return report

    def _save_state(self, state: dict[str, Any]) -> None:
        """Save encoder state to disk."""
        import json

        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(
            json.dumps(state, indent=2),
            encoding="utf-8",
        )
