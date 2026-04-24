"""
Document Deduplication — detects and removes duplicate/near-duplicate content.

Supports three strategies:
  1. Exact: SHA-256 content hash comparison.
  2. Fuzzy: SimHash-based near-duplicate detection.
  3. Semantic: Cosine similarity on embeddings (requires encoder).

Usage:
    dedup = DocumentDeduplicator(strategy="fuzzy", threshold=0.9)
    unique_docs, dupes = dedup.deduplicate(documents)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DuplicateInfo:
    """Information about a detected duplicate."""

    original_index: int
    duplicate_index: int
    similarity: float
    strategy: str


class DocumentDeduplicator:
    """
    Content-aware document deduplication.

    Detects exact and near-duplicate documents before they enter
    the vector store, saving storage and improving retrieval quality.
    """

    def __init__(
        self,
        strategy: str = "exact",
        threshold: float = 0.95,
        hash_bits: int = 128,
    ) -> None:
        """
        Args:
            strategy: 'exact', 'fuzzy', or 'semantic'.
            threshold: Similarity threshold for fuzzy/semantic (0.0-1.0).
            hash_bits: Number of bits for SimHash (fuzzy strategy).
        """
        if strategy not in ("exact", "fuzzy", "semantic"):
            raise ValueError(f"Unknown dedup strategy: {strategy}")
        self.strategy = strategy
        self.threshold = threshold
        self.hash_bits = hash_bits
        self._seen_hashes: Set[str] = set()

    def deduplicate(
        self,
        documents: list[dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
    ) -> tuple[list[dict[str, Any]], list[DuplicateInfo]]:
        """
        Remove duplicate documents.

        Args:
            documents: List of document dicts with 'content' key.
            embeddings: Optional pre-computed embeddings (required for 'semantic').

        Returns:
            (unique_documents, duplicate_info_list)
        """
        if self.strategy == "exact":
            return self._dedup_exact(documents)
        elif self.strategy == "fuzzy":
            return self._dedup_fuzzy(documents)
        elif self.strategy == "semantic":
            if embeddings is None:
                raise ValueError("Semantic dedup requires embeddings")
            return self._dedup_semantic(documents, embeddings)
        return documents, []

    def deduplicate_texts(
        self,
        texts: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> tuple[list[str], list[int], list[DuplicateInfo]]:
        """
        Convenience method for deduplicating plain text strings.

        Returns:
            (unique_texts, unique_indices, duplicate_info)
        """
        docs = [{"content": t} for t in texts]
        unique_docs, dupes = self.deduplicate(docs, embeddings)
        dupe_indices = {d.duplicate_index for d in dupes}
        unique_indices = [i for i in range(len(texts)) if i not in dupe_indices]
        unique_texts = [texts[i] for i in unique_indices]
        return unique_texts, unique_indices, dupes

    def is_duplicate(self, content: str) -> bool:
        """
        Stateful check: has this exact content been seen before?

        Uses a rolling hash set. Good for streaming ingest.
        """
        h = self._content_hash(content)
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False

    def reset(self) -> None:
        """Clear the seen-hashes set."""
        self._seen_hashes.clear()

    # ── Exact dedup ──────────────────────────────────────────

    def _dedup_exact(
        self, documents: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[DuplicateInfo]]:
        seen: dict[str, int] = {}
        unique: list[dict[str, Any]] = []
        duplicates: list[DuplicateInfo] = []

        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            h = self._content_hash(content)
            if h in seen:
                duplicates.append(DuplicateInfo(
                    original_index=seen[h],
                    duplicate_index=i,
                    similarity=1.0,
                    strategy="exact",
                ))
            else:
                seen[h] = i
                unique.append(doc)

        if duplicates:
            logger.info(
                "Exact dedup: removed %d duplicates from %d documents",
                len(duplicates), len(documents),
            )
        return unique, duplicates

    # ── Fuzzy dedup (SimHash) ────────────────────────────────

    def _dedup_fuzzy(
        self, documents: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[DuplicateInfo]]:
        hashes = [self._simhash(doc.get("content", "")) for doc in documents]
        
        # Fast path for large batches
        if len(documents) > 10000:
            return self._dedup_fuzzy_lsh(documents, hashes)

        unique: list[dict[str, Any]] = []
        duplicates: list[DuplicateInfo] = []
        unique_hashes: list[int] = []

        for i, h in enumerate(hashes):
            is_dupe = False
            for j, existing_hash in enumerate(unique_hashes):
                sim = self._simhash_similarity(h, existing_hash)
                if sim >= self.threshold:
                    duplicates.append(DuplicateInfo(
                        original_index=j,  # mapped to unique's index but realistically we just want the index
                        duplicate_index=i,
                        similarity=sim,
                        strategy="fuzzy",
                    ))
                    is_dupe = True
                    break

            if not is_dupe:
                unique_hashes.append(h)
                unique.append(documents[i])

        if duplicates:
            logger.info(
                "Fuzzy dedup (threshold=%.2f): removed %d duplicates from %d documents",
                self.threshold, len(duplicates), len(documents),
            )
        return unique, duplicates

    def _dedup_fuzzy_lsh(
        self, documents: list[dict[str, Any]], hashes: list[int]
    ) -> tuple[list[dict[str, Any]], list[DuplicateInfo]]:
        """LSH band-bucketing for SimHash."""
        num_bands = 4
        band_bits = self.hash_bits // num_bands
        mask = (1 << band_bits) - 1

        buckets: dict[tuple[int, int], list[int]] = {}
        for i, h in enumerate(hashes):
            for b in range(num_bands):
                band_val = (h >> (b * band_bits)) & mask
                band_key = (b, band_val)
                if band_key not in buckets:
                    buckets[band_key] = []
                buckets[band_key].append(i)

        candidates: set[tuple[int, int]] = set()
        for b_indices in buckets.values():
            if len(b_indices) > 1:
                for idx1 in range(len(b_indices)):
                    for idx2 in range(idx1 + 1, len(b_indices)):
                        i, j = b_indices[idx1], b_indices[idx2]
                        if i > j:
                            i, j = j, i
                        candidates.add((i, j))

        duplicates: list[DuplicateInfo] = []
        is_dupe = [False] * len(documents)
        sorted_candidates = sorted(list(candidates), key=lambda x: (x[1], x[0]))

        for i, j in sorted_candidates:
            if is_dupe[j] or is_dupe[i]: 
                continue
            sim = self._simhash_similarity(hashes[i], hashes[j])
            if sim >= self.threshold:
                duplicates.append(DuplicateInfo(
                    original_index=i,
                    duplicate_index=j,
                    similarity=sim,
                    strategy="fuzzy",
                ))
                is_dupe[j] = True

        unique = [doc for idx, doc in enumerate(documents) if not is_dupe[idx]]
        if duplicates:
            logger.info(
                "Fuzzy LSH dedup (threshold=%.2f): removed %d duplicates from %d documents",
                self.threshold, len(duplicates), len(documents),
            )
        return unique, duplicates

    # ── Semantic dedup ───────────────────────────────────────

    def _dedup_semantic(
        self,
        documents: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> tuple[list[dict[str, Any]], list[DuplicateInfo]]:
        # Fast path for large batches
        if len(documents) > 10000:
            return self._dedup_semantic_lsh(documents, embeddings)

        unique_indices: list[int] = []
        unique: list[dict[str, Any]] = []
        duplicates: list[DuplicateInfo] = []

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        normalized = embeddings / norms

        for i in range(len(documents)):
            is_dupe = False
            for j in unique_indices:
                sim = float(np.dot(normalized[i], normalized[j]))
                if sim >= self.threshold:
                    duplicates.append(DuplicateInfo(
                        original_index=j,
                        duplicate_index=i,
                        similarity=sim,
                        strategy="semantic",
                    ))
                    is_dupe = True
                    break

            if not is_dupe:
                unique_indices.append(i)
                unique.append(documents[i])

        if duplicates:
            logger.info(
                "Semantic dedup (threshold=%.2f): removed %d duplicates from %d documents",
                self.threshold, len(duplicates), len(documents),
            )
        return unique, duplicates

    def _dedup_semantic_lsh(
        self,
        documents: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> tuple[list[dict[str, Any]], list[DuplicateInfo]]:
        """LSH Random Projection for semantic dedup."""
        num_planes = 128
        num_bands = 8
        band_size = num_planes // num_bands

        # 1. Random Projections
        np.random.seed(42)  # reproducible projections
        planes = np.random.randn(embeddings.shape[1], num_planes)
        binary_hash = (embeddings @ planes > 0).astype(np.uint8)

        # 2. Bucket by bands
        buckets: dict[tuple[int, tuple], list[int]] = {}
        for i in range(len(documents)):
            for b in range(num_bands):
                start = b * band_size
                end = start + band_size
                band_key = (b, tuple(binary_hash[i, start:end]))
                if band_key not in buckets:
                    buckets[band_key] = []
                buckets[band_key].append(i)

        # 3. Find candidates
        candidates: set[tuple[int, int]] = set()
        for b_indices in buckets.values():
            if len(b_indices) > 1:
                for idx1 in range(len(b_indices)):
                    for idx2 in range(idx1 + 1, len(b_indices)):
                        i, j = b_indices[idx1], b_indices[idx2]
                        if i > j:
                            i, j = j, i
                        candidates.add((i, j))

        # 4. Filter exact similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        normalized = embeddings / norms

        duplicates: list[DuplicateInfo] = []
        is_dupe = np.zeros(len(documents), dtype=bool)
        sorted_candidates = sorted(list(candidates), key=lambda x: (x[1], x[0]))

        for i, j in sorted_candidates:
            if is_dupe[j] or is_dupe[i]:
                continue
            sim = float(np.dot(normalized[i], normalized[j]))
            if sim >= self.threshold:
                duplicates.append(DuplicateInfo(
                    original_index=i,
                    duplicate_index=j,
                    similarity=sim,
                    strategy="semantic",
                ))
                is_dupe[j] = True

        unique = [doc for i, doc in enumerate(documents) if not is_dupe[i]]
        
        if duplicates:
            logger.info(
                "Semantic LSH dedup (threshold=%.2f): removed %d duplicates from %d documents",
                self.threshold, len(duplicates), len(documents),
            )
        return unique, duplicates

    # ── Hashing helpers ──────────────────────────────────────

    @staticmethod
    def _content_hash(content: str) -> str:
        """SHA-256 hash of normalized content."""
        normalized = content.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _simhash(self, text: str) -> int:
        """Compute SimHash fingerprint for near-duplicate detection."""
        tokens = text.lower().split()
        v = [0] * self.hash_bits

        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for i in range(self.hash_bits):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1

        fingerprint = 0
        for i in range(self.hash_bits):
            if v[i] > 0:
                fingerprint |= 1 << i
        return fingerprint

    def _simhash_similarity(self, h1: int, h2: int) -> float:
        """Compute similarity between two SimHash fingerprints."""
        diff = bin(h1 ^ h2).count("1")
        return 1.0 - (diff / self.hash_bits)
