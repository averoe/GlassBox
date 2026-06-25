"""Unit tests for document deduplication (core/dedup.py).

Covers exact, fuzzy, and semantic strategies for both batch dedup
and streaming is_duplicate(). Validates original_index correctness
and thread-safe random state.
"""

import numpy as np
import pytest
from glassbox_rag.core.dedup import DocumentDeduplicator, DuplicateInfo


# ── Exact Dedup ────────────────────────────────────────────────────

class TestExactDedup:
    def test_removes_identical_content(self):
        d = DocumentDeduplicator(strategy="exact")
        docs = [{"content": "hello"}, {"content": "hello"}, {"content": "world"}]
        unique, dupes = d.deduplicate(docs)
        assert len(unique) == 2
        assert len(dupes) == 1
        assert dupes[0].similarity == 1.0

    def test_case_normalized(self):
        """SHA-256 of .strip().lower() — 'Hello' and 'hello' are duplicates."""
        d = DocumentDeduplicator(strategy="exact")
        docs = [{"content": "Hello"}, {"content": "hello"}]
        unique, dupes = d.deduplicate(docs)
        assert len(dupes) == 1

    def test_whitespace_normalized(self):
        d = DocumentDeduplicator(strategy="exact")
        docs = [{"content": "  hello  "}, {"content": "hello"}]
        unique, dupes = d.deduplicate(docs)
        assert len(dupes) == 1

    def test_no_duplicates(self):
        d = DocumentDeduplicator(strategy="exact")
        docs = [{"content": "alpha"}, {"content": "beta"}, {"content": "gamma"}]
        unique, dupes = d.deduplicate(docs)
        assert len(unique) == 3
        assert len(dupes) == 0

    def test_empty_input(self):
        d = DocumentDeduplicator(strategy="exact")
        unique, dupes = d.deduplicate([])
        assert unique == []
        assert dupes == []

    def test_original_index_correct(self):
        d = DocumentDeduplicator(strategy="exact")
        docs = [
            {"content": "first"},
            {"content": "second"},
            {"content": "first"},  # dup of index 0
        ]
        _, dupes = d.deduplicate(docs)
        assert dupes[0].original_index == 0
        assert dupes[0].duplicate_index == 2


# ── Fuzzy Dedup ────────────────────────────────────────────────────

class TestFuzzyDedup:
    def test_removes_near_duplicates(self):
        d = DocumentDeduplicator(strategy="fuzzy", threshold=0.7)
        docs = [
            {"content": "The quick brown fox jumps over the lazy dog"},
            {"content": "The quick brown fox jumped over the lazy dog"},
        ]
        unique, dupes = d.deduplicate(docs)
        assert len(dupes) == 1

    def test_keeps_distinct_documents(self):
        d = DocumentDeduplicator(strategy="fuzzy", threshold=0.95)
        docs = [
            {"content": "The quick brown fox jumps over the lazy dog"},
            {"content": "A completely different document about quantum physics and mathematics"},
        ]
        unique, dupes = d.deduplicate(docs)
        assert len(unique) == 2
        assert len(dupes) == 0

    def test_original_index_maps_to_source_document(self):
        """original_index must reference documents[], not unique_hashes[]."""
        d = DocumentDeduplicator(strategy="fuzzy", threshold=0.7)
        docs = [
            {"content": "The quick brown fox jumps over the lazy dog"},
            {"content": "A completely different document about cats and dogs in the park"},
            {"content": "The quick brown fox jumped over the lazy dog"},  # near-dup of index 0
        ]
        unique, dupes = d.deduplicate(docs)
        if dupes:
            # original_index must be 0 (the first doc), NOT from unique list
            assert dupes[0].original_index == 0
            assert dupes[0].duplicate_index == 2

    def test_multiple_duplicates(self):
        d = DocumentDeduplicator(strategy="fuzzy", threshold=0.7)
        base = "The quick brown fox jumps over the lazy dog in the park"
        docs = [
            {"content": base},
            {"content": base + " today"},
            {"content": base + " now"},
        ]
        unique, dupes = d.deduplicate(docs)
        assert len(unique) <= 2  # at least some should be flagged as dupes


# ── Semantic Dedup ─────────────────────────────────────────────────

class TestSemanticDedup:
    def test_removes_semantically_similar(self):
        d = DocumentDeduplicator(strategy="semantic", threshold=0.95)
        docs = [{"content": "doc1"}, {"content": "doc2"}]
        # Create nearly identical embeddings
        embs = np.array([[1.0, 0.0, 0.0], [0.99, 0.01, 0.0]])
        unique, dupes = d.deduplicate(docs, embeddings=embs)
        assert len(dupes) == 1

    def test_keeps_different_embeddings(self):
        d = DocumentDeduplicator(strategy="semantic", threshold=0.95)
        docs = [{"content": "doc1"}, {"content": "doc2"}]
        embs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        unique, dupes = d.deduplicate(docs, embeddings=embs)
        assert len(unique) == 2
        assert len(dupes) == 0

    def test_requires_embeddings(self):
        d = DocumentDeduplicator(strategy="semantic")
        docs = [{"content": "hello"}]
        with pytest.raises(ValueError, match="requires embeddings"):
            d.deduplicate(docs)


# ── Semantic LSH ───────────────────────────────────────────────────

class TestSemanticLSH:
    def test_thread_safe_random_state(self):
        """np.random.default_rng must not affect global state."""
        rng_before = np.random.get_state()[1][0]
        d = DocumentDeduplicator(strategy="semantic", threshold=0.9)
        docs = [{"content": f"doc {i}"} for i in range(10001)]
        embs = np.random.default_rng(123).standard_normal((10001, 64))
        d.deduplicate(docs, embeddings=embs)
        rng_after = np.random.get_state()[1][0]
        assert rng_before == rng_after  # global state untouched


# ── is_duplicate() Streaming ───────────────────────────────────────

class TestIsDuplicate:
    def test_exact_strategy(self):
        d = DocumentDeduplicator(strategy="exact")
        assert not d.is_duplicate("hello world")
        assert d.is_duplicate("hello world")  # second time → duplicate
        assert not d.is_duplicate("different text")

    def test_fuzzy_strategy_respected(self):
        d = DocumentDeduplicator(strategy="fuzzy", threshold=0.7)
        assert not d.is_duplicate("The quick brown fox jumps over the lazy dog")
        assert d.is_duplicate("The quick brown fox jumped over the lazy dog")

    def test_exact_strategy_misses_near_duplicates(self):
        d = DocumentDeduplicator(strategy="exact")
        assert not d.is_duplicate("The quick brown fox")
        assert not d.is_duplicate("The quick brown fox jumped")  # NOT flagged

    def test_semantic_requires_embedding(self):
        d = DocumentDeduplicator(strategy="semantic", threshold=0.9)
        with pytest.raises(ValueError, match="requires an embedding"):
            d.is_duplicate("hello")

    def test_semantic_is_duplicate(self):
        d = DocumentDeduplicator(strategy="semantic", threshold=0.95)
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.99, 0.01, 0.0])
        assert not d.is_duplicate("doc1", embedding=emb1)
        assert d.is_duplicate("doc2", embedding=emb2)

    def test_reset_clears_state(self):
        d = DocumentDeduplicator(strategy="exact")
        assert not d.is_duplicate("hello")
        assert d.is_duplicate("hello")
        d.reset()
        assert not d.is_duplicate("hello")  # after reset, not a duplicate

    def test_fuzzy_reset_clears_state(self):
        d = DocumentDeduplicator(strategy="fuzzy", threshold=0.7)
        assert not d.is_duplicate("The quick brown fox jumps over the lazy dog")
        d.reset()
        assert not d.is_duplicate("The quick brown fox jumps over the lazy dog")


# ── deduplicate_texts() ───────────────────────────────────────────

class TestDeduplicateTexts:
    def test_basic_dedup(self):
        d = DocumentDeduplicator(strategy="exact")
        texts = ["hello", "world", "hello"]
        unique, indices, dupes = d.deduplicate_texts(texts)
        assert len(unique) == 2
        assert 0 in indices
        assert 1 in indices
        assert 2 not in indices


# ── Invalid Strategy ──────────────────────────────────────────────

class TestInvalidStrategy:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown dedup strategy"):
            DocumentDeduplicator(strategy="magic")
