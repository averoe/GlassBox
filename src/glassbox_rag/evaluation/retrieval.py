"""
Retrieval Metrics — pure functions for evaluating retrieval quality.

All functions operate on lists of document IDs and return floats in [0.0, 1.0].
No LLM calls required — these are deterministic metrics.

Metrics:
    - Recall@K: fraction of relevant documents appearing in top K results
    - Precision@K: fraction of top K results that are relevant
    - MRR: mean reciprocal rank of the first relevant result
    - NDCG: normalized discounted cumulative gain
    - Hit Rate: binary — did any correct document appear in top K
"""

from __future__ import annotations

import math
from typing import List


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Fraction of relevant documents appearing in top K results.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: List of ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        Recall score in [0.0, 1.0].
    """
    if not relevant_ids:
        return 1.0  # No relevant docs to miss
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Fraction of top K results that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: List of ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        Precision score in [0.0, 1.0].
    """
    if k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / k


def mrr(
    retrieved_ids_list: list[list[str]],
    relevant_ids_list: list[list[str]],
) -> float:
    """
    Mean Reciprocal Rank across multiple queries.

    Args:
        retrieved_ids_list: List of retrieved ID lists (one per query).
        relevant_ids_list: List of relevant ID lists (one per query).

    Returns:
        MRR score in [0.0, 1.0].
    """
    if not retrieved_ids_list:
        return 0.0

    reciprocal_ranks: list[float] = []
    for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
        relevant_set = set(relevant)
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def ndcg(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain.

    Weights correct results higher when they appear earlier.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: List of ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        NDCG score in [0.0, 1.0].
    """
    if not relevant_ids:
        return 1.0

    relevant_set = set(relevant_ids)

    # DCG for actual results
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG (all relevant docs at top)
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def hit_rate(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Binary hit rate: did any correct document appear in top K?

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: List of ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        1.0 if any relevant doc is in top K, 0.0 otherwise.
    """
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return 1.0 if top_k & relevant else 0.0
