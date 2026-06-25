"""
Generation Metrics — LLM-based evaluation of generated responses.

Each metric calls the configured LLM with a structured prompt and
parses a 0.0–1.0 score from the response. These require an active
LLM generator backend.

Metrics:
    - Faithfulness: fraction of answer claims supported by retrieved context
    - Hallucination rate: fraction of claims with no basis in context
    - Groundedness: overall alignment between answer and evidence
    - Context relevance: how well retrieved docs match the query
    - LLM-as-judge: configurable quality scoring
"""

from __future__ import annotations

import re
from typing import Any

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


def _extract_score(text: str) -> float:
    """Extract a numeric score from LLM response text."""
    # Try to find a decimal number
    matches = re.findall(r"(\d+\.?\d*)", text.strip())
    if matches:
        score = float(matches[0])
        # Normalize if score is on 0-100 scale
        if score > 1.0:
            score = score / 100.0
        return max(0.0, min(1.0, score))
    return 0.0


async def faithfulness(
    answer: str,
    context_docs: list[str],
    generator: Any,
) -> float:
    """
    Fraction of answer claims supported by retrieved documents.

    Returns 0.0–1.0 where 1.0 means every claim is supported.
    """
    if not answer or not context_docs:
        return 0.0

    context = "\n\n---\n\n".join(context_docs)
    prompt = (
        "You are an expert evaluator. Assess the faithfulness of the answer "
        "to the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Answer:\n{answer}\n\n"
        "Score the faithfulness from 0.0 to 1.0, where 1.0 means every claim "
        "in the answer is directly supported by the context. Return ONLY the "
        "numeric score."
    ).format(context=context[:4000], answer=answer[:2000])

    try:
        result = await generator.generate(prompt)
        return _extract_score(result.text)
    except Exception as e:
        logger.warning("Faithfulness evaluation failed: %s", e)
        return 0.0


async def hallucination_rate(
    answer: str,
    context_docs: list[str],
    generator: Any,
) -> float:
    """
    Fraction of claims with no basis in retrieved context.

    Returns 0.0–1.0 where 0.0 means no hallucinations.
    """
    if not answer or not context_docs:
        return 1.0

    context = "\n\n---\n\n".join(context_docs)
    prompt = (
        "You are an expert evaluator. Assess the hallucination rate of the "
        "answer with respect to the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Answer:\n{answer}\n\n"
        "Score the hallucination rate from 0.0 to 1.0, where 0.0 means no "
        "hallucinations and 1.0 means every claim is hallucinated. "
        "Return ONLY the numeric score."
    ).format(context=context[:4000], answer=answer[:2000])

    try:
        result = await generator.generate(prompt)
        return _extract_score(result.text)
    except Exception as e:
        logger.warning("Hallucination rate evaluation failed: %s", e)
        return 1.0


async def groundedness(
    answer: str,
    context_docs: list[str],
    generator: Any,
) -> float:
    """
    Overall alignment between answer and retrieved evidence.

    Returns 0.0–1.0 where 1.0 means perfectly grounded.
    """
    if not answer or not context_docs:
        return 0.0

    context = "\n\n---\n\n".join(context_docs)
    prompt = (
        "You are an expert evaluator. Assess how well the answer is grounded "
        "in the provided evidence.\n\n"
        "Evidence:\n{context}\n\n"
        "Answer:\n{answer}\n\n"
        "Score the groundedness from 0.0 to 1.0, where 1.0 means the answer "
        "is fully supported by the evidence. Return ONLY the numeric score."
    ).format(context=context[:4000], answer=answer[:2000])

    try:
        result = await generator.generate(prompt)
        return _extract_score(result.text)
    except Exception as e:
        logger.warning("Groundedness evaluation failed: %s", e)
        return 0.0


async def context_relevance(
    query: str,
    context_docs: list[str],
    generator: Any,
) -> float:
    """
    How well retrieved documents match the query, independent of the answer.

    Returns 0.0–1.0 where 1.0 means highly relevant context.
    """
    if not query or not context_docs:
        return 0.0

    context = "\n\n---\n\n".join(context_docs)
    prompt = (
        "You are an expert evaluator. Assess how relevant the retrieved "
        "documents are to answering the query.\n\n"
        "Query: {query}\n\n"
        "Retrieved documents:\n{context}\n\n"
        "Score the relevance from 0.0 to 1.0, where 1.0 means the documents "
        "are perfectly relevant. Return ONLY the numeric score."
    ).format(query=query, context=context[:4000])

    try:
        result = await generator.generate(prompt)
        return _extract_score(result.text)
    except Exception as e:
        logger.warning("Context relevance evaluation failed: %s", e)
        return 0.0


async def llm_as_judge(
    query: str,
    answer: str,
    context_docs: list[str],
    generator: Any,
    prompt_template: str | None = None,
) -> float:
    """
    Configurable LLM-based quality scoring.

    Uses a custom prompt template for quality assessment when
    ground truth is unavailable.

    Args:
        prompt_template: Custom prompt with {query}, {answer}, {context} placeholders.
    """
    if not answer:
        return 0.0

    context = "\n\n---\n\n".join(context_docs) if context_docs else ""

    if prompt_template:
        prompt = prompt_template.format(
            query=query, answer=answer, context=context[:4000],
        )
    else:
        prompt = (
            "You are an expert evaluator. Assess the overall quality of the "
            "answer to the query, considering accuracy, completeness, and "
            "relevance.\n\n"
            "Query: {query}\n\n"
            "Context:\n{context}\n\n"
            "Answer: {answer}\n\n"
            "Score the quality from 0.0 to 1.0. Return ONLY the numeric score."
        ).format(query=query, answer=answer[:2000], context=context[:4000])

    try:
        result = await generator.generate(prompt)
        return _extract_score(result.text)
    except Exception as e:
        logger.warning("LLM-as-judge evaluation failed: %s", e)
        return 0.0
