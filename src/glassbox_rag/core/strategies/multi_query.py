"""
Multi-Query Expansion — generates N semantically distinct versions of the query,
runs retrieval for each independently, aggregates all results using RRF fusion,
and deduplicates.

Implemented as a wrapper around the engine's retrieve path.
"""

from __future__ import annotations

from typing import Any, List

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class MultiQueryExpander:
    """
    Generates N query variants via LLM, retrieves for each,
    and aggregates via RRF fusion.
    """

    EXPANSION_PROMPT = (
        "Generate {n} semantically distinct search queries for the following "
        "question. Return each query on a separate line, no numbering.\n\n"
        "Original question: {query}"
    )

    def __init__(self, config: Any, generator: Any) -> None:
        """
        Args:
            config: MultiQueryConfig from GlassBoxConfig.
            generator: LLMGenerator instance.
        """
        self.enabled = config.enabled
        self.num_queries = config.num_queries
        self._generator = generator

    async def expand_query(self, query: str) -> list[str]:
        """
        Generate N query variants from the original query.

        Returns the original query plus N generated variants.
        """
        if not self.enabled or self._generator is None:
            return [query]

        try:
            prompt = self.EXPANSION_PROMPT.format(n=self.num_queries, query=query)
            result = await self._generator.generate(prompt)
            lines = [
                line.strip()
                for line in result.text.strip().split("\n")
                if line.strip()
            ]
            # Always include original query
            variants = [query] + lines[:self.num_queries]
            logger.debug(
                "Multi-query expanded '%s' into %d variants",
                query[:50], len(variants),
            )
            return variants
        except Exception as e:
            logger.warning("Multi-query expansion failed: %s", e)
            return [query]

    @staticmethod
    def rrf_fuse(
        result_lists: list[list[Any]],
        k: int = 60,
        top_k: int = 10,
    ) -> list[Any]:
        """
        Reciprocal Rank Fusion across multiple result lists.

        Each result list is a list of Document objects with .id attribute.
        Returns deduplicated, fused results sorted by RRF score.
        """
        scores: dict[str, float] = {}
        doc_map: dict[str, Any] = {}

        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                rrf_score = 1.0 / (k + rank + 1)
                scores[doc.id] = scores.get(doc.id, 0.0) + rrf_score
                if doc.id not in doc_map:
                    doc_map[doc.id] = doc

        # Sort by fused score descending
        ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

        results = []
        for doc_id in ranked_ids:
            doc = doc_map[doc_id]
            doc.score = scores[doc_id]
            results.append(doc)

        return results
