"""
Query Rewrite — pre-retrieval hook that rewrites the user's raw query
into a cleaner, more retrieval-friendly form using a lightweight LLM call.

Registered as a PRE_RETRIEVE hook. Configurable on/off, model, prompt template.
"""

from __future__ import annotations

from typing import Any

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class QueryRewriteHook:
    """
    Pre-retrieval hook that rewrites queries via LLM.

    Uses the generator backend already configured on the engine
    (or a separate model if specified in config).
    """

    def __init__(self, config: Any, generator: Any) -> None:
        """
        Args:
            config: QueryRewriteConfig from the engine's GlassBoxConfig.
            generator: LLMGenerator instance for rewriting.
        """
        self.enabled = config.enabled
        self.prompt_template = config.prompt_template
        self._generator = generator

    async def __call__(self, context: dict[str, Any]) -> dict[str, Any]:
        """Hook function: rewrite the query in the context dict."""
        if not self.enabled or self._generator is None:
            return context

        original_query = context.get("query", "")
        if not original_query:
            return context

        try:
            prompt = self.prompt_template.format(query=original_query)
            result = await self._generator.generate(prompt)
            rewritten = result.text.strip()
            if rewritten:
                context["query"] = rewritten
                context["_original_query"] = original_query
                logger.debug(
                    "Query rewritten: '%s' → '%s'",
                    original_query[:50], rewritten[:50],
                )
        except Exception as e:
            logger.warning("Query rewrite failed, using original: %s", e)

        return context
