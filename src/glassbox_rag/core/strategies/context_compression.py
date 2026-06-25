"""
Context Compression — post-retrieval / pre-generation hook that passes
retrieved chunks through a compression model to remove irrelevant sentences
while preserving key information.

Runs before prompt assembly. Keeps generation within token limits on
large result sets. Compression model is configurable.
"""

from __future__ import annotations

from typing import Any

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class ContextCompressor:
    """
    Pre-generation hook that compresses retrieved context using LLM.

    Removes irrelevant sentences from retrieved documents while
    preserving key information. Reduces token usage and improves
    generation focus.
    """

    def __init__(self, config: Any, generator: Any) -> None:
        """
        Args:
            config: ContextCompressionConfig from GlassBoxConfig.
            generator: LLMGenerator instance for compression.
        """
        self.enabled = config.enabled
        self.prompt_template = config.prompt_template
        self._generator = generator

    async def __call__(self, context: dict[str, Any]) -> dict[str, Any]:
        """Hook function: compress context documents before generation."""
        if not self.enabled or self._generator is None:
            return context

        context_docs = context.get("context_documents", [])
        query = context.get("query", "")

        if not context_docs or not query:
            return context

        try:
            # Join all context into one block for compression
            full_context = "\n\n---\n\n".join(context_docs)
            prompt = self.prompt_template.format(
                query=query, context=full_context,
            )
            result = await self._generator.generate(prompt)
            compressed = result.text.strip()

            if compressed:
                # Replace context with compressed version
                context["context_documents"] = [compressed]
                context["_uncompressed_count"] = len(context_docs)
                context["_uncompressed_length"] = len(full_context)
                context["_compressed_length"] = len(compressed)
                logger.debug(
                    "Context compressed: %d docs (%d chars) → %d chars",
                    len(context_docs), len(full_context), len(compressed),
                )
        except Exception as e:
            logger.warning("Context compression failed, using original: %s", e)

        return context
