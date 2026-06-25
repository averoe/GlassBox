"""
Parent-Child Retrieval — post-retrieval hook that resolves parent documents
from chunk metadata.

At retrieval time, chunks are retrieved normally, then this hook resolves
and returns their full parent documents for generation context.
Requires a `parent_id` field in chunk metadata.
"""

from __future__ import annotations

from typing import Any

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class ParentChildResolver:
    """
    Post-retrieval hook that resolves parent documents from chunk metadata.

    Looks up `parent_id` in each chunk's metadata and fetches the full
    parent document from the database. Replaces chunk content with
    parent content for richer generation context.
    """

    def __init__(self, config: Any, database: Any = None) -> None:
        """
        Args:
            config: ParentChildConfig from GlassBoxConfig.
            database: DatabasePlugin instance for parent lookups.
        """
        self.enabled = config.enabled
        self._database = database

    async def __call__(self, context: dict[str, Any]) -> dict[str, Any]:
        """Hook function: resolve parent documents for retrieved chunks."""
        if not self.enabled or self._database is None:
            return context

        documents = context.get("documents", [])
        if not documents:
            return context

        resolved_docs = []
        seen_parents: set[str] = set()

        for doc in documents:
            parent_id = doc.metadata.get("parent_id", "") if hasattr(doc, "metadata") else ""

            if parent_id and parent_id not in seen_parents:
                # Resolve parent document
                try:
                    parent_records = await self._database.query(
                        "documents", {"id": parent_id}
                    )
                    if parent_records:
                        parent = parent_records[0]
                        # Create a new document with parent content
                        from glassbox_rag.core.retriever import Document

                        parent_doc = Document(
                            id=parent_id,
                            content=parent.get("content", doc.content),
                            score=doc.score,
                            metadata={
                                **doc.metadata,
                                "resolved_from_child": doc.id,
                                "is_parent_document": True,
                            },
                        )
                        resolved_docs.append(parent_doc)
                        seen_parents.add(parent_id)
                        logger.debug(
                            "Resolved parent %s from child %s",
                            parent_id[:12], doc.id[:12],
                        )
                    else:
                        # Parent not found, keep original chunk
                        resolved_docs.append(doc)
                except Exception as e:
                    logger.warning("Parent resolution failed for %s: %s", parent_id, e)
                    resolved_docs.append(doc)
            elif parent_id and parent_id in seen_parents:
                # Skip duplicate parent
                continue
            else:
                # No parent_id, keep chunk as-is
                resolved_docs.append(doc)

        context["documents"] = resolved_docs
        return context
