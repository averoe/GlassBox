"""
Retrieval Module — handles querying and result ranking.

Implements adaptive retrieval strategies that intelligently select
the optimal strategy for each query. All retrievers share a
consistent interface.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from glassbox_rag.config import GlassBoxConfig
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class RetrieverError(Exception):
    """Raised when a retrieval operation fails."""


@dataclass
class Document:
    """Represents a document in the vector store."""

    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    documents: List[Document]
    query: str
    strategy: str
    execution_time_ms: float
    total_results: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "execution_time_ms": self.execution_time_ms,
            "total_results": self.total_results,
            "documents": [d.to_dict() for d in self.documents],
        }


class BaseRetriever(ABC):
    """Base class for retrieval strategies."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    async def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        query_text: str = "",
        **kwargs: Any,
    ) -> List[Document]:
        """
        Retrieve documents based on query.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of top results to return.
            query_text: Original query text (for keyword strategies).
            **kwargs: Additional retriever-specific arguments.

        Returns:
            List of retrieved Document objects, sorted by relevance.
        """


class SemanticRetriever(BaseRetriever):
    """Semantic (vector similarity) based retrieval."""

    def __init__(self, config: dict, vector_store: Any = None):
        super().__init__(config)
        self.vector_store = vector_store

    async def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        query_text: str = "",
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieve using vector similarity search."""
        if self.vector_store is None:
            logger.warning("No vector store configured for semantic retrieval")
            return []

        try:
            results = await self.vector_store.search(
                query_vector=query_embedding.tolist(),
                top_k=top_k,
            )

            documents: List[Document] = []
            for doc_id, score, content, metadata in results:
                documents.append(
                    Document(
                        id=doc_id,
                        content=content,
                        score=score,
                        metadata=metadata or {},
                    )
                )

            return documents

        except Exception as e:
            raise RetrieverError(f"Semantic retrieval failed: {e}") from e


class KeywordRetriever(BaseRetriever):
    """
    Keyword / BM25 based retrieval.

    Falls back to simple TF-IDF-like scoring when no external
    search engine is configured.
    """

    def __init__(self, config: dict, database: Any = None):
        super().__init__(config)
        self.database = database

    async def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        query_text: str = "",
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieve using keyword matching."""
        if not query_text:
            return []

        if self.database is None:
            logger.warning("No database configured for keyword retrieval")
            return []

        try:
            # Use database full-text search
            query_terms = query_text.lower().split()
            results = await self.database.search_text(query_terms, top_k=top_k)

            documents: List[Document] = []
            for record in results:
                documents.append(
                    Document(
                        id=record.get("id", ""),
                        content=record.get("content", ""),
                        score=record.get("score", 0.0),
                        metadata=record.get("metadata", {}),
                    )
                )

            return documents

        except Exception as e:
            raise RetrieverError(f"Keyword retrieval failed: {e}") from e


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval combining semantic and keyword strategies."""

    def __init__(
        self,
        config: dict,
        vector_store: Any = None,
        database: Any = None,
    ):
        super().__init__(config)
        self.semantic_retriever = SemanticRetriever(config, vector_store)
        self.keyword_retriever = KeywordRetriever(config, database)
        self.semantic_weight = config.get("weight_semantic", 0.6)
        self.keyword_weight = config.get("weight_keyword", 0.4)

    async def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        query_text: str = "",
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieve using reciprocal rank fusion of semantic + keyword."""
        semantic_docs = await self.semantic_retriever.retrieve(
            query_embedding, top_k=top_k * 2, query_text=query_text,
        )
        keyword_docs = await self.keyword_retriever.retrieve(
            query_embedding, top_k=top_k * 2, query_text=query_text,
        )

        # Reciprocal Rank Fusion (RRF)
        k = 60  # RRF constant
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(semantic_docs):
            rrf = self.semantic_weight / (k + rank + 1)
            scores[doc.id] = scores.get(doc.id, 0) + rrf
            doc_map[doc.id] = doc

        for rank, doc in enumerate(keyword_docs):
            rrf = self.keyword_weight / (k + rank + 1)
            scores[doc.id] = scores.get(doc.id, 0) + rrf
            if doc.id not in doc_map:
                doc_map[doc.id] = doc

        # Sort by fused score
        ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

        results: List[Document] = []
        for doc_id in ranked_ids:
            doc = doc_map[doc_id]
            doc.score = scores[doc_id]
            results.append(doc)

        return results


class AdaptiveRetriever:
    """
    Adaptive retrieval that selects the best strategy per query.

    Uses heuristics to choose between semantic, keyword, and hybrid
    retrieval based on query characteristics.
    """

    def __init__(
        self,
        config: GlassBoxConfig,
        vector_store: Any = None,
        database: Any = None,
    ):
        self.config = config
        self.strategies: Dict[str, BaseRetriever] = {}
        self._init_strategies(vector_store, database)

    def _init_strategies(self, vector_store: Any, database: Any) -> None:
        """Initialize available retrieval strategies."""
        adaptive_config = self.config.retrieval.adaptive

        if not adaptive_config.get("enabled", False):
            self.strategies["semantic"] = SemanticRetriever({}, vector_store)
            return

        for strategy_config in adaptive_config.get("strategies", []):
            name = strategy_config.get("name")
            if name == "semantic":
                self.strategies["semantic"] = SemanticRetriever(
                    strategy_config, vector_store
                )
            elif name == "keyword":
                self.strategies["keyword"] = KeywordRetriever(
                    strategy_config, database
                )
            elif name == "hybrid":
                self.strategies["hybrid"] = HybridRetriever(
                    strategy_config, vector_store, database
                )

        # Ensure at least semantic is available
        if not self.strategies:
            self.strategies["semantic"] = SemanticRetriever({}, vector_store)

    async def select_strategy(self, query: str) -> str:
        """
        Intelligently select retrieval strategy based on query.

        Heuristics:
        - Short keyword-like queries → keyword retrieval
        - Natural language questions → semantic retrieval
        - Mixed / unclear → hybrid if available
        """
        if "hybrid" in self.strategies:
            # Default to hybrid when available — it's the safest bet
            return "hybrid"

        # Simple heuristics
        word_count = len(query.split())
        has_question_mark = "?" in query
        question_words = {"what", "how", "why", "when", "where", "who", "which"}
        first_word = query.strip().split()[0].lower() if query.strip() else ""

        is_natural_language = has_question_mark or first_word in question_words or word_count > 5

        if is_natural_language and "semantic" in self.strategies:
            return "semantic"
        elif word_count <= 3 and "keyword" in self.strategies:
            return "keyword"
        elif "semantic" in self.strategies:
            return "semantic"

        # Fallback to first available
        return next(iter(self.strategies))

    async def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Adaptively retrieve documents.

        Args:
            query: Query text.
            query_embedding: Query embedding.
            top_k: Number of results (defaults to config).

        Returns:
            RetrievalResult with documents and metadata.
        """
        if top_k is None:
            top_k = self.config.retrieval.top_k

        strategy_name = await self.select_strategy(query)

        if strategy_name not in self.strategies:
            available = list(self.strategies.keys())
            raise RetrieverError(
                f"Strategy '{strategy_name}' not available. Available: {available}"
            )

        start_time = time.perf_counter()

        retriever = self.strategies[strategy_name]
        documents = await retriever.retrieve(
            query_embedding,
            top_k=top_k,
            query_text=query,
        )

        execution_time_ms = round((time.perf_counter() - start_time) * 1000, 3)

        # Filter by minimum score
        min_score = self.config.retrieval.min_score
        documents = [doc for doc in documents if (doc.score or 0) >= min_score]

        logger.debug(
            "Retrieved %d documents using '%s' strategy in %.2fms",
            len(documents),
            strategy_name,
            execution_time_ms,
        )

        return RetrievalResult(
            documents=documents,
            query=query,
            strategy=strategy_name,
            execution_time_ms=execution_time_ms,
            total_results=len(documents),
        )
