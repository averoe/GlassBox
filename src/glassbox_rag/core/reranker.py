"""
Cross-encoder reranking module.

Reranks retrieval results using cross-encoder models for
higher quality relevance scoring. Supports multiple backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from glassbox_rag.core.retriever import Document
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class RerankerError(Exception):
    """Raised when reranking fails."""


class BaseReranker(ABC):
    """Base class for rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Rerank documents by relevance to query.

        Args:
            query: The query string.
            documents: Documents to rerank.
            top_k: Return only top_k results after reranking.

        Returns:
            Reranked list of documents with updated scores.
        """


class CohereReranker(BaseReranker):
    """Reranker using Cohere Rerank API."""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key")
        self.model = config.get("model", "rerank-english-v3.0")
        self._client = None

    async def initialize(self) -> None:
        try:
            import cohere

            if not self.api_key:
                raise RerankerError("Cohere API key required for reranking")
            self._client = cohere.Client(api_key=self.api_key)
            logger.info("Cohere reranker initialized: model=%s", self.model)
        except ImportError:
            raise RerankerError(
                "cohere package not installed. Run: pip install cohere"
            )

    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        if not self._client or not documents:
            return documents

        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        top_n = top_k or len(documents)
        texts = [doc.content for doc in documents]

        def _rerank_sync() -> Any:
            return self._client.rerank(
                query=query,
                documents=texts,
                model=self.model,
                top_n=top_n,
            )

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(executor, _rerank_sync)

        reranked: List[Document] = []
        for result in response.results:
            doc = documents[result.index]
            doc.score = result.relevance_score
            reranked.append(doc)

        return reranked


class CrossEncoderReranker(BaseReranker):
    """
    Local cross-encoder reranker using sentence-transformers.

    Requires: pip install sentence-transformers
    """

    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._model = None

    async def initialize(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder reranker initialized: model=%s", self.model_name)
        except ImportError:
            raise RerankerError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        if not self._model or not documents:
            return documents

        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        pairs = [(query, doc.content) for doc in documents]

        def _predict_sync() -> Any:
            return self._model.predict(pairs)

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            scores = await loop.run_in_executor(executor, _predict_sync)

        for doc, score in zip(documents, scores):
            doc.score = float(score)

        reranked = sorted(documents, key=lambda d: d.score or 0, reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked


class HuggingFaceReranker(BaseReranker):
    """Reranker using Hugging Face Inference API."""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key")
        self.model = config.get("model", "BAAI/bge-reranker-base")

    async def initialize(self) -> None:
        if not self.api_key:
            raise RerankerError("Hugging Face API key required")
        logger.info("HuggingFace reranker initialized: model=%s", self.model)

    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        if not documents:
            return documents

        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": {
                "source_sentence": query,
                "sentences": [doc.content for doc in documents],
            }
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers=headers,
                json=payload,
            )
            if resp.status_code != 200:
                raise RerankerError(f"HF reranker API error: {resp.status_code}")
            scores = resp.json()

        if isinstance(scores, list):
            for doc, score in zip(documents, scores):
                doc.score = float(score) if isinstance(score, (int, float)) else 0.0

        reranked = sorted(documents, key=lambda d: d.score or 0, reverse=True)
        if top_k:
            reranked = reranked[:top_k]
        return reranked


# ═══════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════

_RERANKER_MAP = {
    "cohere": CohereReranker,
    "cross-encoder": CrossEncoderReranker,
    "huggingface": HuggingFaceReranker,
}


async def create_reranker(
    reranker_type: str,
    config: Dict[str, Any],
) -> BaseReranker:
    """Create and initialize a reranker."""
    if reranker_type not in _RERANKER_MAP:
        available = list(_RERANKER_MAP.keys())
        raise ValueError(
            f"Unknown reranker type '{reranker_type}'. Available: {available}"
        )

    reranker = _RERANKER_MAP[reranker_type](config)
    await reranker.initialize()
    return reranker
