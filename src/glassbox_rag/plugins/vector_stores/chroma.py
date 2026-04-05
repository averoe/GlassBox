"""ChromaDB vector store plugin — real implementation."""

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from glassbox_rag.plugins.base import VectorStorePlugin
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class ChromaVectorStore(VectorStorePlugin):
    """ChromaDB vector store implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.path = config.get("path", "./data/chroma")
        self.collection_name = config.get("collection_name", "glassbox_docs")
        self._client = None
        self._collection = None

    async def initialize(self) -> bool:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=self.path,
                settings=Settings(anonymized_telemetry=False),
            )

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "ChromaDB connected: path=%s, collection=%s (%d vectors)",
                self.path,
                self.collection_name,
                self._collection.count(),
            )
            return True

        except ImportError:
            logger.error("chromadb not installed. Run: pip install glassbox-rag[vector-stores]")
            return False
        except Exception as e:
            logger.error("Failed to initialize ChromaDB: %s", e)
            return False

    async def shutdown(self) -> None:
        self._client = None
        self._collection = None

    async def health_check(self) -> bool:
        return self._collection is not None

    async def add_vectors(
        self,
        vectors: List[List[float]],
        ids: Optional[List[str]] = None,
        contents: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Add vectors to ChromaDB."""
        if not vectors or self._collection is None:
            return []

        if ids is None:
            ids = [str(uuid4()) for _ in vectors]

        # ChromaDB requires documents or embeddings
        documents = contents or ["" for _ in vectors]
        metadatas = metadata or [{} for _ in vectors]

        # Ensure metadata values are simple types (Chroma requirement)
        clean_metadatas = []
        for m in metadatas:
            clean = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            clean_metadatas.append(clean)

        # Batch add (ChromaDB handles batching internally up to a limit)
        batch_size = 5000
        for i in range(0, len(vectors), batch_size):
            end = min(i + batch_size, len(vectors))
            self._collection.upsert(
                ids=ids[i:end],
                embeddings=vectors[i:end],
                documents=documents[i:end],
                metadatas=clean_metadatas[i:end],
            )

        logger.debug("Added %d vectors to ChromaDB collection '%s'", len(vectors), self.collection_name)
        return ids

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """Search for similar vectors in ChromaDB."""
        if self._collection is None:
            return []

        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            output: List[Tuple[str, float, str, Dict[str, Any]]] = []
            if results and results["ids"]:
                ids = results["ids"][0]
                distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
                documents = results["documents"][0] if results.get("documents") else [""] * len(ids)
                metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)

                for doc_id, distance, content, meta in zip(ids, distances, documents, metadatas):
                    # ChromaDB returns distances, convert to similarity scores
                    # For cosine distance: similarity = 1 - distance
                    score = max(0.0, 1.0 - distance)
                    output.append((doc_id, score, content or "", meta or {}))

            return output

        except Exception as e:
            logger.error("ChromaDB search failed: %s", e)
            return []

    async def get_vector(self, vector_id: str) -> Optional[Dict[str, Any]]:
        if self._collection is None:
            return None
        try:
            result = self._collection.get(
                ids=[vector_id],
                include=["embeddings", "documents", "metadatas"],
            )
            if result and result["ids"]:
                return {
                    "id": result["ids"][0],
                    "vector": result["embeddings"][0] if result.get("embeddings") else None,
                    "content": result["documents"][0] if result.get("documents") else "",
                    "metadata": result["metadatas"][0] if result.get("metadatas") else {},
                }
            return None
        except Exception as e:
            logger.error("Failed to get vector %s: %s", vector_id, e)
            return None

    async def delete_vector(self, vector_id: str) -> bool:
        if self._collection is None:
            return False
        try:
            self._collection.delete(ids=[vector_id])
            return True
        except Exception as e:
            logger.error("Failed to delete vector %s: %s", vector_id, e)
            return False

    async def count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()
