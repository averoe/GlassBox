"""Qdrant vector store plugin — real implementation."""

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from glassbox_rag.plugins.base import VectorStorePlugin
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class QdrantVectorStore(VectorStorePlugin):
    """Qdrant vector store implementation using qdrant-client."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6333)
        self.collection_name = config.get("collection_name", "glassbox_docs")
        self.vector_size = config.get("vector_size", 384)
        self.distance_metric = config.get("distance_metric", "cosine")
        self._client = None

    async def initialize(self) -> bool:
        """Initialize Qdrant client and ensure collection exists."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._client = QdrantClient(host=self.host, port=self.port)

            # Map distance metric
            distance_map = {
                "cosine": Distance.COSINE,
                "euclid": Distance.EUCLID,
                "dot": Distance.DOT,
            }
            distance = distance_map.get(self.distance_metric, Distance.COSINE)

            # Create collection if it doesn't exist
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=distance,
                    ),
                )
                logger.info(
                    "Created Qdrant collection '%s' (size=%d, distance=%s)",
                    self.collection_name,
                    self.vector_size,
                    self.distance_metric,
                )

            logger.info("Qdrant connected: %s:%d/%s", self.host, self.port, self.collection_name)
            return True

        except ImportError:
            logger.error("qdrant-client not installed. Run: pip install glassbox-rag[vector-stores]")
            return False
        except Exception as e:
            logger.error("Failed to initialize Qdrant: %s", e)
            return False

    async def shutdown(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    async def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    async def add_vectors(
        self,
        vectors: List[List[float]],
        ids: Optional[List[str]] = None,
        contents: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Add vectors to Qdrant."""
        if not vectors or self._client is None:
            return []

        from qdrant_client.models import PointStruct

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in vectors]

        points = []
        for i, (vec_id, vector) in enumerate(zip(ids, vectors)):
            payload: Dict[str, Any] = {}
            if contents and i < len(contents):
                payload["content"] = contents[i]
            if metadata and i < len(metadata):
                payload.update(metadata[i])

            points.append(
                PointStruct(
                    id=vec_id,
                    vector=vector,
                    payload=payload,
                )
            )

        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.debug("Added %d vectors to Qdrant collection '%s'", len(vectors), self.collection_name)
        return ids

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """Search for similar vectors in Qdrant."""
        if self._client is None:
            return []

        try:
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
            )

            output: List[Tuple[str, float, str, Dict[str, Any]]] = []
            for hit in results:
                payload = hit.payload or {}
                content = payload.pop("content", "")
                output.append((str(hit.id), hit.score, content, payload))

            return output

        except Exception as e:
            logger.error("Qdrant search failed: %s", e)
            return []

    async def get_vector(self, vector_id: str) -> Optional[Dict[str, Any]]:
        if self._client is None:
            return None
        try:
            results = self._client.retrieve(
                collection_name=self.collection_name,
                ids=[vector_id],
            )
            if results:
                point = results[0]
                return {
                    "id": str(point.id),
                    "vector": point.vector,
                    "payload": point.payload,
                }
            return None
        except Exception as e:
            logger.error("Failed to get vector %s: %s", vector_id, e)
            return None

    async def delete_vector(self, vector_id: str) -> bool:
        if self._client is None:
            return False
        try:
            from qdrant_client.models import PointIdsList
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[vector_id]),
            )
            return True
        except Exception as e:
            logger.error("Failed to delete vector %s: %s", vector_id, e)
            return False

    async def count(self) -> int:
        if self._client is None:
            return 0
        try:
            info = self._client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0
