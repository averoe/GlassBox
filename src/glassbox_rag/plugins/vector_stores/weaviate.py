"""Weaviate vector store plugin — real implementation."""

from typing import Any, Dict, List, Optional, Tuple

from glassbox_rag.plugins.base import VectorStorePlugin
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class WeaviateVectorStore(VectorStorePlugin):
    """Weaviate vector store implementation using weaviate-client."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.url = config.get("url", "http://localhost:8080")
        self.api_key = config.get("api_key")
        self.class_name = config.get("class_name", "Document")
        self.vectorizer = config.get("vectorizer", "none")  # We'll handle vectorization ourselves
        self._client = None

    async def initialize(self) -> bool:
        """Initialize Weaviate client and ensure class exists."""
        try:
            import weaviate
            from weaviate.exceptions import WeaviateConnectionError

            # Initialize client
            auth_config = None
            if self.api_key:
                auth_config = weaviate.auth.AuthApiKey(api_key=self.api_key)

            self._client = weaviate.Client(
                url=self.url,
                auth_client_secret=auth_config,
                timeout_config=(5, 60)  # (connect, read) timeouts
            )

            # Test connection
            if not self._client.is_ready():
                raise WeaviateConnectionError("Weaviate client is not ready")

            # Create class if it doesn't exist
            if not self._client.schema.exists(self.class_name):
                class_obj = {
                    "class": self.class_name,
                    "vectorizer": self.vectorizer,
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                        },
                        {
                            "name": "docId",
                            "dataType": ["string"],
                        },
                        {
                            "name": "metadata",
                            "dataType": ["object"],
                        },
                    ]
                }
                self._client.schema.create_class(class_obj)
                logger.info("Created Weaviate class: %s", self.class_name)

            # Get class info
            class_info = self._client.schema.get(self.class_name)
            object_count = self._client.query.aggregate(self.class_name).with_meta_count().do()

            logger.info(
                "Weaviate connected: url=%s, class=%s, objects=%s",
                self.url,
                self.class_name,
                object_count.get("data", {}).get("Aggregate", {}).get(self.class_name, [{}])[0].get("meta", {}).get("count", 0),
            )
            return True

        except ImportError:
            logger.error("weaviate-client not installed. Run: pip install weaviate-client")
            return False
        except Exception as e:
            logger.error("Failed to initialize Weaviate: %s", e)
            return False

    async def add_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        contents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Add vectors to Weaviate."""
        if not self._client:
            logger.error("Weaviate client not initialized")
            return False

        try:
            # Prepare objects for batch import
            objects = []
            for i, (vector, doc_id) in enumerate(zip(vectors, ids)):
                obj = {
                    "class": self.class_name,
                    "properties": {
                        "docId": doc_id,
                        "content": contents[i],
                        "metadata": metadata[i] if metadata and i < len(metadata) else {},
                    },
                    "vector": vector,
                }
                objects.append(obj)

            # Batch import
            result = self._client.batch.create_objects(objects)

            if result.get("errors"):
                logger.error("Batch import errors: %s", result["errors"])
                return False

            logger.info("Added %d objects to Weaviate", len(objects))
            return True

        except Exception as e:
            logger.error("Failed to add vectors to Weaviate: %s", e)
            return False

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """Search for similar vectors in Weaviate."""
        if not self._client:
            logger.error("Weaviate client not initialized")
            return []

        try:
            # Perform vector search
            result = (
                self._client.query
                .get(self.class_name, ["docId", "content", "metadata"])
                .with_near_vector({"vector": query_vector})
                .with_limit(top_k)
                .with_additional(["certainty", "distance"])
                .do()
            )

            results = []
            objects = result.get("data", {}).get("Get", {}).get(self.class_name, [])

            for obj in objects:
                additional = obj.get("_additional", {})
                certainty = additional.get("certainty", 0.0)
                distance = additional.get("distance", 1.0)

                # Convert distance to similarity score (higher is better)
                score = 1.0 - distance if distance is not None else certainty

                results.append((
                    obj.get("docId", ""),
                    score,
                    obj.get("content", ""),
                    obj.get("metadata", {}),
                ))

            logger.debug("Weaviate search returned %d results", len(results))
            return results

        except Exception as e:
            logger.error("Failed to search Weaviate: %s", e)
            return []

    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Weaviate."""
        if not self._client:
            logger.error("Weaviate client not initialized")
            return False

        try:
            # Delete by docId
            for doc_id in ids:
                self._client.data_object.delete(
                    class_name=self.class_name,
                    where={"path": ["docId"], "operator": "Equal", "valueString": doc_id}
                )

            logger.info("Deleted %d objects from Weaviate", len(ids))
            return True

        except Exception as e:
            logger.error("Failed to delete vectors from Weaviate: %s", e)
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self._client:
            return {"error": "Client not initialized"}

        try:
            # Get object count
            result = self._client.query.aggregate(self.class_name).with_meta_count().do()
            count = result.get("data", {}).get("Aggregate", {}).get(self.class_name, [{}])[0].get("meta", {}).get("count", 0)

            return {
                "class_name": self.class_name,
                "object_count": count,
                "url": self.url,
            }
        except Exception as e:
            return {"error": str(e)}