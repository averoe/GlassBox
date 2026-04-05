"""Pinecone vector store plugin — real implementation."""

from typing import Any, Dict, List, Optional, Tuple

from glassbox_rag.plugins.base import VectorStorePlugin
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class PineconeVectorStore(VectorStorePlugin):
    """Pinecone vector store implementation using pinecone-client."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.environment = config.get("environment", "us-east-1-aws")
        self.index_name = config.get("index_name", "glassbox-docs")
        self.dimension = config.get("dimension", 384)
        self.metric = config.get("metric", "cosine")
        self._index = None

    async def initialize(self) -> bool:
        """Initialize Pinecone client and ensure index exists."""
        try:
            import pinecone

            # Initialize Pinecone
            pinecone.init(api_key=self.api_key, environment=self.environment)

            # Check if index exists, create if not
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    pods=1,
                    replicas=1,
                )
                logger.info("Created new Pinecone index: %s", self.index_name)

            # Connect to index
            self._index = pinecone.Index(self.index_name)

            # Get index stats
            stats = self._index.describe_index_stats()
            logger.info(
                "Pinecone connected: index=%s, dimension=%d, vectors=%d",
                self.index_name,
                self.dimension,
                stats.total_vector_count,
            )
            return True

        except ImportError:
            logger.error("pinecone-client not installed. Run: pip install pinecone-client")
            return False
        except Exception as e:
            logger.error("Failed to initialize Pinecone: %s", e)
            return False

    async def add_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        contents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Add vectors to Pinecone index."""
        if not self._index:
            logger.error("Pinecone index not initialized")
            return False

        try:
            # Prepare vectors for upsert
            upsert_data = []
            for i, (vector, doc_id) in enumerate(zip(vectors, ids)):
                vector_data = {
                    "id": doc_id,
                    "values": vector,
                    "metadata": {
                        "content": contents[i],
                        **(metadata[i] if metadata and i < len(metadata) else {}),
                    }
                }
                upsert_data.append(vector_data)

            # Upsert in batches
            batch_size = 100
            for i in range(0, len(upsert_data), batch_size):
                batch = upsert_data[i : i + batch_size]
                self._index.upsert(vectors=batch)

            logger.info("Added %d vectors to Pinecone index", len(vectors))
            return True

        except Exception as e:
            logger.error("Failed to add vectors to Pinecone: %s", e)
            return False

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """Search for similar vectors in Pinecone index."""
        if not self._index:
            logger.error("Pinecone index not initialized")
            return []

        try:
            # Search
            response = self._index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
            )

            results = []
            for match in response.matches:
                results.append((
                    match.id,
                    match.score,
                    match.metadata.get("content", ""),
                    {k: v for k, v in match.metadata.items() if k != "content"},
                ))

            logger.debug("Pinecone search returned %d results", len(results))
            return results

        except Exception as e:
            logger.error("Failed to search Pinecone: %s", e)
            return []

    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Pinecone index."""
        if not self._index:
            logger.error("Pinecone index not initialized")
            return False

        try:
            self._index.delete(ids=ids)
            logger.info("Deleted %d vectors from Pinecone", len(ids))
            return True

        except Exception as e:
            logger.error("Failed to delete vectors from Pinecone: %s", e)
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self._index:
            return {"error": "Index not initialized"}

        try:
            stats = self._index.describe_index_stats()
            return {
                "index_name": self.index_name,
                "dimension": self.dimension,
                "total_vectors": stats.total_vector_count,
                "metric": self.metric,
            }
        except Exception as e:
            return {"error": str(e)}