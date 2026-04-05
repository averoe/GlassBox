"""MongoDB database plugin — real implementation."""

from typing import Any, Dict, List, Optional, Tuple
import json

from glassbox_rag.plugins.base import DatabasePlugin
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class MongoDBDatabase(DatabasePlugin):
    """MongoDB database implementation using pymongo."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 27017)
        self.database_name = config.get("database", "glassbox_rag")
        self.collection_name = config.get("collection", "documents")
        self.username = config.get("username")
        self.password = config.get("password")
        self.auth_source = config.get("auth_source", "admin")
        self._client = None
        self._db = None
        self._collection = None

    async def initialize(self) -> bool:
        """Initialize MongoDB client and ensure database/collection exists."""
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure

            # Build connection string
            connection_string = f"mongodb://{self.host}:{self.port}"
            if self.username and self.password:
                connection_string = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/?authSource={self.auth_source}"

            # Connect to MongoDB
            self._client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)

            # Test connection
            self._client.admin.command('ping')
            logger.info("Connected to MongoDB at %s:%d", self.host, self.port)

            # Get database and collection
            self._db = self._client[self.database_name]
            self._collection = self._db[self.collection_name]

            # Create indexes
            self._collection.create_index("id", unique=True)
            self._collection.create_index("content")  # For text search
            self._collection.create_index([("embedding", "2dsphere")])  # For geospatial queries if needed

            doc_count = self._collection.count_documents({})
            logger.info(
                "MongoDB initialized: db=%s, collection=%s, documents=%d",
                self.database_name,
                self.collection_name,
                doc_count,
            )
            return True

        except ImportError:
            logger.error("pymongo not installed. Run: pip install pymongo")
            return False
        except Exception as e:
            logger.error("Failed to initialize MongoDB: %s", e)
            return False

    async def store_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Store a document in MongoDB."""
        if not self._collection:
            logger.error("MongoDB collection not initialized")
            return False

        try:
            document = {
                "id": doc_id,
                "content": content,
                "metadata": metadata or {},
                "embedding": embedding,
                "created_at": {"$date": None},  # Will be set by MongoDB
                "updated_at": {"$date": None},
            }

            # Upsert document
            result = self._collection.replace_one(
                {"id": doc_id},
                document,
                upsert=True
            )

            logger.debug("Stored document: %s", doc_id)
            return True

        except Exception as e:
            logger.error("Failed to store document %s: %s", doc_id, e)
            return False

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document from MongoDB."""
        if not self._collection:
            logger.error("MongoDB collection not initialized")
            return None

        try:
            doc = self._collection.find_one({"id": doc_id})
            if doc:
                # Convert MongoDB document to our format
                return {
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "embedding": doc.get("embedding"),
                }
            return None

        except Exception as e:
            logger.error("Failed to get document %s: %s", doc_id, e)
            return None

    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search documents using MongoDB text search."""
        if not self._collection:
            logger.error("MongoDB collection not initialized")
            return []

        try:
            # Build search query
            search_query = {"$text": {"$search": query}}
            if metadata_filter:
                search_query.update(metadata_filter)

            # Search with text score
            cursor = self._collection.find(
                search_query,
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)

            results = []
            for doc in cursor:
                results.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "score": doc.get("score", 0.0),
                })

            logger.debug("MongoDB search returned %d results", len(results))
            return results

        except Exception as e:
            logger.error("Failed to search documents: %s", e)
            return []

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from MongoDB."""
        if not self._collection:
            logger.error("MongoDB collection not initialized")
            return False

        try:
            result = self._collection.delete_one({"id": doc_id})
            deleted = result.deleted_count > 0
            if deleted:
                logger.debug("Deleted document: %s", doc_id)
            return deleted

        except Exception as e:
            logger.error("Failed to delete document %s: %s", doc_id, e)
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self._collection:
            return {"error": "Collection not initialized"}

        try:
            stats = {
                "database": self.database_name,
                "collection": self.collection_name,
                "document_count": self._collection.count_documents({}),
                "indexes": list(self._collection.list_indexes()),
            }
            return stats

        except Exception as e:
            return {"error": str(e)}