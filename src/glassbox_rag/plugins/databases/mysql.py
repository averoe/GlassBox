"""MySQL database plugin — real implementation."""

from typing import Any, Dict, List, Optional, Tuple
import json

from glassbox_rag.plugins.base import DatabasePlugin
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class MySQLDatabase(DatabasePlugin):
    """MySQL database implementation using mysql-connector-python."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 3306)
        self.database = config.get("database", "glassbox_rag")
        self.username = config.get("username", "root")
        self.password = config.get("password", "")
        self.table_name = config.get("table", "documents")
        self._connection = None

    async def initialize(self) -> bool:
        """Initialize MySQL connection and ensure table exists."""
        try:
            import mysql.connector
            from mysql.connector import Error

            # Connect to MySQL
            self._connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
            )

            if self._connection.is_connected():
                logger.info("Connected to MySQL at %s:%d", self.host, self.port)

                # Create table if it doesn't exist
                cursor = self._connection.cursor()
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id VARCHAR(255) PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSON,
                    embedding JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FULLTEXT(content)
                )
                """
                cursor.execute(create_table_query)
                self._connection.commit()

                # Get document count
                cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                count = cursor.fetchone()[0]

                logger.info(
                    "MySQL initialized: db=%s, table=%s, documents=%d",
                    self.database,
                    self.table_name,
                    count,
                )
                return True

        except ImportError:
            logger.error("mysql-connector-python not installed. Run: pip install mysql-connector-python")
            return False
        except Exception as e:
            logger.error("Failed to initialize MySQL: %s", e)
            return False

    async def store_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Store a document in MySQL."""
        if not self._connection:
            logger.error("MySQL connection not initialized")
            return False

        try:
            cursor = self._connection.cursor()

            # Insert or update document
            query = f"""
            INSERT INTO {self.table_name} (id, content, metadata, embedding)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                content = VALUES(content),
                metadata = VALUES(metadata),
                embedding = VALUES(embedding),
                updated_at = CURRENT_TIMESTAMP
            """

            metadata_json = json.dumps(metadata) if metadata else None
            embedding_json = json.dumps(embedding) if embedding else None

            cursor.execute(query, (doc_id, content, metadata_json, embedding_json))
            self._connection.commit()

            logger.debug("Stored document: %s", doc_id)
            return True

        except Exception as e:
            logger.error("Failed to store document %s: %s", doc_id, e)
            return False

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document from MySQL."""
        if not self._connection:
            logger.error("MySQL connection not initialized")
            return None

        try:
            cursor = self._connection.cursor(dictionary=True)

            query = f"SELECT * FROM {self.table_name} WHERE id = %s"
            cursor.execute(query, (doc_id,))

            result = cursor.fetchone()
            if result:
                # Parse JSON fields
                metadata = json.loads(result["metadata"]) if result["metadata"] else {}
                embedding = json.loads(result["embedding"]) if result["embedding"] else None

                return {
                    "id": result["id"],
                    "content": result["content"],
                    "metadata": metadata,
                    "embedding": embedding,
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
        """Search documents using MySQL full-text search."""
        if not self._connection:
            logger.error("MySQL connection not initialized")
            return []

        try:
            cursor = self._connection.cursor(dictionary=True)

            # Build search query
            search_query = f"""
            SELECT *, MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE) as relevance
            FROM {self.table_name}
            WHERE MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE)
            """

            params = [query, query]

            # Add metadata filters if provided
            if metadata_filter:
                conditions = []
                for key, value in metadata_filter.items():
                    conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') = %s")
                    params.append(value)

                if conditions:
                    search_query += " AND " + " AND ".join(conditions)

            search_query += " ORDER BY relevance DESC LIMIT %s"
            params.append(limit)

            cursor.execute(search_query, params)

            results = []
            for row in cursor.fetchall():
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                embedding = json.loads(row["embedding"]) if row["embedding"] else None

                results.append({
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": metadata,
                    "embedding": embedding,
                    "score": float(row["relevance"]),
                })

            logger.debug("MySQL search returned %d results", len(results))
            return results

        except Exception as e:
            logger.error("Failed to search documents: %s", e)
            return []

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from MySQL."""
        if not self._connection:
            logger.error("MySQL connection not initialized")
            return False

        try:
            cursor = self._connection.cursor()

            query = f"DELETE FROM {self.table_name} WHERE id = %s"
            cursor.execute(query, (doc_id,))
            self._connection.commit()

            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug("Deleted document: %s", doc_id)
            return deleted

        except Exception as e:
            logger.error("Failed to delete document %s: %s", doc_id, e)
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self._connection:
            return {"error": "Connection not initialized"}

        try:
            cursor = self._connection.cursor()

            # Get table statistics
            cursor.execute(f"SELECT COUNT(*) as count FROM {self.table_name}")
            count_result = cursor.fetchone()

            return {
                "database": self.database,
                "table": self.table_name,
                "document_count": count_result[0] if count_result else 0,
                "host": self.host,
                "port": self.port,
            }

        except Exception as e:
            return {"error": str(e)}