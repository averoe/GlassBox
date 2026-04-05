"""PostgreSQL database plugin — production implementation with asyncpg connection pool."""

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from glassbox_rag.plugins.base import DatabasePlugin
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class PostgreSQLDatabase(DatabasePlugin):
    """
    PostgreSQL database using asyncpg with connection pooling.

    Uses asyncpg (fully async) instead of psycopg2 (sync), with
    a configurable pool size (min_size=2, max_size=20).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 5432)
        self.database = config.get("database", "glassbox_db")
        self.user = config.get("user", "glassbox")
        self.password = config.get("password", "")
        self.min_pool = config.get("min_pool_size", 2)
        self.max_pool = config.get("max_pool_size", 20)
        self._pool = None

    async def initialize(self) -> bool:
        try:
            connected = await self.connect()
            if connected:
                await self.ensure_tables()
            return connected
        except Exception as e:
            logger.error("Failed to initialize PostgreSQL: %s", e)
            return False

    async def connect(self) -> bool:
        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.min_pool,
                max_size=self.max_pool,
            )
            logger.info(
                "PostgreSQL connected: %s@%s:%d/%s (pool %d-%d)",
                self.user, self.host, self.port, self.database,
                self.min_pool, self.max_pool,
            )
            return True

        except ImportError:
            # Fallback to psycopg2 if asyncpg not available
            logger.warning("asyncpg not installed, trying psycopg2 fallback")
            return await self._connect_psycopg2()
        except Exception as e:
            logger.error("Failed to connect to PostgreSQL: %s", e)
            return False

    async def _connect_psycopg2(self) -> bool:
        """Fallback sync connection using psycopg2."""
        try:
            import psycopg2

            self._sync_conn = psycopg2.connect(
                host=self.host, port=self.port,
                database=self.database, user=self.user,
                password=self.password,
            )
            self._sync_conn.autocommit = False
            self._pool = None  # Mark that we're using sync mode
            logger.info("PostgreSQL connected (psycopg2 sync fallback)")
            return True
        except ImportError:
            logger.error(
                "Neither asyncpg nor psycopg2 installed. "
                "Run: pip install glassbox-rag[databases]"
            )
            return False
        except Exception as e:
            logger.error("psycopg2 fallback failed: %s", e)
            return False

    async def shutdown(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
        elif hasattr(self, "_sync_conn") and self._sync_conn:
            self._sync_conn.close()

    async def health_check(self) -> bool:
        if self._pool:
            try:
                async with self._pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                return True
            except Exception:
                return False
        return False

    async def ensure_tables(self) -> None:
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_created
                ON documents(created_at)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_content_fts
                ON documents USING gin(to_tsvector('english', content))
            """)
        logger.debug("PostgreSQL tables ensured")

    async def insert(self, table: str, data: Dict[str, Any]) -> str:
        if not self._pool:
            raise RuntimeError("Database not connected")

        record_id = data.get("id", str(uuid4()))
        content = data.get("content", "")
        metadata = json.dumps(data.get("metadata", {}))

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {table} (id, content, metadata)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """,
                record_id, content, metadata,
            )
        return record_id

    async def update(self, table: str, record_id: str, data: Dict[str, Any]) -> bool:
        if not self._pool:
            raise RuntimeError("Database not connected")

        sets = ["updated_at = NOW()"]
        args: List[Any] = []
        idx = 1

        if "content" in data:
            sets.append(f"content = ${idx}")
            args.append(data["content"])
            idx += 1
        if "metadata" in data:
            sets.append(f"metadata = ${idx}::jsonb")
            args.append(json.dumps(data["metadata"]))
            idx += 1

        args.append(record_id)
        query = f"UPDATE {table} SET {', '.join(sets)} WHERE id = ${idx}"

        async with self._pool.acquire() as conn:
            result = await conn.execute(query, *args)
            return "UPDATE 1" in result

    async def delete(self, table: str, record_id: str) -> bool:
        if not self._pool:
            raise RuntimeError("Database not connected")

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {table} WHERE id = $1", record_id
            )
            return "DELETE 1" in result

    async def query(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("Database not connected")

        query_str = f"SELECT * FROM {table}"
        args: List[Any] = []

        if filters:
            conditions = []
            for i, (key, value) in enumerate(filters.items(), 1):
                conditions.append(f"{key} = ${i}")
                args.append(value)
            query_str += " WHERE " + " AND ".join(conditions)

        query_str += " ORDER BY created_at DESC LIMIT 1000"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query_str, *args)
            return [dict(row) for row in rows]

    async def search_text(
        self,
        terms: List[str],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Full-text search using PostgreSQL tsvector + GIN index."""
        if not self._pool:
            raise RuntimeError("Database not connected")

        query_string = " | ".join(terms)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, metadata,
                       ts_rank(to_tsvector('english', content),
                               to_tsquery('english', $1)) AS score
                FROM documents
                WHERE to_tsvector('english', content) @@ to_tsquery('english', $1)
                ORDER BY score DESC
                LIMIT $2
                """,
                query_string, top_k,
            )
            return [dict(row) for row in rows]
