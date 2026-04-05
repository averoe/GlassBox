"""SQLite database plugin — real implementation using aiosqlite."""

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from glassbox_rag.plugins.base import DatabasePlugin
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteDatabase(DatabasePlugin):
    """SQLite database implementation using aiosqlite for async I/O."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.path = config.get("path", "./data/glassbox.db")
        self._db = None

    async def initialize(self) -> bool:
        try:
            connected = await self.connect()
            if connected:
                await self.ensure_tables()
            return connected
        except Exception as e:
            logger.error("Failed to initialize SQLite: %s", e)
            return False

    async def connect(self) -> bool:
        try:
            import aiosqlite
            import os

            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._db = await aiosqlite.connect(self.path)
            self._db.row_factory = aiosqlite.Row
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA foreign_keys=ON")
            logger.info("SQLite connected: %s", self.path)
            return True
        except ImportError:
            logger.error("aiosqlite not installed. Run: pip install glassbox-rag[databases]")
            return False
        except Exception as e:
            logger.error("Failed to connect to SQLite: %s", e)
            return False

    async def shutdown(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def health_check(self) -> bool:
        if self._db is None:
            return False
        try:
            async with self._db.execute("SELECT 1") as cursor:
                await cursor.fetchone()
            return True
        except Exception:
            return False

    async def ensure_tables(self) -> None:
        if self._db is None:
            return
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_created
            ON documents(created_at)
        """)
        await self._db.commit()
        logger.debug("SQLite tables ensured")

    async def insert(self, table: str, data: Dict[str, Any]) -> str:
        if self._db is None:
            raise RuntimeError("Database not connected")

        record_id = data.get("id", str(uuid4()))
        content = data.get("content", "")
        metadata = json.dumps(data.get("metadata", {}))

        await self._db.execute(
            f"INSERT OR REPLACE INTO {table} (id, content, metadata) VALUES (?, ?, ?)",
            (record_id, content, metadata),
        )
        await self._db.commit()
        return record_id

    async def update(self, table: str, record_id: str, data: Dict[str, Any]) -> bool:
        if self._db is None:
            raise RuntimeError("Database not connected")

        set_clauses = []
        values = []

        if "content" in data:
            set_clauses.append("content = ?")
            values.append(data["content"])
        if "metadata" in data:
            set_clauses.append("metadata = ?")
            values.append(json.dumps(data["metadata"]))

        set_clauses.append("updated_at = CURRENT_TIMESTAMP")
        values.append(record_id)

        query = f"UPDATE {table} SET {', '.join(set_clauses)} WHERE id = ?"
        cursor = await self._db.execute(query, values)
        await self._db.commit()
        return cursor.rowcount > 0

    async def delete(self, table: str, record_id: str) -> bool:
        if self._db is None:
            raise RuntimeError("Database not connected")

        cursor = await self._db.execute(
            f"DELETE FROM {table} WHERE id = ?", (record_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def query(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._db is None:
            raise RuntimeError("Database not connected")

        query_str = f"SELECT * FROM {table}"
        values: List[Any] = []

        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"{key} = ?")
                values.append(value)
            query_str += " WHERE " + " AND ".join(conditions)

        query_str += " ORDER BY created_at DESC LIMIT 1000"

        rows = []
        async with self._db.execute(query_str, values) as cursor:
            async for row in cursor:
                row_dict = dict(row)
                if "metadata" in row_dict and isinstance(row_dict["metadata"], str):
                    try:
                        row_dict["metadata"] = json.loads(row_dict["metadata"])
                    except json.JSONDecodeError:
                        pass
                rows.append(row_dict)

        return rows

    async def search_text(
        self,
        terms: List[str],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Simple LIKE-based text search."""
        if self._db is None:
            raise RuntimeError("Database not connected")

        conditions = []
        values: List[str] = []
        for term in terms:
            conditions.append("content LIKE ?")
            values.append(f"%{term}%")

        query = f"SELECT * FROM documents WHERE {' OR '.join(conditions)} LIMIT ?"
        values.append(str(top_k))

        rows = []
        async with self._db.execute(query, values) as cursor:
            async for row in cursor:
                row_dict = dict(row)
                # Simple relevance score: count matching terms
                content_lower = row_dict.get("content", "").lower()
                match_count = sum(1 for t in terms if t.lower() in content_lower)
                row_dict["score"] = match_count / max(len(terms), 1)
                if "metadata" in row_dict and isinstance(row_dict["metadata"], str):
                    try:
                        row_dict["metadata"] = json.loads(row_dict["metadata"])
                    except json.JSONDecodeError:
                        row_dict["metadata"] = {}
                rows.append(row_dict)

        return sorted(rows, key=lambda x: x.get("score", 0), reverse=True)
