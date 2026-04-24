"""
Trace storage backends — Redis and PostgreSQL.

Allows persisting traces beyond in-memory storage for
production deployments.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class TraceBackend(ABC):
    """Base class for trace storage backends."""

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the backend."""

    @abstractmethod
    async def store_trace(self, trace_id: str, trace_data: Dict) -> None:
        """Store a completed trace."""

    @abstractmethod
    async def get_trace(self, trace_id: str) -> Dict | None:
        """Retrieve a trace by ID."""

    @abstractmethod
    async def list_traces(self, limit: int = 100) -> list[Dict]:
        """List recent traces."""

    @abstractmethod
    async def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace by ID."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored traces."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the backend."""


class MemoryBackend(TraceBackend):
    """In-memory trace storage (default)."""

    def __init__(self, max_traces: int = 10000):
        self._traces: dict[str, Dict] = {}
        self._max = max_traces

    async def initialize(self) -> bool:
        return True

    async def store_trace(self, trace_id: str, trace_data: Dict) -> None:
        self._traces[trace_id] = trace_data
        if len(self._traces) > self._max:
            oldest = min(self._traces, key=lambda k: self._traces[k].get("start_time", ""))
            del self._traces[oldest]

    async def get_trace(self, trace_id: str) -> Dict | None:
        return self._traces.get(trace_id)

    async def list_traces(self, limit: int = 100) -> list[Dict]:
        sorted_traces = sorted(
            self._traces.values(),
            key=lambda t: t.get("start_time", ""),
            reverse=True,
        )
        return sorted_traces[:limit]

    async def delete_trace(self, trace_id: str) -> bool:
        return self._traces.pop(trace_id, None) is not None

    async def clear(self) -> None:
        self._traces.clear()

    async def shutdown(self) -> None:
        pass


class RedisBackend(TraceBackend):
    """Redis-backed trace storage.

    Requires: pip install redis
    Uses sorted sets for ordering by timestamp and hash maps for data.
    """

    def __init__(self, config: dict[str, Any]):
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6379)
        self.db = config.get("db", 0)
        self.password = config.get("password")
        self.prefix = config.get("prefix", "glassbox:trace:")
        self.ttl_seconds = config.get("ttl_seconds", 86400 * 30)  # 30 days
        self._client = None

    async def initialize(self) -> bool:
        try:
            import redis.asyncio as aioredis

            self._client = aioredis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
            )
            await self._client.ping()
            logger.info("Redis trace backend connected: %s:%d", self.host, self.port)
            return True
        except ImportError:
            logger.error("redis package not installed. Run: pip install redis")
            return False
        except Exception as e:
            logger.error("Failed to connect to Redis: %s", e)
            return False

    async def store_trace(self, trace_id: str, trace_data: Dict) -> None:
        if not self._client:
            return
        key = f"{self.prefix}{trace_id}"
        data = json.dumps(trace_data, default=str)
        await self._client.set(key, data, ex=self.ttl_seconds)
        # Add to sorted set for listing
        start_time = trace_data.get("start_time", "")
        await self._client.zadd(
            f"{self.prefix}index",
            {trace_id: hash(start_time) & 0xFFFFFFFF},
        )

    async def get_trace(self, trace_id: str) -> Dict | None:
        if not self._client:
            return None
        key = f"{self.prefix}{trace_id}"
        data = await self._client.get(key)
        if data:
            return json.loads(data)
        return None

    async def list_traces(self, limit: int = 100) -> list[Dict]:
        if not self._client:
            return []
        ids = await self._client.zrevrange(f"{self.prefix}index", 0, limit - 1)
        traces = []
        for trace_id in ids:
            trace = await self.get_trace(trace_id)
            if trace:
                traces.append(trace)
        return traces

    async def delete_trace(self, trace_id: str) -> bool:
        if not self._client:
            return False
        key = f"{self.prefix}{trace_id}"
        deleted = await self._client.delete(key)
        await self._client.zrem(f"{self.prefix}index", trace_id)
        return deleted > 0

    async def clear(self) -> None:
        if not self._client:
            return
        # Get all trace keys and delete
        keys = []
        async for key in self._client.scan_iter(f"{self.prefix}*"):
            keys.append(key)
        if keys:
            await self._client.delete(*keys)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.close()


class PostgreSQLBackend(TraceBackend):
    """PostgreSQL-backed trace storage.

    Requires: pip install asyncpg
    Uses JSONB for trace data with indexed trace_id and timestamp.
    """

    def __init__(self, config: dict[str, Any]):
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 5432)
        self.database = config.get("database", "glassbox_db")
        self.user = config.get("user", "glassbox")
        self.password = config.get("password", "")
        self.retention_days = config.get("retention_days", 30)
        self._pool = None

    async def initialize(self) -> bool:
        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=2,
                max_size=10,
            )
            # Create table
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS glassbox_traces (
                        trace_id TEXT PRIMARY KEY,
                        request_id TEXT,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        duration_ms FLOAT,
                        trace_data JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_traces_start
                    ON glassbox_traces(start_time DESC)
                """)

            logger.info(
                "PostgreSQL trace backend connected: %s:%d/%s (pool 2-10)",
                self.host, self.port, self.database,
            )
            return True
        except ImportError:
            logger.error("asyncpg not installed. Run: pip install asyncpg")
            return False
        except Exception as e:
            logger.error("Failed to init PostgreSQL trace backend: %s", e)
            return False

    async def store_trace(self, trace_id: str, trace_data: Dict) -> None:
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO glassbox_traces
                    (trace_id, request_id, start_time, end_time, duration_ms, trace_data)
                VALUES ($1, $2, $3::timestamp, $4::timestamp, $5, $6::jsonb)
                ON CONFLICT (trace_id) DO UPDATE SET trace_data = $6::jsonb
                """,
                trace_id,
                trace_data.get("request_id", ""),
                trace_data.get("start_time"),
                trace_data.get("end_time"),
                trace_data.get("duration_ms", 0),
                json.dumps(trace_data, default=str),
            )

    async def get_trace(self, trace_id: str) -> Dict | None:
        if not self._pool:
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT trace_data FROM glassbox_traces WHERE trace_id = $1",
                trace_id,
            )
            if row:
                return json.loads(row["trace_data"])
        return None

    async def list_traces(self, limit: int = 100) -> list[Dict]:
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT trace_data FROM glassbox_traces
                ORDER BY start_time DESC LIMIT $1
                """,
                limit,
            )
            return [json.loads(row["trace_data"]) for row in rows]

    async def delete_trace(self, trace_id: str) -> bool:
        if not self._pool:
            return False
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM glassbox_traces WHERE trace_id = $1", trace_id
            )
            return "DELETE 1" in result

    async def clear(self) -> None:
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            await conn.execute("TRUNCATE glassbox_traces")

    async def shutdown(self) -> None:
        if self._pool:
            await self._pool.close()


# ═══════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════

async def create_trace_backend(
    backend_type: str,
    config: dict[str, Any],
) -> TraceBackend:
    """Create and initialize a trace backend."""
    backends = {
        "memory": lambda c: MemoryBackend(c.get("max_traces", 10000)),
        "redis": lambda c: RedisBackend(c),
        "postgresql": lambda c: PostgreSQLBackend(c),
    }

    if backend_type not in backends:
        available = list(backends.keys())
        raise ValueError(
            f"Unknown trace backend '{backend_type}'. Available: {available}"
        )

    backend = backends[backend_type](config)
    if not await backend.initialize():
        logger.warning(
            "Trace backend '%s' failed to initialize, falling back to memory",
            backend_type,
        )
        backend = MemoryBackend()
        await backend.initialize()

    return backend
