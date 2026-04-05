"""
Integration tests that run against real backends.

These tests require Docker services running:
    docker compose -f docker-compose.test.yml up -d qdrant postgres redis

Skip with: pytest -m 'not integration_real'
"""

import os
import pytest

# Mark all tests in this module
pytestmark = pytest.mark.integration_real

# Backend connection config from env vars (set by docker-compose.test.yml)
QDRANT_HOST = os.getenv("GLASSBOX_QDRANT_HOST", "localhost")
POSTGRES_HOST = os.getenv("GLASSBOX_POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("GLASSBOX_POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("GLASSBOX_POSTGRES_USER", "glassbox")
POSTGRES_PASSWORD = os.getenv("GLASSBOX_POSTGRES_PASSWORD", "glassbox_test")
POSTGRES_DB = os.getenv("GLASSBOX_POSTGRES_DB", "glassbox_test")
REDIS_URL = os.getenv("GLASSBOX_REDIS_URL", "redis://localhost:6379")


def _check_service(host: str, port: int) -> bool:
    """Check if a service is reachable."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


# ═══════════════════════════════════════════════════════════════════
#  Qdrant
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not _check_service(QDRANT_HOST, 6333),
    reason="Qdrant not running"
)
class TestQdrantIntegration:
    @pytest.mark.asyncio
    async def test_qdrant_lifecycle(self):
        from glassbox_rag.plugins.vector_stores.qdrant import QdrantVectorStore

        store = QdrantVectorStore({
            "host": QDRANT_HOST,
            "port": 6333,
            "collection_name": "test_glassbox",
            "vector_size": 4,
        })

        assert await store.initialize()

        # Health check
        assert await store.health_check()

        # Upsert
        ids = await store.add_vectors(
            vectors=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            contents=["hello world", "goodbye world"],
            metadata=[{"source": "test"}, {"source": "test"}],
        )
        assert len(ids) == 2

        # Search
        results = await store.search(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            top_k=2,
        )
        assert len(results) > 0

        # Cleanup
        await store.shutdown()

    @pytest.mark.asyncio
    async def test_qdrant_count(self):
        from glassbox_rag.plugins.vector_stores.qdrant import QdrantVectorStore

        store = QdrantVectorStore({
            "host": QDRANT_HOST,
            "port": 6333,
            "collection_name": "test_count",
            "vector_size": 4,
        })
        await store.initialize()

        await store.add_vectors(
            vectors=[[0.1, 0.2, 0.3, 0.4]],
            contents=["test"],
            metadata=[{}],
        )
        count = await store.count()
        assert count >= 1

        await store.shutdown()


# ═══════════════════════════════════════════════════════════════════
#  PostgreSQL (with asyncpg connection pool)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not _check_service(POSTGRES_HOST, POSTGRES_PORT),
    reason="PostgreSQL not running"
)
class TestPostgresIntegration:
    @pytest.mark.asyncio
    async def test_postgres_lifecycle(self):
        from glassbox_rag.plugins.databases.postgresql import PostgreSQLDatabase

        db = PostgreSQLDatabase({
            "host": POSTGRES_HOST,
            "port": POSTGRES_PORT,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "database": POSTGRES_DB,
            "min_pool_size": 2,
            "max_pool_size": 5,
        })

        assert await db.initialize()
        assert await db.health_check()

        # Insert
        doc_id = await db.insert("documents", {
            "content": "Test document content",
            "metadata": {"source": "integration_test"},
        })
        assert doc_id

        # Query
        results = await db.query("documents", {"id": doc_id})
        assert len(results) >= 1

        # Update
        updated = await db.update("documents", doc_id, {
            "content": "Updated content",
        })
        assert updated

        # Delete
        deleted = await db.delete("documents", doc_id)
        assert deleted

        await db.shutdown()

    @pytest.mark.asyncio
    async def test_postgres_fulltext_search(self):
        from glassbox_rag.plugins.databases.postgresql import PostgreSQLDatabase

        db = PostgreSQLDatabase({
            "host": POSTGRES_HOST,
            "port": POSTGRES_PORT,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "database": POSTGRES_DB,
        })
        await db.initialize()

        await db.insert("documents", {
            "id": "fts_test_001",
            "content": "Machine learning algorithms improve over time",
            "metadata": {},
        })

        results = await db.search_text(["machine", "learning"], top_k=5)
        assert len(results) >= 1
        assert any("machine" in r.get("content", "").lower() for r in results)

        await db.delete("documents", "fts_test_001")
        await db.shutdown()


# ═══════════════════════════════════════════════════════════════════
#  Redis (rate limiter + trace backend)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not _check_service("localhost", 6379),
    reason="Redis not running"
)
class TestRedisIntegration:
    @pytest.mark.asyncio
    async def test_redis_rate_limiter(self):
        from glassbox_rag.utils.auth import RedisRateLimiter

        limiter = RedisRateLimiter(
            redis_url=REDIS_URL,
            max_requests=5,
            window_seconds=60,
            prefix="test:ratelimit:",
        )
        assert await limiter.initialize()

        # Should allow first 5 requests
        for _ in range(5):
            assert await limiter.is_allowed("test_client")

        # 6th should be blocked
        assert not await limiter.is_allowed("test_client")

        # Different client should be fine
        assert await limiter.is_allowed("other_client")

        remaining = await limiter.get_remaining("test_client")
        assert remaining == 0

        await limiter.shutdown()

    @pytest.mark.asyncio
    async def test_redis_trace_backend(self):
        from glassbox_rag.trace.backends import RedisBackend

        backend = RedisBackend({
            "host": "localhost",
            "port": 6379,
            "prefix": "test:traces:",
            "ttl_seconds": 60,
        })
        assert await backend.initialize()

        # Store
        await backend.store_trace("test_trace_001", {
            "trace_id": "test_trace_001",
            "request_id": "req_001",
            "start_time": "2026-01-01T00:00:00",
        })

        # Retrieve
        trace = await backend.get_trace("test_trace_001")
        assert trace is not None
        assert trace["trace_id"] == "test_trace_001"

        # List
        traces = await backend.list_traces(10)
        assert len(traces) >= 1

        # Delete
        deleted = await backend.delete_trace("test_trace_001")
        assert deleted

        await backend.shutdown()


# ═══════════════════════════════════════════════════════════════════
#  JWT Auth
# ═══════════════════════════════════════════════════════════════════

class TestJWTAuth:
    """JWT auth doesn't need external services — always runs."""

    def test_create_and_validate_token(self):
        from glassbox_rag.utils.auth import JWTAuth

        auth = JWTAuth(secret="test_secret_key", issuer="glassbox")
        token = auth.create_token(subject="user123", expires_in=3600)

        claims = auth.validate_token(token)
        assert claims["sub"] == "user123"
        assert claims["iss"] == "glassbox"

    def test_expired_token(self):
        from glassbox_rag.utils.auth import JWTAuth, AuthError

        auth = JWTAuth(secret="test_secret", leeway_seconds=0)
        token = auth.create_token(subject="user", expires_in=-10)

        with pytest.raises(AuthError, match="expired"):
            auth.validate_token(token)

    def test_tampered_token(self):
        from glassbox_rag.utils.auth import JWTAuth, AuthError

        auth = JWTAuth(secret="correct_secret")
        token = auth.create_token(subject="user")

        # Tamper with the token
        parts = token.split(".")
        parts[1] = parts[1] + "tampered"
        tampered = ".".join(parts)

        with pytest.raises(AuthError):
            auth.validate_token(tampered)

    def test_wrong_secret(self):
        from glassbox_rag.utils.auth import JWTAuth, AuthError

        auth1 = JWTAuth(secret="secret_1")
        auth2 = JWTAuth(secret="secret_2")

        token = auth1.create_token(subject="user")
        with pytest.raises(AuthError, match="signature"):
            auth2.validate_token(token)

    def test_issuer_validation(self):
        from glassbox_rag.utils.auth import JWTAuth, AuthError

        auth = JWTAuth(secret="key", issuer="expected_issuer")
        # Create token with wrong issuer
        bad_auth = JWTAuth(secret="key", issuer="wrong_issuer")
        token = bad_auth.create_token(subject="user")

        with pytest.raises(AuthError, match="issuer"):
            auth.validate_token(token)


# ═══════════════════════════════════════════════════════════════════
#  Telemetry
# ═══════════════════════════════════════════════════════════════════

class TestTelemetry:
    """Telemetry tests that don't need external services."""

    def test_prometheus_metrics(self):
        from glassbox_rag.utils.telemetry import PrometheusMetrics

        prom = PrometheusMetrics()
        if prom._initialized:
            prom.record_request("retrieve", 0.5, tokens=100)
            prom.record_operation("encode", 0.1)
            prom.set_vectors_count(1000)

            output = prom.generate_metrics()
            assert b"glassbox" in output

    def test_telemetry_hub_status(self):
        from glassbox_rag.utils.telemetry import TelemetryHub

        hub = TelemetryHub(otel_enabled=False, prometheus_enabled=False)
        status = hub.get_telemetry_status()
        assert status["otel_enabled"] is False
        assert status["prometheus_enabled"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
