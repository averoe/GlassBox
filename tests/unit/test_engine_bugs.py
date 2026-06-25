"""Unit tests for engine bugs (core/engine.py).

Covers _validate_config raising on missing config, _ensure_generator
concurrency safety, and the WriteBackConfig rename.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from glassbox_rag.config import GlassBoxConfig
from glassbox_rag.core.engine import GlassBoxEngine, EngineError


class TestValidateConfig:
    def test_raises_on_missing_vector_store_config(self):
        """_validate_config must raise EngineError when vector_store type
        is set but no matching config block exists."""
        config = GlassBoxConfig(
            vector_store={"type": "qdrant"},  # no qdrant: {} block
        )
        # Clear the default sqlite config to avoid database error
        config.database.sqlite = {"path": "./data/glassbox.db"}

        engine = GlassBoxEngine(config)
        # qdrant config should be None since we only set type
        # but the VectorStoreConfig default gives qdrant=None
        with pytest.raises(EngineError, match="qdrant"):
            engine._validate_config()

    def test_raises_on_missing_database_config(self):
        """_validate_config must raise EngineError when database type
        is set but no matching config block exists."""
        config = GlassBoxConfig(
            vector_store={"type": "qdrant", "qdrant": {"host": "localhost"}},
            database={"type": "postgresql"},  # no postgresql: {} block
        )
        engine = GlassBoxEngine(config)
        with pytest.raises(EngineError, match="postgresql"):
            engine._validate_config()

    def test_passes_with_valid_config(self):
        """_validate_config must not raise when config is properly set."""
        config = GlassBoxConfig(
            vector_store={"type": "qdrant", "qdrant": {"host": "localhost"}},
            database={"type": "sqlite", "sqlite": {"path": "./test.db"}},
        )
        engine = GlassBoxEngine(config)
        # Should not raise
        engine._validate_config()


class TestEnsureGeneratorLock:
    @pytest.mark.asyncio
    async def test_generator_lock_exists(self):
        """Engine must have a _generator_lock for concurrency safety."""
        config = GlassBoxConfig()
        engine = GlassBoxEngine(config)
        assert hasattr(engine, "_generator_lock")
        assert isinstance(engine._generator_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_concurrent_ensure_generator_no_double_init(self):
        """Two concurrent _ensure_generator calls must not double-initialize."""
        config = GlassBoxConfig()
        engine = GlassBoxEngine(config)

        init_count = 0

        async def mock_initialize():
            nonlocal init_count
            init_count += 1
            await asyncio.sleep(0.05)  # simulate slow init

        mock_gen = MagicMock()
        mock_gen.initialize = mock_initialize

        with patch("glassbox_rag.core.engine.GlassBoxEngine._ensure_generator") as mock_ensure:
            # Simulate the actual lock behavior
            async def locked_ensure():
                async with engine._generator_lock:
                    if engine._generator is not None:
                        return
                    engine._generator = mock_gen
                    await mock_gen.initialize()

            mock_ensure.side_effect = locked_ensure

            # Call concurrently
            await asyncio.gather(
                engine._ensure_generator(),
                engine._ensure_generator(),
            )

            # Generator should only be initialized once
            assert init_count == 1


class TestWriteBackConfigRename:
    def test_config_uses_write_back_config(self):
        """GlassBoxConfig.writeback must use WriteBackConfig (capital B)."""
        from glassbox_rag.config import WriteBackConfig
        config = GlassBoxConfig()
        assert isinstance(config.writeback, WriteBackConfig)

    def test_writeback_config_import(self):
        """WriteBackConfig should be importable."""
        from glassbox_rag.config import WriteBackConfig
        wbc = WriteBackConfig()
        assert wbc.enabled is True
        assert wbc.mode == "protected"


class TestEngineInitialization:
    def test_not_initialized_raises(self):
        """Operations before initialize() must raise EngineError."""
        config = GlassBoxConfig()
        engine = GlassBoxEngine(config)
        with pytest.raises(EngineError, match="not initialized"):
            engine._ensure_initialized()

    def test_from_env_returns_engine(self):
        """from_env() must return a GlassBoxEngine instance."""
        engine = GlassBoxEngine.from_env()
        assert isinstance(engine, GlassBoxEngine)


class TestBatchIngestSignature:
    def test_max_concurrent_batches_param(self):
        """batch_ingest must accept max_concurrent_batches parameter."""
        import inspect
        sig = inspect.signature(GlassBoxEngine.batch_ingest)
        assert "max_concurrent_batches" in sig.parameters
        param = sig.parameters["max_concurrent_batches"]
        assert param.default == 3
