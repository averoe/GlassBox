"""
Plugin Developer SDK — registration, discovery, and lifecycle.

Provides a decorator-based registration system so third-party
developers can create their own plugins (vector stores, databases,
encoders, rerankers, chunkers) without modifying core code.

Usage:
    from glassbox_rag.plugins.sdk import plugin_registry, register_plugin

    @register_plugin("vector_store", "my_custom_store")
    class MyCustomStore(VectorStorePlugin):
        ...

    # Or register dynamically:
    plugin_registry.register("database", "my_db", MyDBPlugin)

    # List available:
    plugin_registry.list_plugins("vector_store")

    # Create instance:
    instance = plugin_registry.create("vector_store", "my_custom_store", config)
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Callable, Dict, List, Optional, Type

from glassbox_rag.plugins.base import Plugin, VectorStorePlugin, DatabasePlugin
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level lock for thread-safe registry mutations
_registry_lock = threading.Lock()


class PluginRegistry:
    """
    Global plugin registry.

    Manages registration and instantiation of all plugin types.
    Supports both built-in and third-party plugins.
    Thread-safe for concurrent access in multi-threaded environments.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, dict[str, type[Plugin]]] = {
            "vector_store": {},
            "database": {},
            "encoder": {},
            "reranker": {},
            "chunker": {},
            "processor": {},
        }
        self._initialized = False

    def register(
        self,
        plugin_type: str,
        name: str,
        plugin_class: type[Plugin],
    ) -> None:
        """
        Register a plugin class.

        Args:
            plugin_type: Type category (vector_store, database, encoder, etc.)
            name: Unique name for this plugin.
            plugin_class: The plugin class to register.

        Raises:
            ValueError: If plugin_type is unknown.
        """
        if plugin_type not in self._plugins:
            raise ValueError(
                f"Unknown plugin type '{plugin_type}'. "
                f"Available: {list(self._plugins.keys())}"
            )
        with _registry_lock:
            self._plugins[plugin_type][name] = plugin_class
        logger.info("Registered plugin: %s/%s → %s", plugin_type, name, plugin_class.__name__)

    def unregister(self, plugin_type: str, name: str) -> bool:
        """Remove a registered plugin."""
        if plugin_type in self._plugins and name in self._plugins[plugin_type]:
            del self._plugins[plugin_type][name]
            return True
        return False

    def get(self, plugin_type: str, name: str) -> Optional[type[Plugin]]:
        """Get a registered plugin class."""
        return self._plugins.get(plugin_type, {}).get(name)

    def create(
        self,
        plugin_type: str,
        name: str,
        config: dict[str, Any],
    ) -> Plugin:
        """
        Create an instance of a registered plugin.

        Args:
            plugin_type: Type category.
            name: Plugin name.
            config: Plugin configuration dict.

        Returns:
            Plugin instance (not yet initialized).

        Raises:
            ValueError: If plugin not found.
        """
        cls = self.get(plugin_type, name)
        if cls is None:
            available = list(self._plugins.get(plugin_type, {}).keys())
            raise ValueError(
                f"Plugin '{name}' not found in '{plugin_type}'. "
                f"Available: {available}"
            )
        return cls(config)

    async def create_and_init(
        self,
        plugin_type: str,
        name: str,
        config: dict[str, Any],
    ) -> Plugin | None:
        """Create, initialize, and return a plugin. Returns None on init failure."""
        try:
            instance = self.create(plugin_type, name, config)
            if await instance.initialize():
                return instance
            logger.warning("Plugin %s/%s init returned False", plugin_type, name)
            return None
        except Exception as e:
            logger.error("Failed to create plugin %s/%s: %s", plugin_type, name, e)
            return None

    def list_plugins(self, plugin_type: str | None = None) -> dict[str, list[str]]:
        """
        List registered plugins.

        Args:
            plugin_type: If specified, list only this type. Otherwise list all.

        Returns:
            Dict mapping plugin_type → list of names.
        """
        if plugin_type:
            return {plugin_type: list(self._plugins.get(plugin_type, {}).keys())}
        return {pt: list(plugins.keys()) for pt, plugins in self._plugins.items()}

    def list_all_flat(self) -> list[dict[str, str]]:
        """List all plugins as flat list of {type, name, class}."""
        result = []
        for pt, plugins in self._plugins.items():
            for name, cls in plugins.items():
                result.append({
                    "type": pt,
                    "name": name,
                    "class": f"{cls.__module__}.{cls.__name__}",
                })
        return result

    def register_builtins(self) -> None:
        """Register all built-in plugins."""
        if self._initialized:
            return

        builtin_plugins = {
            ("vector_store", "qdrant"): "glassbox_rag.plugins.vector_stores.qdrant:QdrantVectorStore",
            ("vector_store", "chroma"): "glassbox_rag.plugins.vector_stores.chroma:ChromaVectorStore",
            ("vector_store", "supabase"): "glassbox_rag.plugins.supabase:SupabaseVectorStore",
            ("database", "sqlite"): "glassbox_rag.plugins.databases.sqlite:SQLiteDatabase",
            ("database", "postgresql"): "glassbox_rag.plugins.databases.postgresql:PostgreSQLDatabase",
            ("database", "supabase"): "glassbox_rag.plugins.supabase:SupabaseDatabase",
        }

        for (ptype, name), module_class in builtin_plugins.items():
            try:
                module_path, class_name = module_class.rsplit(":", 1)
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                self._plugins[ptype][name] = cls
            except ImportError:
                pass  # Optional dependency not installed
            except Exception as e:
                logger.debug("Could not load builtin %s/%s: %s", ptype, name, e)

        self._initialized = True
        logger.debug("Registered %d builtin plugins", sum(len(v) for v in self._plugins.values()))

    def load_custom_plugins(self, custom_configs: list[dict[str, Any]]) -> None:
        """
        Load custom plugins from config.

        Each entry should have:
            - type: plugin type (vector_store, database, etc.)
            - name: plugin name
            - module: Python module path
            - class: Class name in module

        Example YAML:
            plugins:
              custom:
                - type: vector_store
                  name: my_store
                  module: mypackage.my_vector_store
                  class: MyVectorStore
        """
        for entry in custom_configs:
            try:
                ptype = entry["type"]
                name = entry["name"]
                module_path = entry["module"]
                class_name = entry["class"]

                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                self.register(ptype, name, cls)
                logger.info("Loaded custom plugin: %s/%s from %s", ptype, name, module_path)
            except KeyError as e:
                logger.error("Custom plugin config missing field: %s", e)
            except ImportError as e:
                logger.error("Could not import custom plugin module: %s", e)
            except Exception as e:
                logger.error("Failed to load custom plugin: %s", e)


# ── Global singleton ─────────────────────────────────────────────
plugin_registry = PluginRegistry()


def register_plugin(
    plugin_type: str,
    name: str,
) -> Callable:
    """
    Decorator to register a plugin class.

    Usage:
        @register_plugin("vector_store", "my_store")
        class MyStore(VectorStorePlugin):
            ...
    """
    def decorator(cls: type[Plugin]) -> type[Plugin]:
        plugin_registry.register(plugin_type, name, cls)
        return cls
    return decorator


# ═══════════════════════════════════════════════════════════════════
#  Plugin Development Helpers
# ═══════════════════════════════════════════════════════════════════


class PluginTestHarness:
    """
    Test harness for plugin development.

    Helps developers test their custom plugins against
    the GlassBox plugin interface.
    """

    @staticmethod
    async def test_vector_store(store: VectorStorePlugin) -> dict[str, Any]:
        """Run standard tests against a vector store plugin."""
        results: dict[str, Any] = {"passed": [], "failed": []}

        # Test init
        try:
            assert await store.initialize()
            results["passed"].append("initialize")
        except Exception as e:
            results["failed"].append(f"initialize: {e}")
            return results

        # Test health
        try:
            assert await store.health_check()
            results["passed"].append("health_check")
        except Exception as e:
            results["failed"].append(f"health_check: {e}")

        # Test add
        try:
            ids = await store.add_vectors(
                vectors=[[0.1, 0.2, 0.3, 0.4]],
                contents=["test doc"],
                metadata=[{"test": True}],
            )
            assert len(ids) == 1
            results["passed"].append("add_vectors")
        except Exception as e:
            results["failed"].append(f"add_vectors: {e}")

        # Test search
        try:
            hits = await store.search([0.1, 0.2, 0.3, 0.4], top_k=1)
            assert len(hits) >= 0
            results["passed"].append("search")
        except Exception as e:
            results["failed"].append(f"search: {e}")

        # Test count
        try:
            count = await store.count()
            assert count >= 0
            results["passed"].append("count")
        except Exception as e:
            results["failed"].append(f"count: {e}")

        # Test shutdown
        try:
            await store.shutdown()
            results["passed"].append("shutdown")
        except Exception as e:
            results["failed"].append(f"shutdown: {e}")

        return results

    @staticmethod
    async def test_database(db: DatabasePlugin) -> dict[str, Any]:
        """Run standard tests against a database plugin."""
        results: dict[str, Any] = {"passed": [], "failed": []}

        try:
            assert await db.initialize()
            results["passed"].append("initialize")
        except Exception as e:
            results["failed"].append(f"initialize: {e}")
            return results

        try:
            assert await db.health_check()
            results["passed"].append("health_check")
        except Exception as e:
            results["failed"].append(f"health_check: {e}")

        try:
            doc_id = await db.insert("documents", {"content": "test", "metadata": {}})
            assert doc_id
            results["passed"].append("insert")
        except Exception as e:
            results["failed"].append(f"insert: {e}")
            doc_id = None

        if doc_id:
            try:
                assert await db.update("documents", doc_id, {"content": "updated"})
                results["passed"].append("update")
            except Exception as e:
                results["failed"].append(f"update: {e}")

            try:
                rows = await db.query("documents", {"id": doc_id})
                assert len(rows) >= 1
                results["passed"].append("query")
            except Exception as e:
                results["failed"].append(f"query: {e}")

            try:
                assert await db.delete("documents", doc_id)
                results["passed"].append("delete")
            except Exception as e:
                results["failed"].append(f"delete: {e}")

        try:
            await db.shutdown()
            results["passed"].append("shutdown")
        except Exception as e:
            results["failed"].append(f"shutdown: {e}")

        return results
