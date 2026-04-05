"""Base classes for all plugins.

Defines contracts for vector stores, databases, and custom extensions.
The EmbedderPlugin has been unified into core/encoder.py's BaseEncoder.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class Plugin(ABC):
    """Base class for all plugins."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the plugin.

        Returns:
            True if initialization successful, False otherwise.
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the plugin and release resources."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the plugin is healthy and operational."""


class VectorStorePlugin(Plugin):
    """Base class for vector store plugins."""

    @abstractmethod
    async def add_vectors(
        self,
        vectors: List[List[float]],
        ids: Optional[List[str]] = None,
        contents: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Add vectors to the store.

        Args:
            vectors: List of embedding vectors.
            ids: Optional IDs for each vector.
            contents: Optional text content for each vector.
            metadata: Optional metadata for each vector.

        Returns:
            List of vector IDs.
        """

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of (id, score, content, metadata) tuples.
        """

    @abstractmethod
    async def get_vector(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Get a vector and its metadata by ID."""

    @abstractmethod
    async def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID."""

    @abstractmethod
    async def count(self) -> int:
        """Get total number of vectors in the store."""


class DatabasePlugin(Plugin):
    """Base class for database plugins."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the database."""

    @abstractmethod
    async def insert(self, table: str, data: Dict[str, Any]) -> str:
        """
        Insert a record.

        Args:
            table: Table name.
            data: Record data.

        Returns:
            Record ID.
        """

    @abstractmethod
    async def update(self, table: str, record_id: str, data: Dict[str, Any]) -> bool:
        """Update a record."""

    @abstractmethod
    async def delete(self, table: str, record_id: str) -> bool:
        """Delete a record."""

    @abstractmethod
    async def query(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query records."""

    @abstractmethod
    async def search_text(
        self,
        terms: List[str],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Full-text search across documents."""

    @abstractmethod
    async def ensure_tables(self) -> None:
        """Create required tables if they don't exist."""
