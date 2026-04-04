"""Base classes for all plugins."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Plugin(ABC):
    """Base class for all plugins."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize plugin."""
        self.config = config

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin.

        Returns:
            True if initialization successful, False otherwise.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        pass


class VectorStorePlugin(Plugin):
    """Base class for vector store plugins."""

    @abstractmethod
    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Add vectors to the store.

        Args:
            vectors: List of embedding vectors.
            metadata: Optional metadata for each vector.

        Returns:
            List of vector IDs.
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        **kwargs,
    ) -> List[tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            **kwargs: Additional search parameters.

        Returns:
            List of (id, score) tuples.
        """
        pass

    @abstractmethod
    async def get_vector(self, vector_id: str) -> Optional[List[float]]:
        """Get a vector by ID."""
        pass

    @abstractmethod
    async def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID."""
        pass


class EmbedderPlugin(Plugin):
    """Base class for embedder plugins."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        pass

    @abstractmethod
    async def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        pass


class DatabasePlugin(Plugin):
    """Base class for database plugins."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the database."""
        pass

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
        pass

    @abstractmethod
    async def update(self, table: str, record_id: str, data: Dict[str, Any]) -> bool:
        """Update a record."""
        pass

    @abstractmethod
    async def delete(self, table: str, record_id: str) -> bool:
        """Delete a record."""
        pass

    @abstractmethod
    async def query(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query records."""
        pass
