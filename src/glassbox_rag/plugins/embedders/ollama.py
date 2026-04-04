"""Ollama embedder plugin implementation."""

from typing import List, Dict, Any
import aiohttp

from glassbox_rag.plugins.base import EmbedderPlugin


class OllamaEmbedder(EmbedderPlugin):
    """Ollama embedder implementation for local LLM-based embeddings."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama embedder."""
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "nomic-embed-text")
        self._embedding_dim = config.get("embedding_dim", 384)
        
        self.session: aiohttp.ClientSession = None

    def initialize(self) -> bool:
        """Initialize the embedder."""
        try:
            # TODO: Test connection to Ollama service
            print(f"Initialized Ollama embedder with model {self.model} at {self.base_url}")
            return True
        except Exception as e:
            print(f"Failed to initialize Ollama embedder: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown the embedder."""
        if self.session:
            # TODO: Close session properly
            pass

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts using Ollama."""
        if not texts:
            return []

        try:
            embeddings = []
            
            # TODO: Implement actual Ollama API calls
            # async with aiohttp.ClientSession() as session:
            #     for text in texts:
            #         async with session.post(
            #             f"{self.base_url}/api/embeddings",
            #             json={"model": self.model, "prompt": text}
            #         ) as resp:
            #             if resp.status == 200:
            #                 data = await resp.json()
            #                 embeddings.append(data["embedding"])
            
            # For now, return dummy embeddings
            import numpy as np
            embeddings = [np.random.randn(self._embedding_dim).tolist() for _ in texts]
            
            return embeddings
        except Exception as e:
            print(f"Error embedding texts: {e}")
            return []

    async def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim
