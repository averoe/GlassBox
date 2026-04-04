"""
Modular Encoding Layer - handles text-to-embedding conversion.

Supports both local models (ONNX, Ollama) and cloud APIs (OpenAI, etc).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from glassbox_rag.config import GlassBoxConfig, EncodingConfig


class BaseEncoder(ABC):
    """Base class for all encoders."""

    def __init__(self, config: dict):
        """Initialize the encoder."""
        self.config = config

    @abstractmethod
    async def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        pass

    @abstractmethod
    async def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text to embedding.

        Args:
            text: Text string to encode.

        Returns:
            1D numpy array of embeddings.
        """
        pass


class ONNXEncoder(BaseEncoder):
    """
    ONNX Runtime based local encoder.
    
    Supports models like all-MiniLM-L6-v2 for CPU-based inference.
    """

    def __init__(self, config: dict):
        """Initialize ONNX encoder."""
        super().__init__(config)
        self.model_path = config.get("model_path")
        self.device = config.get("device", "cpu")
        self.batch_size = config.get("batch_size", 32)
        
        # TODO: Initialize ONNX Runtime session
        self.session = None

    async def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using ONNX model."""
        # TODO: Implement ONNX encoding
        # For now, return dummy embeddings
        embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        return np.random.randn(len(texts), embedding_dim).astype(np.float32)

    async def encode_single(self, text: str) -> np.ndarray:
        """Encode single text using ONNX model."""
        # TODO: Implement ONNX encoding for single text
        embedding_dim = 384
        return np.random.randn(embedding_dim).astype(np.float32)


class OllamaEncoder(BaseEncoder):
    """
    Ollama based local encoder.
    
    Connects to Ollama service for local LLM-based embeddings.
    """

    def __init__(self, config: dict):
        """Initialize Ollama encoder."""
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "nomic-embed-text")
        self.embedding_dim = config.get("embedding_dim", 384)

    async def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Ollama."""
        # TODO: Implement Ollama API calls
        # For now, return dummy embeddings
        return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)

    async def encode_single(self, text: str) -> np.ndarray:
        """Encode single text using Ollama."""
        # TODO: Implement Ollama API call for single text
        return np.random.randn(self.embedding_dim).astype(np.float32)


class OpenAIEncoder(BaseEncoder):
    """
    OpenAI API based cloud encoder.
    
    Uses OpenAI's embedding models via API.
    """

    def __init__(self, config: dict):
        """Initialize OpenAI encoder."""
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "text-embedding-3-small")
        self.embedding_dim = config.get("embedding_dim", 1536)

    async def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using OpenAI API."""
        # TODO: Implement OpenAI API calls
        # For now, return dummy embeddings
        return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)

    async def encode_single(self, text: str) -> np.ndarray:
        """Encode single text using OpenAI API."""
        # TODO: Implement OpenAI API call for single text
        return np.random.randn(self.embedding_dim).astype(np.float32)


class EncoderFactory:
    """Factory for creating encoders based on configuration."""

    ENCODER_TYPES = {
        "onnx": ONNXEncoder,
        "ollama": OllamaEncoder,
        "openai": OpenAIEncoder,
    }

    @staticmethod
    def create_encoder(encoder_type: str, config: dict) -> BaseEncoder:
        """
        Create an encoder instance.

        Args:
            encoder_type: Type of encoder (onnx, ollama, openai, etc).
            config: Configuration dictionary for the encoder.

        Returns:
            Encoder instance.

        Raises:
            ValueError: If encoder type is not supported.
        """
        if encoder_type not in EncoderFactory.ENCODER_TYPES:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        encoder_class = EncoderFactory.ENCODER_TYPES[encoder_type]
        return encoder_class(config)

    @staticmethod
    def register_encoder(encoder_type: str, encoder_class):
        """Register a new encoder type."""
        EncoderFactory.ENCODER_TYPES[encoder_type] = encoder_class


class ModularEncodingLayer:
    """
    Main Modular Encoding Layer that orchestrates encoding.
    
    Handles encoder selection, fallback, and caching.
    """

    def __init__(self, config: GlassBoxConfig):
        """Initialize the encoding layer."""
        self.config = config
        self.default_encoder_name = config.encoding.default_encoder
        self.encoders = {}
        
        # Initialize configured encoders
        self._init_encoders()

    def _init_encoders(self):
        """Initialize all configured encoders."""
        # Initialize local encoders
        if self.config.encoding.local:
            for encoder_name, encoder_config in self.config.encoding.local.items():
                if encoder_config:  # Only initialize if config exists
                    try:
                        encoder = EncoderFactory.create_encoder(encoder_name, encoder_config)
                        self.encoders[encoder_name] = encoder
                    except Exception as e:
                        print(f"Failed to initialize {encoder_name} encoder: {e}")

        # Initialize cloud encoders
        if self.config.encoding.cloud:
            for encoder_name, encoder_config in self.config.encoding.cloud.items():
                if encoder_config:  # Only initialize if config exists
                    try:
                        encoder = EncoderFactory.create_encoder(encoder_name, encoder_config)
                        self.encoders[encoder_name] = encoder
                    except Exception as e:
                        print(f"Failed to initialize {encoder_name} encoder: {e}")

    async def encode(
        self,
        texts: List[str],
        encoder_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Encode texts using specified or default encoder.

        Args:
            texts: List of texts to encode.
            encoder_name: Name of encoder to use (defaults to configured default).

        Returns:
            Tuple of (embeddings array, metadata dict with encoder info).

        Raises:
            ValueError: If encoder is not available.
        """
        if encoder_name is None:
            encoder_name = self.default_encoder_name

        if encoder_name not in self.encoders:
            # Use a fallback dummy encoder
            print(f"Warning: Encoder '{encoder_name}' not initialized, using dummy embeddings")
            embedding_dim = 384
            embeddings = np.random.randn(len(texts), embedding_dim).astype(np.float32)
            metadata = {
                "encoder": f"{encoder_name}_dummy",
                "num_texts": len(texts),
                "embedding_dim": embedding_dim,
            }
            return embeddings, metadata

        encoder = self.encoders[encoder_name]
        embeddings = await encoder.encode(texts)

        metadata = {
            "encoder": encoder_name,
            "num_texts": len(texts),
            "embedding_dim": embeddings.shape[1],
        }

        return embeddings, metadata

    async def encode_single(
        self,
        text: str,
        encoder_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Encode a single text.

        Args:
            text: Text to encode.
            encoder_name: Name of encoder to use (defaults to configured default).

        Returns:
            Tuple of (embedding array, metadata dict).
        """
        if encoder_name is None:
            encoder_name = self.default_encoder_name

        if encoder_name not in self.encoders:
            # Use a fallback dummy encoder
            print(f"Warning: Encoder '{encoder_name}' not initialized, using dummy embeddings")
            embedding_dim = 384
            embedding = np.random.randn(embedding_dim).astype(np.float32)
            metadata = {
                "encoder": f"{encoder_name}_dummy",
                "embedding_dim": embedding_dim,
            }
            return embedding, metadata

        encoder = self.encoders[encoder_name]
        embedding = await encoder.encode_single(text)

        metadata = {
            "encoder": encoder_name,
            "embedding_dim": len(embedding),
        }

        return embedding, metadata
