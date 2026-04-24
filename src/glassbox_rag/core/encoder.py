"""
Modular Encoding Layer — handles text-to-embedding conversion.

Unified encoder hierarchy: all encoders implement BaseEncoder.
Real implementations for OpenAI, Ollama, and ONNX — no dummy fallbacks.
"""

from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Tuple, Type

import numpy as np

from glassbox_rag.config import GlassBoxConfig
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class EncoderError(Exception):
    """Raised when an encoding operation fails."""


class EncoderNotAvailableError(EncoderError):
    """Raised when the requested encoder is not installed or configured."""


class BaseEncoder(ABC):
    """Base class for all encoders."""

    def __init__(self, config: dict):
        self.config = config
        self._initialized = False

    @abstractmethod
    async def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            numpy array of shape (len(texts), embedding_dim).

        Raises:
            EncoderError: If encoding fails.
        """

    @abstractmethod
    async def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text to embedding.

        Returns:
            1D numpy array of embeddings.
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension for this encoder."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the encoder (load models, validate keys, etc)."""

    async def shutdown(self) -> None:
        """Optional cleanup hook."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} dim={self.embedding_dim}>"


# ═══════════════════════════════════════════════════════════════════
#  OpenAI Encoder
# ═══════════════════════════════════════════════════════════════════

class OpenAIEncoder(BaseEncoder):
    """
    OpenAI API based cloud encoder.

    Requires: pip install openai
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key: str | None = config.get("api_key")
        self.model: str = config.get("model", "text-embedding-3-small")
        self._embedding_dim: int = config.get("embedding_dim", 1536)
        self._client = None

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    async def initialize(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise EncoderNotAvailableError(
                "openai package not installed. Run: pip install glassbox-rag[embeddings]"
            )

        if not self.api_key:
            raise EncoderError("OpenAI API key not provided. Set OPENAI_API_KEY env var.")

        self._client = AsyncOpenAI(api_key=self.api_key)
        self._initialized = True
        logger.info("Initialized OpenAI encoder: model=%s, dim=%d", self.model, self._embedding_dim)

    async def encode(self, texts: list[str]) -> np.ndarray:
        if not self._initialized or self._client is None:
            raise EncoderError("OpenAI encoder not initialized. Call initialize() first.")

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dim)

        try:
            # Process in batches of 2048 (OpenAI limit)
            all_embeddings: list[list[float]] = []
            batch_size = 2048

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = await self._client.embeddings.create(
                    input=batch,
                    model=self.model,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            return np.array(all_embeddings, dtype=np.float32)

        except Exception as e:
            raise EncoderError(f"OpenAI encoding failed: {e}") from e

    async def encode_single(self, text: str) -> np.ndarray:
        result = await self.encode([text])
        return result[0]


# ═══════════════════════════════════════════════════════════════════
#  Ollama Encoder
# ═══════════════════════════════════════════════════════════════════

class OllamaEncoder(BaseEncoder):
    """
    Ollama based local encoder.

    Requires: pip install ollama  OR  uses HTTP API directly via httpx.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url: str = config.get("base_url", "http://localhost:11434")
        self.model: str = config.get("model", "nomic-embed-text")
        self._embedding_dim: int = config.get("embedding_dim", 768)
        self._client = None

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    async def initialize(self) -> None:
        import httpx

        # Verify Ollama is reachable
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/version")
                resp.raise_for_status()
        except Exception as e:
            raise EncoderNotAvailableError(
                f"Cannot reach Ollama at {self.base_url}: {e}"
            ) from e

        self._initialized = True
        logger.info(
            "Initialized Ollama encoder: url=%s, model=%s",
            self.base_url,
            self.model,
        )

    async def encode(self, texts: list[str]) -> np.ndarray:
        if not self._initialized:
            raise EncoderError("Ollama encoder not initialized. Call initialize() first.")

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dim)

        import httpx

        embeddings: list[list[float]] = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            for text in texts:
                try:
                    resp = await client.post(
                        f"{self.base_url}/api/embed",
                        json={"model": self.model, "input": text},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    emb = data.get("embeddings", [data.get("embedding", [])])
                    if isinstance(emb[0], list):
                        embeddings.append(emb[0])
                    else:
                        embeddings.append(emb)
                except Exception as e:
                    raise EncoderError(f"Ollama encoding failed for text: {e}") from e

        result = np.array(embeddings, dtype=np.float32)
        # Auto-detect dimension from first response
        if result.shape[1] != self._embedding_dim:
            self._embedding_dim = result.shape[1]

        return result

    async def encode_single(self, text: str) -> np.ndarray:
        result = await self.encode([text])
        return result[0]


# ═══════════════════════════════════════════════════════════════════
#  ONNX Encoder
# ═══════════════════════════════════════════════════════════════════

class ONNXEncoder(BaseEncoder):
    """
    ONNX Runtime based local encoder.

    Requires: pip install onnxruntime tokenizers
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.model_path: str | None = config.get("model_path")
        self.device: str = config.get("device", "cpu")
        self.batch_size: int = config.get("batch_size", 32)
        self._embedding_dim: int = config.get("embedding_dim", 384)
        self._session = None
        self._tokenizer = None

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    async def initialize(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise EncoderNotAvailableError(
                "onnxruntime not installed. Run: pip install glassbox-rag[embeddings]"
            )

        if not self.model_path:
            raise EncoderError("ONNX model_path not provided in config.")

        import os
        model_file = os.path.join(self.model_path, "model.onnx")
        if not os.path.exists(model_file):
            raise EncoderError(f"ONNX model not found at {model_file}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]

        try:
            self._session = ort.InferenceSession(model_file, providers=providers)
        except Exception as e:
            raise EncoderError(f"Failed to load ONNX model: {e}") from e

        # Load tokenizer if available
        tokenizer_path = os.path.join(self.model_path, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            try:
                from tokenizers import Tokenizer
                self._tokenizer = Tokenizer.from_file(tokenizer_path)
            except ImportError:
                logger.warning("tokenizers package not installed, using basic tokenization")
        else:
            logger.warning("No tokenizer.json found at %s", self.model_path)

        self._initialized = True
        logger.info(
            "Initialized ONNX encoder: path=%s, device=%s",
            self.model_path,
            self.device,
        )

    async def encode(self, texts: list[str]) -> np.ndarray:
        if not self._initialized or self._session is None:
            raise EncoderError("ONNX encoder not initialized. Call initialize() first.")

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dim)

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._encode_batch(batch)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Synchronous batch encoding via ONNX Runtime."""
        if self._tokenizer is not None:
            encoded = self._tokenizer.encode_batch(texts)
            max_len = min(max(len(e.ids) for e in encoded), 512)

            input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
            attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)

            for j, enc in enumerate(encoded):
                length = min(len(enc.ids), max_len)
                input_ids[j, :length] = enc.ids[:length]
                attention_mask[j, :length] = 1
        else:
            # Fallback: simple whitespace tokenization (very basic)
            max_len = 128
            input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
            attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)
            for j, text in enumerate(texts):
                tokens = text.split()[:max_len]
                for k, _ in enumerate(tokens):
                    input_ids[j, k] = hash(_) % 30000
                    attention_mask[j, k] = 1

        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Some models also need token_type_ids
        input_names = [inp.name for inp in self._session.get_inputs()]
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._session.run(None, feed)
        # Mean pooling over token embeddings (first output)
        token_embeddings = outputs[0]  # (batch, seq, dim)
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.sum(mask_expanded, axis=1).clip(min=1e-9)
        embeddings = summed / counts

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
        embeddings = embeddings / norms

        self._embedding_dim = embeddings.shape[1]
        return embeddings.astype(np.float32)

    async def encode_single(self, text: str) -> np.ndarray:
        result = await self.encode([text])
        return result[0]


# ═══════════════════════════════════════════════════════════════════
#  Hugging Face Inference API Encoder
# ═══════════════════════════════════════════════════════════════════

class HuggingFaceEncoder(BaseEncoder):
    """
    Hugging Face Inference API based encoder.

    Requires: pip install aiohttp
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.api_url = config.get("api_url", f"https://api-inference.huggingface.co/models/{self.model}")
        self._embedding_dim = config.get("embedding_dim", 384)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    async def initialize(self) -> None:
        if not self.api_key:
            raise EncoderError("Hugging Face API key not provided. Set HF_API_KEY env var.")

        # Test connection
        import aiohttp
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    raise EncoderError(f"Hugging Face API error: {response.status}")
        
        self._initialized = True
        logger.info("Initialized Hugging Face encoder: model=%s, dim=%d", self.model, self._embedding_dim)

    async def encode(self, texts: list[str]) -> np.ndarray:
        if not self._initialized:
            raise EncoderError("Hugging Face encoder not initialized. Call initialize() first.")

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dim)

        import aiohttp

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"inputs": texts, "options": {"wait_for_model": True}}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise EncoderError(f"Hugging Face API error {response.status}: {error_text}")

                result = await response.json()

                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        embeddings = result
                    elif isinstance(result[0], dict) and "embedding" in result[0]:
                        embeddings = [item["embedding"] for item in result]
                    else:
                        raise EncoderError(f"Unexpected API response format: {type(result)}")
                else:
                    raise EncoderError(f"Unexpected API response format: {type(result)}")

        return np.array(embeddings, dtype=np.float32)

    async def encode_single(self, text: str) -> np.ndarray:
        result = await self.encode([text])
        return result[0]


# ═══════════════════════════════════════════════════════════════════
#  Cohere Encoder
# ═══════════════════════════════════════════════════════════════════

class CohereEncoder(BaseEncoder):
    """
    Cohere API based encoder.

    Requires: pip install cohere
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "embed-english-v3.0")
        self._embedding_dim = config.get("embedding_dim", 1024)
        self._client = None

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    async def initialize(self) -> None:
        try:
            import cohere
        except ImportError:
            raise EncoderNotAvailableError(
                "cohere package not installed. Run: pip install glassbox-rag[embeddings]"
            )

        if not self.api_key:
            raise EncoderError("Cohere API key not provided. Set COHERE_API_KEY env var.")

        self._client = cohere.Client(api_key=self.api_key)

        # Test the client — wrap synchronous call in executor to avoid blocking event loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: self._client.embed(texts=["test"], model=self.model)
        )

        self._initialized = True
        logger.info("Initialized Cohere encoder: model=%s, dim=%d", self.model, self._embedding_dim)

    async def encode(self, texts: list[str]) -> np.ndarray:
        if not self._initialized or self._client is None:
            raise EncoderError("Cohere encoder not initialized. Call initialize() first.")

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dim)

        # Cohere embed is synchronous, so we run it in a thread
        def _embed_sync():
            response = self._client.embed(texts=texts, model=self.model)
            return response.embeddings

        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, _embed_sync)

        return np.array(embeddings, dtype=np.float32)

    async def encode_single(self, text: str) -> np.ndarray:
        result = await self.encode([text])
        return result[0]


# ═══════════════════════════════════════════════════════════════════
#  Google Generative AI Encoder
# ═══════════════════════════════════════════════════════════════════

class GoogleEncoder(BaseEncoder):
    """
    Google Generative AI embeddings encoder.

    Requires: pip install google-generativeai
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "models/text-embedding-004")
        self._embedding_dim = config.get("embedding_dim", 768)
        self._client = None

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    async def initialize(self) -> None:
        try:
            import google.generativeai as genai
        except ImportError:
            raise EncoderNotAvailableError(
                "google-generativeai package not installed. Run: pip install google-generativeai"
            )

        if not self.api_key:
            raise EncoderError("Google API key not provided. Set GOOGLE_API_KEY env var.")

        genai.configure(api_key=self.api_key)
        self._client = genai

        # Create a reusable executor to avoid per-batch thread pool churn
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Test the client
        try:
            result = genai.embed_content(model=self.model, content="test")
            if not result or 'embedding' not in result:
                raise EncoderError("Failed to get embeddings from Google API")
        except Exception as e:
            raise EncoderError(f"Google API test failed: {e}")

        self._initialized = True
        logger.info("Initialized Google encoder: model=%s, dim=%d", self.model, self._embedding_dim)

    async def encode(self, texts: list[str]) -> np.ndarray:
        if not self._initialized or self._client is None:
            raise EncoderError("Google encoder not initialized. Call initialize() first.")

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dim)

        import google.generativeai as genai

        # Google API is synchronous, so we run it in the reusable executor
        def _embed_batch(text_batch: list[str]) -> list[list[float]]:
            embeddings = []
            for text in text_batch:
                try:
                    result = genai.embed_content(model=self.model, content=text)
                    if result and 'embedding' in result:
                        embeddings.append(result['embedding'])
                    else:
                        raise EncoderError(f"No embedding returned for text: {text[:50]}...")
                except Exception as e:
                    raise EncoderError(f"Google embedding failed for text: {e}")
            return embeddings

        # Process in smaller batches to avoid rate limits
        batch_size = 10
        all_embeddings = []

        loop = asyncio.get_running_loop()
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await loop.run_in_executor(self._executor, _embed_batch, batch)
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    async def encode_single(self, text: str) -> np.ndarray:
        result = await self.encode([text])
        return result[0]

    async def shutdown(self) -> None:
        """Shutdown the reusable executor."""
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None


# ═══════════════════════════════════════════════════════════════════
#  Encoder Factory & Registry
# ═══════════════════════════════════════════════════════════════════

class EncoderFactory:
    """Factory for creating encoders based on configuration."""

    _ENCODER_TYPES: dict[str, type[BaseEncoder]] = {
        "onnx": ONNXEncoder,
        "ollama": OllamaEncoder,
        "openai": OpenAIEncoder,
        "huggingface": HuggingFaceEncoder,
        "cohere": CohereEncoder,
        "google": GoogleEncoder,
    }

    @classmethod
    def create_encoder(cls, encoder_type: str, config: dict) -> BaseEncoder:
        """
        Create an encoder instance (not yet initialized).

        Raises:
            ValueError: If encoder type is not registered.
        """
        if encoder_type not in cls._ENCODER_TYPES:
            available = ", ".join(cls._ENCODER_TYPES.keys())
            raise ValueError(
                f"Unknown encoder type '{encoder_type}'. Available: {available}"
            )

        encoder_class = cls._ENCODER_TYPES[encoder_type]
        return encoder_class(config)

    @classmethod
    def register_encoder(cls, encoder_type: str, encoder_class: type[BaseEncoder]) -> None:
        """Register a custom encoder type."""
        cls._ENCODER_TYPES[encoder_type] = encoder_class
        logger.info("Registered custom encoder: %s", encoder_type)

    @classmethod
    def available_encoders(cls) -> list[str]:
        """List available encoder types."""
        return list(cls._ENCODER_TYPES.keys())


# ═══════════════════════════════════════════════════════════════════
#  Embedding LRU Cache
# ═══════════════════════════════════════════════════════════════════

class EmbeddingCache:
    """
    Thread-safe LRU cache for embedding results.

    Avoids redundant encoder calls for repeated text inputs.
    Keys are (text_hash, encoder_name) tuples; values are numpy arrays.
    """

    def __init__(
        self,
        max_size: int = 2048,
        max_memory_bytes: int | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        self._max_size = max_size
        self._max_memory_bytes = max_memory_bytes
        self._ttl_seconds = ttl_seconds
        self._cache: Ordereddict[str, tuple[np.ndarray, float]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._current_memory = 0

    @staticmethod
    def _key(text: str, encoder_name: str) -> str:
        """Generate a cache key from text content and encoder name."""
        h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]
        return f"{encoder_name}:{h}"

    def get(self, text: str, encoder_name: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding. Returns None on miss."""
        import time as _time
        key = self._key(text, encoder_name)
        with self._lock:
            if key in self._cache:
                embedding, timestamp = self._cache[key]
                # Check TTL
                if self._ttl_seconds is not None:
                    if _time.monotonic() - timestamp > self._ttl_seconds:
                        self._current_memory -= embedding.nbytes
                        del self._cache[key]
                        self._misses += 1
                        return None
                self._hits += 1
                self._cache.move_to_end(key)
                return embedding.copy()
            self._misses += 1
            return None

    def put(self, text: str, encoder_name: str, embedding: np.ndarray) -> None:
        """Store an embedding in the cache."""
        import time as _time
        key = self._key(text, encoder_name)
        now = _time.monotonic()
        with self._lock:
            if key in self._cache:
                old_emb, _ = self._cache[key]
                self._current_memory -= old_emb.nbytes
                self._cache.move_to_end(key)
            else:
                # Evict by count
                while len(self._cache) >= self._max_size:
                    _, (evicted_emb, _) = self._cache.popitem(last=False)
                    self._current_memory -= evicted_emb.nbytes
                # Evict by memory
                if self._max_memory_bytes is not None:
                    while (
                        self._current_memory + embedding.nbytes > self._max_memory_bytes
                        and self._cache
                    ):
                        _, (evicted_emb, _) = self._cache.popitem(last=False)
                        self._current_memory -= evicted_emb.nbytes
            emb_copy = embedding.copy()
            self._cache[key] = (emb_copy, now)
            self._current_memory += emb_copy.nbytes

    def get_batch(self, texts: list[str], encoder_name: str) -> tuple[list[int], list[np.ndarray], list[int]]:
        """
        Batch cache lookup.

        Returns:
            (hit_indices, hit_embeddings, miss_indices)
        """
        hit_indices: list[int] = []
        hit_embeddings: list[np.ndarray] = []
        miss_indices: list[int] = []
        for i, text in enumerate(texts):
            cached = self.get(text, encoder_name)
            if cached is not None:
                hit_indices.append(i)
                hit_embeddings.append(cached)
            else:
                miss_indices.append(i)
        return hit_indices, hit_embeddings, miss_indices

    def put_batch(self, texts: list[str], encoder_name: str, embeddings: np.ndarray) -> None:
        """Store a batch of embeddings."""
        for text, emb in zip(texts, embeddings):
            self.put(text, encoder_name, emb)

    def save(self, path: str | Path) -> None:
        """Persist the cache to disk."""
        import pickle
        with self._lock:
            with open(path, "wb") as f:
                pickle.dump({
                    "cache": self._cache,
                    "hits": self._hits,
                    "misses": self._misses,
                    "current_memory": self._current_memory,
                }, f)

    def load(self, path: str | Path) -> None:
        """Load the cache from disk."""
        import pickle
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return
        with self._lock:
            with open(p, "rb") as f:
                data = pickle.load(f)
                self._cache = data.get("cache", OrderedDict())
                self._hits = data.get("hits", 0)
                self._misses = data.get("misses", 0)
                self._current_memory = data.get("current_memory", 0)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._current_memory = 0

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "memory_bytes": self._current_memory,
                "max_memory_bytes": self._max_memory_bytes,
                "ttl_seconds": self._ttl_seconds,
            }


# ═══════════════════════════════════════════════════════════════════
#  Modular Encoding Layer (orchestrator)
# ═══════════════════════════════════════════════════════════════════

class ModularEncodingLayer:
    """
    Main Modular Encoding Layer that orchestrates encoding.

    Manages multiple encoders, handles initialization, and selects
    the correct encoder per request.  Never returns dummy data —
    raises EncoderNotAvailableError if no encoder is available.

    Includes an LRU cache to avoid redundant embedding API calls.
    """

    def __init__(self, config: GlassBoxConfig, cache_size: int = 2048):
        self.config = config
        self.default_encoder_name = config.encoding.default_encoder
        self.encoders: dict[str, BaseEncoder] = {}
        self.cache = EmbeddingCache(max_size=cache_size)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all configured encoders."""
        # Initialize local encoders
        if self.config.encoding.local:
            for name, enc_config in self.config.encoding.local.items():
                if enc_config:
                    await self._try_init_encoder(name, enc_config)

        # Initialize cloud encoders
        if self.config.encoding.cloud:
            cloud_dict = self.config.encoding.cloud.model_dump(exclude_none=True)
            for name, enc_config in cloud_dict.items():
                if enc_config:
                    # Unwrap SecretStr if present
                    if "api_key" in enc_config:
                        # Extract from the original pydantic object to guarantee SecretStr handling
                        model_obj = getattr(self.config.encoding.cloud, name, None)
                        if model_obj and hasattr(model_obj, "api_key") and model_obj.api_key:
                            enc_config["api_key"] = model_obj.api_key.get_secret_value()
                    await self._try_init_encoder(name, enc_config)

        self._initialized = True

        if not self.encoders:
            from glassbox_rag.config import GlassBoxConfig  # avoid circular
            raise EncoderError(
                "No encoders initialized. Configure at least one encoder "
                "under 'encoding.local' or 'encoding.cloud' in your config file."
            )
        else:
            logger.info(
                "Encoding layer ready: %d encoder(s) — %s (default: %s)",
                len(self.encoders),
                list(self.encoders.keys()),
                self.default_encoder_name,
            )

    async def _try_init_encoder(self, name: str, config: dict) -> None:
        """Try to initialize an encoder, log errors but don't crash."""
        try:
            encoder = EncoderFactory.create_encoder(name, config)
            await encoder.initialize()
            self.encoders[name] = encoder
        except EncoderNotAvailableError as e:
            logger.warning("Encoder '%s' not available: %s", name, e)
        except Exception as e:
            logger.error("Failed to initialize encoder '%s': %s", name, e)

    async def encode(
        self,
        texts: list[str],
        encoder_name: str | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Encode texts using specified or default encoder.

        Uses LRU cache to skip re-encoding previously seen texts.

        Raises:
            EncoderNotAvailableError: If no suitable encoder is available.
            EncoderError: If encoding fails.
        """
        encoder_name = encoder_name or self.default_encoder_name

        if encoder_name not in self.encoders:
            available = list(self.encoders.keys())
            raise EncoderNotAvailableError(
                f"Encoder '{encoder_name}' not available. "
                f"Available encoders: {available}"
            )

        encoder = self.encoders[encoder_name]

        # Check cache for each text; only encode cache-misses
        hit_indices, hit_embeddings, miss_indices = self.cache.get_batch(texts, encoder_name)

        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            miss_embeddings = await encoder.encode(miss_texts)
            self.cache.put_batch(miss_texts, encoder_name, miss_embeddings)
        else:
            miss_embeddings = np.empty((0, 0), dtype=np.float32)

        # Reassemble in original order
        dim = (
            hit_embeddings[0].shape[0] if hit_embeddings
            else miss_embeddings.shape[1] if len(miss_embeddings.shape) > 1
            else 0
        )
        result = np.empty((len(texts), dim), dtype=np.float32)
        for idx, emb in zip(hit_indices, hit_embeddings):
            result[idx] = emb
        miss_ptr = 0
        for idx in miss_indices:
            result[idx] = miss_embeddings[miss_ptr]
            miss_ptr += 1

        cache_stats = self.cache.stats()
        metadata = {
            "encoder": encoder_name,
            "num_texts": len(texts),
            "embedding_dim": dim,
            "cache_hits": len(hit_indices),
            "cache_misses": len(miss_indices),
            "cache_hit_rate": cache_stats["hit_rate"],
        }

        return result, metadata

    async def encode_single(
        self,
        text: str,
        encoder_name: str | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Encode a single text (with cache)."""
        encoder_name = encoder_name or self.default_encoder_name

        if encoder_name not in self.encoders:
            available = list(self.encoders.keys())
            raise EncoderNotAvailableError(
                f"Encoder '{encoder_name}' not available. "
                f"Available encoders: {available}"
            )

        # Check cache first
        cached = self.cache.get(text, encoder_name)
        if cached is not None:
            return cached, {"encoder": encoder_name, "embedding_dim": len(cached), "cache": "hit"}

        encoder = self.encoders[encoder_name]
        embedding = await encoder.encode_single(text)
        self.cache.put(text, encoder_name, embedding)

        metadata = {
            "encoder": encoder_name,
            "embedding_dim": len(embedding),
            "cache": "miss",
        }

        return embedding, metadata

    async def shutdown(self) -> None:
        """Shutdown all encoders."""
        for name, encoder in self.encoders.items():
            try:
                await encoder.shutdown()
            except Exception as e:
                logger.error("Error shutting down encoder '%s': %s", name, e)
