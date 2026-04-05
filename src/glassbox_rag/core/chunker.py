"""
Text Chunking Module — splits documents into retrieval-ready chunks.

Provides multiple strategies: fixed-size, sentence-based, recursive,
and optional semantic chunking. Includes chunk-size monitoring.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import re

from glassbox_rag.config import ChunkingConfig
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Character count."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.text.split())

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "size": self.size,
            "word_count": self.word_count,
            "metadata": self.metadata,
        }


@dataclass
class ChunkingStats:
    """Statistics about a chunking operation."""

    total_chunks: int = 0
    total_chars: int = 0
    min_chunk_size: int = 0
    max_chunk_size: int = 0
    avg_chunk_size: float = 0.0
    strategy: str = ""

    def to_dict(self) -> Dict:
        return {
            "total_chunks": self.total_chunks,
            "total_chars": self.total_chars,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "avg_chunk_size": round(self.avg_chunk_size, 1),
            "strategy": self.strategy,
        }


class BaseChunker(ABC):
    """Base class for text chunkers."""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Input text to chunk.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List[Chunk]: Ordered list of non-empty chunks.
        """

    def chunk_with_stats(
        self,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> tuple[List[Chunk], ChunkingStats]:
        """Chunk text and return stats."""
        chunks = self.chunk(text, metadata)
        stats = self._compute_stats(chunks)
        return chunks, stats

    def _compute_stats(self, chunks: List[Chunk]) -> ChunkingStats:
        if not chunks:
            return ChunkingStats(strategy=self.__class__.__name__)

        sizes = [c.size for c in chunks]
        return ChunkingStats(
            total_chunks=len(chunks),
            total_chars=sum(sizes),
            min_chunk_size=min(sizes),
            max_chunk_size=max(sizes),
            avg_chunk_size=sum(sizes) / len(sizes),
            strategy=self.__class__.__name__,
        )


class FixedSizeChunker(BaseChunker):
    """
    Splits text into fixed-size character chunks with optional overlap.

    Simple and predictable. Good default for structured text.
    """

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []

        chunks: List[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Don't create empty chunks
            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=idx,
                        start_char=start,
                        end_char=min(end, len(text)),
                        metadata=dict(metadata) if metadata else {},
                    )
                )
                idx += 1

            start = end - self.chunk_overlap
            if start >= len(text):
                break
            # Prevent infinite loop when overlap >= chunk_size
            if end >= len(text):
                break

        return chunks


class SentenceChunker(BaseChunker):
    """
    Splits text on sentence boundaries, then groups sentences
    into chunks up to chunk_size characters.
    """

    _SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []

        sentences = self._SENTENCE_PATTERN.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_size = 0
        char_pos = 0
        chunk_start = 0
        idx = 0

        for sentence in sentences:
            sent_len = len(sentence)

            if current_size + sent_len > self.chunk_size and current_sentences:
                # Emit current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=idx,
                        start_char=chunk_start,
                        end_char=chunk_start + len(chunk_text),
                        metadata=dict(metadata) if metadata else {},
                    )
                )
                idx += 1

                # Handle overlap: keep last N characters worth of sentences
                if self.chunk_overlap > 0:
                    overlap_sentences: List[str] = []
                    overlap_size = 0
                    for s in reversed(current_sentences):
                        if overlap_size + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_size += len(s)
                        else:
                            break
                    current_sentences = overlap_sentences
                    current_size = overlap_size
                    chunk_start = char_pos - overlap_size
                else:
                    current_sentences = []
                    current_size = 0
                    chunk_start = char_pos

            current_sentences.append(sentence)
            current_size += sent_len
            char_pos += sent_len + 1  # +1 for space

        # Final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=idx,
                        start_char=chunk_start,
                        end_char=chunk_start + len(chunk_text),
                        metadata=dict(metadata) if metadata else {},
                    )
                )

        return chunks


class RecursiveChunker(BaseChunker):
    """
    Recursively splits text using a hierarchy of separators.

    Tries to split on paragraph boundaries first, then sentences,
    then words, then characters. Inspired by LangChain's
    RecursiveCharacterTextSplitter.
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.separators = config.separators

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []

        raw_chunks = self._split_recursive(text, self.separators)

        chunks: List[Chunk] = []
        char_pos = 0

        for idx, chunk_text in enumerate(raw_chunks):
            # Find actual position in original text
            pos = text.find(chunk_text, char_pos)
            if pos == -1:
                pos = char_pos

            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=idx,
                    start_char=pos,
                    end_char=pos + len(chunk_text),
                    metadata=dict(metadata) if metadata else {},
                )
            )
            char_pos = pos + len(chunk_text)

        return chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Find the best separator
        separator = separators[-1] if separators else ""
        remaining_seps = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                remaining_seps = []
                break
            if sep in text:
                separator = sep
                remaining_seps = separators[i + 1 :]
                break

        # Split by chosen separator
        if separator:
            parts = text.split(separator)
        else:
            # Character-level split as last resort
            parts = list(text)

        # Merge small parts into chunks
        merged: List[str] = []
        current_parts: List[str] = []
        current_size = 0

        for part in parts:
            part_len = len(part) + len(separator)

            if current_size + part_len > self.chunk_size and current_parts:
                merged_text = separator.join(current_parts)
                if merged_text.strip():
                    merged.append(merged_text)

                # Overlap
                if self.chunk_overlap > 0:
                    overlap_parts: List[str] = []
                    overlap_size = 0
                    for p in reversed(current_parts):
                        if overlap_size + len(p) <= self.chunk_overlap:
                            overlap_parts.insert(0, p)
                            overlap_size += len(p)
                        else:
                            break
                    current_parts = overlap_parts
                    current_size = overlap_size
                else:
                    current_parts = []
                    current_size = 0

            current_parts.append(part)
            current_size += part_len

        # Final part
        if current_parts:
            merged_text = separator.join(current_parts)
            if merged_text.strip():
                merged.append(merged_text)

        # Recursively split any chunks that are still too large
        result: List[str] = []
        for chunk_text in merged:
            if len(chunk_text) > self.chunk_size and remaining_seps:
                result.extend(self._split_recursive(chunk_text, remaining_seps))
            else:
                result.append(chunk_text)

        return result


# ═══════════════════════════════════════════════════════════════════
#  Chunk Monitor
# ═══════════════════════════════════════════════════════════════════

class ChunkSizeMonitor:
    """
    Monitors chunk sizes across ingestion operations.

    Tracks distribution, warns about outliers, and provides
    recommendations for chunk_size tuning.
    """

    def __init__(self, target_size: int = 512):
        self.target_size = target_size
        self._all_sizes: List[int] = []
        self._total_documents: int = 0

    def record(self, chunks: List[Chunk]) -> None:
        """Record chunk sizes from an operation."""
        self._total_documents += 1
        self._all_sizes.extend(c.size for c in chunks)

    def get_report(self) -> Dict:
        """Get monitoring report."""
        if not self._all_sizes:
            return {"total_documents": 0, "total_chunks": 0}

        sizes = self._all_sizes
        too_small = sum(1 for s in sizes if s < self.target_size * 0.25)
        too_large = sum(1 for s in sizes if s > self.target_size * 2)

        sorted_sizes = sorted(sizes)
        n = len(sorted_sizes)

        return {
            "total_documents": self._total_documents,
            "total_chunks": len(sizes),
            "target_size": self.target_size,
            "min_size": min(sizes),
            "max_size": max(sizes),
            "avg_size": round(sum(sizes) / len(sizes), 1),
            "median_size": sorted_sizes[n // 2],
            "too_small_count": too_small,
            "too_large_count": too_large,
            "recommendations": self._get_recommendations(too_small, too_large, sizes),
        }

    def _get_recommendations(
        self,
        too_small: int,
        too_large: int,
        sizes: List[int],
    ) -> List[str]:
        recs: List[str] = []
        total = len(sizes)

        if too_small > total * 0.3:
            recs.append(
                f"⚠ {too_small}/{total} chunks ({too_small*100//total}%) are under "
                f"25% of target size. Consider reducing chunk_size or using "
                f"sentence-based chunking."
            )
        if too_large > total * 0.1:
            recs.append(
                f"⚠ {too_large}/{total} chunks ({too_large*100//total}%) exceed "
                f"2× target size. Consider using recursive chunking strategy."
            )
        if not recs:
            recs.append("✓ Chunk size distribution looks healthy.")

        return recs


# ═══════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════

_CHUNKER_MAP = {
    "fixed": FixedSizeChunker,
    "sentence": SentenceChunker,
    "recursive": RecursiveChunker,
}


def create_chunker(config: ChunkingConfig) -> BaseChunker:
    """Create a chunker based on config strategy."""
    strategy = config.strategy

    if strategy not in _CHUNKER_MAP:
        available = ", ".join(_CHUNKER_MAP.keys())
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Available: {available}")

    return _CHUNKER_MAP[strategy](config)
