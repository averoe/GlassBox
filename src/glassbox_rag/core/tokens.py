"""
Token Counter & Context Window Management.

Provides token counting utilities and context window budgeting
to ensure prompts stay within model limits. Supports multiple
tokenizer backends (tiktoken, simple whitespace fallback).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


# ── Known model context windows ──────────────────────────────────

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    # Anthropic
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3.5-sonnet": 200_000,
    # Google
    "gemini-1.5-pro": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    # Cohere
    "command-r-plus": 128_000,
    "command-r": 128_000,
    # Open-source
    "llama-3-70b": 8_192,
    "llama-3-8b": 8_192,
    "mistral-7b": 32_768,
    "mixtral-8x7b": 32_768,
}


@dataclass
class TokenBudget:
    """Budget allocation for a context window."""

    total: int
    system_prompt: int = 0
    context_documents: int = 0
    query: int = 0
    reserved_output: int = 0

    @property
    def used(self) -> int:
        return self.system_prompt + self.context_documents + self.query

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.used - self.reserved_output)

    @property
    def utilization(self) -> float:
        return round(self.used / self.total, 4) if self.total else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "total": self.total,
            "system_prompt_tokens": self.system_prompt,
            "context_tokens": self.context_documents,
            "query_tokens": self.query,
            "reserved_output": self.reserved_output,
            "used": self.used,
            "remaining": self.remaining,
            "utilization": self.utilization,
        }


class TokenCounter:
    """
    Multi-backend token counter.

    Tries tiktoken first (accurate for OpenAI models), falls back
    to a regex-based word splitter (~75% of tiktoken accuracy).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        custom_context_windows: dict[str, int] | None = None,
    ) -> None:
        self.model = model
        self._custom_windows = custom_context_windows or {}
        self._encoder = None
        self._backend = "regex"  # default fallback

        try:
            import tiktoken
            self._encoder = tiktoken.encoding_for_model(model)
            self._backend = "tiktoken"
        except (ImportError, KeyError):
            # tiktoken not installed or model not recognized — use fallback
            pass

    # ── Core counting ─────────────────────────────────────────

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        if self._backend == "tiktoken" and self._encoder:
            return len(self._encoder.encode(text))
        return self._regex_count(text)

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """
        Count tokens in a chat messages list.

        Each message has ~4 overhead tokens for role/separators.
        """
        total = 0
        for msg in messages:
            total += 4  # role + separators overhead
            total += self.count(msg.get("content", ""))
            total += self.count(msg.get("role", ""))
        total += 2  # priming tokens
        return total

    def count_batch(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts."""
        return [self.count(t) for t in texts]

    # ── Context window management ─────────────────────────────

    def get_context_window(self, model: str | None = None) -> int:
        """Get context window size for a model."""
        m = model or self.model
        # Check user-provided overrides first
        if m in self._custom_windows:
            return self._custom_windows[m]
        # Try exact match, then prefix match in known models
        if m in MODEL_CONTEXT_WINDOWS:
            return MODEL_CONTEXT_WINDOWS[m]
        for key, val in MODEL_CONTEXT_WINDOWS.items():
            if m.startswith(key):
                return val
        return 4096  # safe default

    def create_budget(
        self,
        system_prompt: str = "",
        query: str = "",
        reserved_output: int = 1024,
        model: str | None = None,
    ) -> TokenBudget:
        """
        Create a token budget for a generation request.

        Returns a budget showing how many tokens are available
        for context documents.
        """
        window = self.get_context_window(model)
        return TokenBudget(
            total=window,
            system_prompt=self.count(system_prompt),
            query=self.count(query),
            reserved_output=reserved_output,
        )

    def truncate_to_budget(
        self,
        documents: list[str],
        budget: TokenBudget,
    ) -> list[str]:
        """
        Truncate a list of context documents to fit within the budget.

        Returns the longest prefix of documents that fits, with the
        last document potentially truncated.
        """
        available = budget.remaining
        result: list[str] = []
        used = 0

        for doc in documents:
            doc_tokens = self.count(doc)
            if used + doc_tokens <= available:
                result.append(doc)
                used += doc_tokens
            else:
                # Truncate last document if it partially fits
                remaining_tokens = available - used
                if remaining_tokens > 50:  # Only include if meaningful
                    truncated = self._truncate_text(doc, remaining_tokens)
                    if truncated:
                        result.append(truncated)
                        used += self.count(truncated)
                break

        budget.context_documents = used
        return result

    # ── Private helpers ───────────────────────────────────────

    @staticmethod
    def _regex_count(text: str) -> int:
        """Fallback regex tokenizer (~75% accuracy of tiktoken)."""
        # Split on whitespace and punctuation boundaries
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return len(tokens)

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens."""
        if self._backend == "tiktoken" and self._encoder:
            token_ids = self._encoder.encode(text)[:max_tokens]
            return self._encoder.decode(token_ids)
        # Regex fallback: split and rejoin
        words = text.split()
        # Rough ratio: ~1.3 tokens per word for English
        max_words = int(max_tokens / 1.3)
        return " ".join(words[:max_words])

    @property
    def backend(self) -> str:
        """Return the active tokenizer backend name."""
        return self._backend

    def to_dict(self) -> dict[str, str | int]:
        return {
            "model": self.model,
            "backend": self._backend,
            "context_window": self.get_context_window(),
        }
