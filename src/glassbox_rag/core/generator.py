"""
First-class LLM Generation Step — retrieve + generate in one pipeline.

Provides both synchronous and streaming generation modes, with
automatic context window management and token budgeting.

Supports OpenAI, Ollama, and generic HTTP-based LLM backends.

Usage:
    generator = LLMGenerator(config)
    await generator.initialize()

    # Full response
    response = await generator.generate(query, documents)

    # Streaming
    async for chunk in generator.stream(query, documents):
        print(chunk, end="")
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Type

from glassbox_rag.core.tokens import TokenCounter, TokenBudget
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Result of a generation request."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata,
        }


@dataclass
class StreamEvent:
    """Structured streaming event for retrieve + generate."""

    type: str  # "metadata", "token", "done"
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """Configuration for a generation request."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    system_prompt: str = (
        "You are a helpful assistant. Answer questions using the provided context. "
        "If you cannot find the answer in the context, say so clearly."
    )


# ── Base LLM Backend ────────────────────────────────────────────

class BaseLLMBackend(ABC):
    """Base class for LLM backends."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend (validate keys, etc)."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> GenerationResult:
        """Generate a complete response."""

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> AsyncIterator[str]:
        """Stream response tokens."""
        yield ""  # pragma: no cover

    async def shutdown(self) -> None:
        """Cleanup hook."""


# ── OpenAI Backend ──────────────────────────────────────────────

class OpenAIBackend(BaseLLMBackend):
    """OpenAI API backend (GPT-4o, etc.)."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "gpt-4o-mini")
        self._client = None

    async def initialize(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        if not self.api_key:
            raise ValueError("OpenAI API key required for LLM generation")

        self._client = AsyncOpenAI(api_key=self.api_key)
        self._initialized = True
        logger.info("OpenAI LLM backend initialized: model=%s", self.model)

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> GenerationResult:
        if not self._client:
            raise RuntimeError("OpenAI backend not initialized")

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        choice = response.choices[0]
        usage = response.usage

        return GenerationResult(
            text=choice.message.content or "",
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason or "",
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> AsyncIterator[str]:
        if not self._client:
            raise RuntimeError("OpenAI backend not initialized")

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
        )

        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content


# ── Ollama Backend ──────────────────────────────────────────────

class OllamaBackend(BaseLLMBackend):
    """Ollama local LLM backend."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "llama3")

    async def initialize(self) -> None:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/version")
                resp.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Cannot reach Ollama at {self.base_url}: {e}")
        self._initialized = True
        logger.info("Ollama LLM backend initialized: model=%s", self.model)

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> GenerationResult:
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "top_p": top_p,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return GenerationResult(
            text=data.get("message", {}).get("content", ""),
            model=self.model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            total_tokens=(
                data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            ),
            finish_reason=data.get("done_reason", "stop"),
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> AsyncIterator[str]:
        import httpx
        import json as json_module

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "top_p": top_p,
                    },
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line.strip():
                        data = json_module.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if data.get("done"):
                            break


# ── LLM Backend Factory ────────────────────────────────────────

_LLM_BACKENDS: dict[str, type[BaseLLMBackend]] = {
    "openai": OpenAIBackend,
    "ollama": OllamaBackend,
}


def register_llm_backend(name: str, cls: type[BaseLLMBackend]) -> None:
    """Register a custom LLM backend."""
    _LLM_BACKENDS[name] = cls


# ── Main Generator ──────────────────────────────────────────────

class LLMGenerator:
    """
    First-class LLM generation step for the RAG pipeline.

    Provides retrieve-then-generate with automatic context window
    management, token budgeting, and streaming support.
    """

    def __init__(
        self,
        backend_type: str = "openai",
        backend_config: dict[str, Any] | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self.generation_config = generation_config or GenerationConfig()
        self._backend_type = backend_type
        self._backend_config = backend_config or {}

        if backend_type not in _LLM_BACKENDS:
            raise ValueError(
                f"Unknown LLM backend '{backend_type}'. "
                f"Available: {list(_LLM_BACKENDS.keys())}"
            )

        self._backend: BaseLLMBackend = _LLM_BACKENDS[backend_type](self._backend_config)
        self.token_counter = TokenCounter(model=self.generation_config.model)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the LLM backend."""
        await self._backend.initialize()
        self._initialized = True

    async def generate(
        self,
        query: str,
        context_documents: list[str] | None = None,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """
        Generate a response using retrieved context.

        Args:
            query: User query.
            context_documents: List of context text passages.
            system_prompt: Override system prompt.
            temperature: Override temperature.
            max_tokens: Override max tokens.

        Returns:
            GenerationResult with full response and token usage.
        """
        if not self._initialized:
            raise RuntimeError("LLM generator not initialized")

        cfg = self.generation_config
        sys_prompt = system_prompt or cfg.system_prompt
        temp = temperature if temperature is not None else cfg.temperature
        max_tok = max_tokens or cfg.max_tokens

        # Build messages with context window budgeting
        messages = self._build_messages(query, context_documents or [], sys_prompt)

        result = await self._backend.generate(
            messages, temperature=temp, max_tokens=max_tok, top_p=cfg.top_p,
        )

        # Add budget info to metadata
        budget = self.token_counter.create_budget(
            system_prompt=sys_prompt, query=query, reserved_output=max_tok,
        )
        result.metadata["token_budget"] = budget.to_dict()

        return result

    async def stream(
        self,
        query: str,
        context_documents: list[str] | None = None,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream a response as structured StreamEvent objects.

        Yields:
            StreamEvent(type="token", content=...) for each token.
            StreamEvent(type="done", metadata=...) as the final event.

        Usage:
            async for event in generator.stream("What is RAG?", docs):
                if event.type == "token":
                    print(event.content, end="", flush=True)
                elif event.type == "done":
                    print(f"\nFinished: {event.metadata}")
        """
        if not self._initialized:
            raise RuntimeError("LLM generator not initialized")

        cfg = self.generation_config
        sys_prompt = system_prompt or cfg.system_prompt
        temp = temperature if temperature is not None else cfg.temperature
        max_tok = max_tokens or cfg.max_tokens

        messages = self._build_messages(query, context_documents or [], sys_prompt)

        token_count = 0
        async for token in self._backend.stream(
            messages, temperature=temp, max_tokens=max_tok, top_p=cfg.top_p,
        ):
            token_count += 1
            yield StreamEvent(type="token", content=token)

        yield StreamEvent(
            type="done",
            metadata={"tokens_streamed": token_count, "model": self._backend_type},
        )

    def _build_messages(
        self,
        query: str,
        context_documents: list[str],
        system_prompt: str,
    ) -> list[dict[str, str]]:
        """
        Build chat messages with automatic context truncation.

        Uses TokenCounter to ensure the prompt fits within the
        model's context window.
        """
        budget = self.token_counter.create_budget(
            system_prompt=system_prompt,
            query=query,
            reserved_output=self.generation_config.max_tokens,
        )

        # Truncate context documents to fit
        if context_documents:
            context_documents = self.token_counter.truncate_to_budget(
                context_documents, budget,
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        if context_documents:
            context_text = "\n\n---\n\n".join(context_documents)
            messages.append({
                "role": "user",
                "content": (
                    f"Context:\n{context_text}\n\n"
                    f"Question: {query}\n\n"
                    "Answer based on the context above."
                ),
            })
        else:
            messages.append({"role": "user", "content": query})

        return messages

    async def shutdown(self) -> None:
        """Shutdown the LLM backend."""
        await self._backend.shutdown()
        self._initialized = False
