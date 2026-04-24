"""
Query Pipeline Hooks — extensible pre/post processing system.

Allows users to register custom functions that run at specific
points in the RAG pipeline:
  - pre_retrieve: modify query before retrieval
  - post_retrieve: modify/filter results after retrieval
  - pre_generate: modify context/prompt before generation
  - post_generate: modify/log generated response

Usage:
    from glassbox_rag.core.hooks import HookManager, HookPoint

    hooks = HookManager()

    @hooks.on(HookPoint.PRE_RETRIEVE)
    async def log_query(ctx):
        print(f"Query: {ctx['query']}")
        return ctx

    @hooks.on(HookPoint.POST_RETRIEVE)
    async def filter_low_score(ctx):
        ctx['documents'] = [d for d in ctx['documents'] if d.score > 0.5]
        return ctx
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)

# Type alias for hook functions
HookFn = Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]


class HookPoint(Enum):
    """Points in the pipeline where hooks can be attached."""

    PRE_RETRIEVE = "pre_retrieve"
    POST_RETRIEVE = "post_retrieve"
    PRE_GENERATE = "pre_generate"
    POST_GENERATE = "post_generate"
    PRE_INGEST = "pre_ingest"
    POST_INGEST = "post_ingest"
    PRE_RERANK = "pre_rerank"
    POST_RERANK = "post_rerank"
    ON_ERROR = "on_error"


class HookManager:
    """
    Manages pipeline hooks for extensible query processing.

    Hooks are async callables that receive a context dict and must
    return the (possibly modified) context dict. Multiple hooks
    per point are executed in registration order.
    """

    def __init__(self) -> None:
        self._hooks: dict[HookPoint, list[tuple[int, HookFn]]] = {
            point: [] for point in HookPoint
        }

    def register(self, point: HookPoint, fn: HookFn, *, priority: int = 0) -> None:
        """
        Register a hook function at a specific pipeline point.

        Args:
            point: Where in the pipeline this hook runs.
            fn: Async function that receives and returns a context dict.
            priority: Lower values run first (default=0).
        """
        self._hooks[point].append((priority, fn))
        self._hooks[point].sort(key=lambda x: x[0])
        logger.debug(
            "Registered hook '%s' at %s (priority=%d, total: %d)",
            fn.__name__, point.value, priority, len(self._hooks[point]),
        )

    def unregister(self, point: HookPoint, fn: HookFn) -> bool:
        """Remove a registered hook. Returns True if found and removed."""
        hooks = self._hooks[point]
        for i, (_, registered_fn) in enumerate(hooks):
            if registered_fn is fn:
                hooks.pop(i)
                return True
        return False

    def on(self, point: HookPoint, *, priority: int = 0) -> Callable:
        """
        Decorator to register an async hook function.

        Usage:
            @hooks.on(HookPoint.PRE_RETRIEVE)
            async def my_hook(ctx):
                ctx['query'] = ctx['query'].lower()
                return ctx
        """
        def decorator(fn: HookFn) -> HookFn:
            self.register(point, fn, priority=priority)
            return fn
        return decorator

    async def run(self, point: HookPoint, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute all hooks for a given point sequentially.

        Each hook receives the context from the previous hook.
        If a hook raises an exception:
          - For ON_ERROR hooks: the exception is logged and swallowed.
          - For all others: the exception propagates (pipeline fails).

        Args:
            point: The pipeline point to execute hooks for.
            context: The current pipeline context dict.

        Returns:
            The (possibly modified) context dict.
        """
        hooks = self._hooks.get(point, [])
        if not hooks:
            return context

        for _, hook in hooks:
            try:
                result = await hook(context)
                if result is not None:
                    context = result
                else:
                    logger.warning(
                        "Hook '%s' at %s returned None — context unchanged",
                        hook.__name__, point.value,
                    )
            except Exception as e:
                if point == HookPoint.ON_ERROR:
                    logger.debug("Error hook '%s' raised: %s", hook.__name__, e)
                else:
                    logger.error(
                        "Hook '%s' at %s failed: %s", hook.__name__, point.value, e,
                    )
                    raise

        return context

    def clear(self, point: HookPoint | None = None) -> None:
        """Clear hooks — all points or a specific point."""
        if point:
            self._hooks[point].clear()
        else:
            for p in HookPoint:
                self._hooks[p].clear()

    def list_hooks(self) -> dict[str, list[str]]:
        """List all registered hooks by point."""
        return {
            point.value: [fn.__name__ for _, fn in fns]
            for point, fns in self._hooks.items()
            if fns
        }

    @property
    def total_hooks(self) -> int:
        return sum(len(fns) for fns in self._hooks.values())
