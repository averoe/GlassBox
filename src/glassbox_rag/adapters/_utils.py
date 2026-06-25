"""
Shared utilities for GlassBox RAG adapters.

Provides safe sync-to-async bridging for use in adapters that need
synchronous wrappers around async engine methods.
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine


def run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """
    Run a coroutine safely from a synchronous context.

    Handles three scenarios:
      1. No running event loop → use asyncio.run() directly.
      2. Running event loop (FastAPI, Jupyter, etc.) → offload to a
         background thread via ThreadPoolExecutor to avoid
         "RuntimeError: This event loop is already running".
      3. nest_asyncio patched loop → works transparently via case 1.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to use asyncio.run()
        return asyncio.run(coro)

    # Already inside an event loop (FastAPI, Jupyter, etc.)
    # Schedule in a background thread to avoid blocking the loop.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()
