"""Unit tests for pipeline hooks (core/hooks.py).

Covers registration, ordering by priority, ON_ERROR isolation,
None return warning, timeout enforcement, and multi-hook chains.
"""

import asyncio
import pytest
from glassbox_rag.core.hooks import HookManager, HookPoint, HookTimeoutError


class TestHookRegistration:
    def test_register_and_list(self):
        hm = HookManager()

        async def my_hook(ctx):
            return ctx

        hm.register(HookPoint.PRE_RETRIEVE, my_hook)
        hooks = hm.list_hooks()
        assert "pre_retrieve" in hooks
        assert "my_hook" in hooks["pre_retrieve"]

    def test_unregister(self):
        hm = HookManager()

        async def my_hook(ctx):
            return ctx

        hm.register(HookPoint.PRE_RETRIEVE, my_hook)
        assert hm.total_hooks == 1

        removed = hm.unregister(HookPoint.PRE_RETRIEVE, my_hook)
        assert removed is True
        assert hm.total_hooks == 0

    def test_unregister_nonexistent(self):
        hm = HookManager()

        async def my_hook(ctx):
            return ctx

        removed = hm.unregister(HookPoint.PRE_RETRIEVE, my_hook)
        assert removed is False

    def test_decorator_registration(self):
        hm = HookManager()

        @hm.on(HookPoint.PRE_RETRIEVE)
        async def my_hook(ctx):
            return ctx

        assert hm.total_hooks == 1

    def test_clear_specific_point(self):
        hm = HookManager()

        async def h1(ctx):
            return ctx

        async def h2(ctx):
            return ctx

        hm.register(HookPoint.PRE_RETRIEVE, h1)
        hm.register(HookPoint.POST_RETRIEVE, h2)
        assert hm.total_hooks == 2

        hm.clear(HookPoint.PRE_RETRIEVE)
        assert hm.total_hooks == 1

    def test_clear_all(self):
        hm = HookManager()

        async def h1(ctx):
            return ctx

        hm.register(HookPoint.PRE_RETRIEVE, h1)
        hm.register(HookPoint.POST_RETRIEVE, h1)
        hm.clear()
        assert hm.total_hooks == 0


class TestHookOrdering:
    @pytest.mark.asyncio
    async def test_priority_order(self):
        """Lower priority values run first."""
        hm = HookManager()
        execution_order = []

        async def hook_a(ctx):
            execution_order.append("a")
            return ctx

        async def hook_b(ctx):
            execution_order.append("b")
            return ctx

        async def hook_c(ctx):
            execution_order.append("c")
            return ctx

        hm.register(HookPoint.PRE_RETRIEVE, hook_c, priority=10)
        hm.register(HookPoint.PRE_RETRIEVE, hook_a, priority=0)
        hm.register(HookPoint.PRE_RETRIEVE, hook_b, priority=5)

        await hm.run(HookPoint.PRE_RETRIEVE, {})
        assert execution_order == ["a", "b", "c"]


class TestHookExecution:
    @pytest.mark.asyncio
    async def test_context_chain(self):
        """Each hook receives context from the previous one."""
        hm = HookManager()

        async def add_key(ctx):
            ctx["added"] = True
            return ctx

        async def check_key(ctx):
            assert ctx["added"] is True
            ctx["checked"] = True
            return ctx

        hm.register(HookPoint.PRE_RETRIEVE, add_key, priority=0)
        hm.register(HookPoint.PRE_RETRIEVE, check_key, priority=1)

        result = await hm.run(HookPoint.PRE_RETRIEVE, {"query": "test"})
        assert result["added"] is True
        assert result["checked"] is True

    @pytest.mark.asyncio
    async def test_none_return_warning(self):
        """Hook returning None leaves context unchanged (with warning)."""
        hm = HookManager()

        async def bad_hook(ctx):
            pass  # returns None

        hm.register(HookPoint.PRE_RETRIEVE, bad_hook)
        result = await hm.run(HookPoint.PRE_RETRIEVE, {"query": "test"})
        assert result["query"] == "test"  # context unchanged

    @pytest.mark.asyncio
    async def test_no_hooks_returns_context(self):
        hm = HookManager()
        result = await hm.run(HookPoint.PRE_RETRIEVE, {"query": "test"})
        assert result["query"] == "test"


class TestOnErrorIsolation:
    @pytest.mark.asyncio
    async def test_on_error_hook_swallows_exceptions(self):
        """ON_ERROR hooks must not propagate exceptions."""
        hm = HookManager()

        async def bad_error_hook(ctx):
            raise RuntimeError("error hook failed")

        hm.register(HookPoint.ON_ERROR, bad_error_hook)

        # Should not raise
        result = await hm.run(HookPoint.ON_ERROR, {"error": "something"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_non_error_hook_propagates(self):
        """Non-ON_ERROR hooks propagate exceptions."""
        hm = HookManager()

        async def bad_hook(ctx):
            raise RuntimeError("hook failed")

        hm.register(HookPoint.PRE_RETRIEVE, bad_hook)

        with pytest.raises(RuntimeError, match="hook failed"):
            await hm.run(HookPoint.PRE_RETRIEVE, {})


class TestHookTimeout:
    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        """A hook exceeding timeout raises HookTimeoutError."""
        hm = HookManager(hook_timeout=0.1)

        async def slow_hook(ctx):
            await asyncio.sleep(5)
            return ctx

        hm.register(HookPoint.PRE_RETRIEVE, slow_hook)

        with pytest.raises(HookTimeoutError, match="exceeded"):
            await hm.run(HookPoint.PRE_RETRIEVE, {})

    @pytest.mark.asyncio
    async def test_fast_hook_completes(self):
        """A hook completing within timeout works normally."""
        hm = HookManager(hook_timeout=5.0)

        async def fast_hook(ctx):
            ctx["done"] = True
            return ctx

        hm.register(HookPoint.PRE_RETRIEVE, fast_hook)
        result = await hm.run(HookPoint.PRE_RETRIEVE, {})
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_on_error_timeout_swallowed(self):
        """Timeout in ON_ERROR hook is swallowed, not propagated."""
        hm = HookManager(hook_timeout=0.1)

        async def slow_error_hook(ctx):
            await asyncio.sleep(5)
            return ctx

        hm.register(HookPoint.ON_ERROR, slow_error_hook)

        # Should not raise
        result = await hm.run(HookPoint.ON_ERROR, {"error": "test"})
        assert "error" in result

    def test_custom_timeout_value(self):
        hm = HookManager(hook_timeout=60.0)
        assert hm.hook_timeout == 60.0

    def test_default_timeout_value(self):
        hm = HookManager()
        assert hm.hook_timeout == 30.0
