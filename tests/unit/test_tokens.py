"""Unit tests for token counter and context window management (core/tokens.py).

Covers count(), count_messages(), truncate_to_budget() non-mutation,
context window lookup for known/unknown models, and budget creation.
"""

import pytest
from glassbox_rag.core.tokens import (
    TokenCounter,
    TokenBudget,
    MODEL_CONTEXT_WINDOWS,
)


class TestTokenCount:
    def test_count_empty(self):
        tc = TokenCounter()
        assert tc.count("") == 0

    def test_count_basic(self):
        tc = TokenCounter()
        count = tc.count("hello world")
        assert count > 0

    def test_count_batch(self):
        tc = TokenCounter()
        counts = tc.count_batch(["hello", "world", "foo bar baz"])
        assert len(counts) == 3
        assert all(c > 0 for c in counts)

    def test_count_messages(self):
        tc = TokenCounter()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        count = tc.count_messages(messages)
        assert count > 0
        # Should be more than just content tokens (overhead per message)
        content_tokens = tc.count("You are helpful.") + tc.count("Hello!")
        assert count > content_tokens

    def test_backend_property(self):
        tc = TokenCounter()
        assert tc.backend in ("tiktoken", "regex")


class TestContextWindow:
    def test_known_model(self):
        tc = TokenCounter(model="gpt-4o")
        assert tc.get_context_window() == 128_000

    def test_unknown_model_fallback(self):
        tc = TokenCounter(model="totally-unknown-model-xyz")
        assert tc.get_context_window() == 4096  # safe default

    def test_prefix_match(self):
        tc = TokenCounter()
        # "gpt-4o-2024-05-13" should match "gpt-4o" prefix
        window = tc.get_context_window("gpt-4o-2024-05-13")
        assert window == 128_000

    def test_custom_windows(self):
        tc = TokenCounter(custom_context_windows={"my-model": 50_000})
        assert tc.get_context_window("my-model") == 50_000

    def test_custom_overrides_builtin(self):
        tc = TokenCounter(custom_context_windows={"gpt-4o": 999})
        assert tc.get_context_window("gpt-4o") == 999

    def test_claude_35_haiku_present(self):
        assert "claude-3.5-haiku" in MODEL_CONTEXT_WINDOWS
        assert MODEL_CONTEXT_WINDOWS["claude-3.5-haiku"] == 200_000

    def test_claude_opus_4_present(self):
        assert "claude-opus-4" in MODEL_CONTEXT_WINDOWS
        assert MODEL_CONTEXT_WINDOWS["claude-opus-4"] == 200_000

    def test_claude_sonnet_4_present(self):
        assert "claude-sonnet-4" in MODEL_CONTEXT_WINDOWS
        assert MODEL_CONTEXT_WINDOWS["claude-sonnet-4"] == 200_000

    def test_gpt4o_mini_versioned_present(self):
        assert "gpt-4o-mini-2024-07-18" in MODEL_CONTEXT_WINDOWS
        assert MODEL_CONTEXT_WINDOWS["gpt-4o-mini-2024-07-18"] == 128_000


class TestTokenBudget:
    def test_budget_creation(self):
        tc = TokenCounter()
        budget = tc.create_budget(
            system_prompt="You are helpful.",
            query="What is RAG?",
            reserved_output=1024,
        )
        assert budget.total > 0
        assert budget.system_prompt > 0
        assert budget.query > 0
        assert budget.reserved_output == 1024
        assert budget.remaining > 0

    def test_budget_utilization(self):
        budget = TokenBudget(total=1000, system_prompt=100, query=50)
        assert budget.used == 150
        assert budget.utilization == 0.15

    def test_budget_to_dict(self):
        budget = TokenBudget(total=1000, system_prompt=100)
        d = budget.to_dict()
        assert d["total"] == 1000
        assert "remaining" in d
        assert "utilization" in d


class TestTruncateToBudget:
    def test_truncation_does_not_mutate_budget(self):
        """truncate_to_budget must NOT mutate the passed-in budget."""
        tc = TokenCounter()
        budget = tc.create_budget(
            system_prompt="You are helpful.",
            query="What is RAG?",
            reserved_output=1024,
        )
        original_context_docs = budget.context_documents
        assert original_context_docs == 0

        docs = ["Document one content here.", "Document two with more content."]
        result_docs, new_budget = tc.truncate_to_budget(docs, budget)

        # Original budget must be unchanged
        assert budget.context_documents == 0
        # New budget should have context_documents set
        assert new_budget.context_documents > 0
        assert len(result_docs) > 0

    def test_truncation_returns_tuple(self):
        tc = TokenCounter()
        budget = tc.create_budget(query="test")
        docs = ["hello world"]
        result = tc.truncate_to_budget(docs, budget)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_truncation_idempotent(self):
        """Calling truncate twice with same budget gives same result."""
        tc = TokenCounter()
        budget = tc.create_budget(query="test", reserved_output=100)
        docs = ["Short doc.", "Another doc."]

        result1, budget1 = tc.truncate_to_budget(docs, budget)
        result2, budget2 = tc.truncate_to_budget(docs, budget)

        assert result1 == result2
        assert budget1.context_documents == budget2.context_documents

    def test_truncation_respects_budget(self):
        """Documents exceeding budget are truncated or excluded."""
        tc = TokenCounter()
        # Create a very tight budget
        budget = TokenBudget(
            total=100,
            system_prompt=30,
            query=20,
            reserved_output=40,
        )
        # remaining = 100 - 30 - 20 - 40 = 10 tokens
        docs = ["A " * 100]  # Very long doc
        result_docs, new_budget = tc.truncate_to_budget(docs, budget)

        # Should either truncate or exclude the doc
        if result_docs:
            assert new_budget.context_documents <= budget.remaining

    def test_empty_documents(self):
        tc = TokenCounter()
        budget = tc.create_budget(query="test")
        result_docs, new_budget = tc.truncate_to_budget([], budget)
        assert result_docs == []
        assert new_budget.context_documents == 0


class TestTokenCounterToDict:
    def test_to_dict(self):
        tc = TokenCounter(model="gpt-4o")
        d = tc.to_dict()
        assert d["model"] == "gpt-4o"
        assert "backend" in d
        assert "context_window" in d
