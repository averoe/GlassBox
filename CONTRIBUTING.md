# Contributing to GlassBox RAG

Thank you for your interest in contributing to GlassBox RAG. This document covers the development setup, coding standards, and submission process.

---

## Development Setup

### Prerequisites

- **Python 3.9+** (3.11 or 3.12 recommended)
- **Git**
- (Optional) Docker for integration tests requiring external services

### Getting Started

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/GlassBox.git
cd glassbox-rag

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev,embeddings,multimodal,telemetry,generation,tokens]"

# 4. Verify installation
python -c "import glassbox_rag; print(glassbox_rag.__version__)"
glassbox-rag version
```

### Running Tests

```bash
# Run all unit tests
pytest tests/unit -v

# Run with coverage
pytest tests/unit --cov=glassbox_rag --cov-report=term-missing

# Run integration tests (requires external services)
pytest tests/integration -v

# Run integration tests that need real Qdrant, PostgreSQL, Redis
pytest tests/integration -v -m integration_real
```

### Code Quality

```bash
# Format
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/
flake8 src/ tests/

# Type check
mypy src/glassbox_rag/
```

---

## Coding Standards

### Style

- **Formatter:** Black (100 character line length)
- **Import order:** isort with Black profile
- **Linter:** ruff + flake8
- **Type hints:** Required on all public functions. Use PEP 585/604 syntax (`list[str]` not `List[str]`, `str | None` not `Optional[str]`) with `from __future__ import annotations`.

### Patterns

- All I/O-bound code should be `async`.
- Synchronous blocking calls (C extensions, SDK clients) must be wrapped in `asyncio.get_running_loop().run_in_executor()`.
- Use `get_logger(__name__)` from `glassbox_rag.utils.logging` for all logging.
- Never return dummy/fallback data -- raise descriptive errors instead.
- Use Pydantic `SecretStr` for any API keys or secrets to prevent leakage in logs and tracebacks.

### Documentation

- All public classes and methods require docstrings (Google style).
- Update `README.md` if adding user-facing features.
- Update `CHANGELOG.md` for every release.
- Update CLI `_DOCS_TOPICS` in `cli.py` if adding new configuration sections or API methods.

---

## Pull Request Process

1. **Create a branch** from `main`: `git checkout -b feat/my-feature`
2. **Make changes** following the coding standards above
3. **Add tests** for new functionality (aim for >80% coverage on new code)
4. **Run the full quality check:**
   ```bash
   black . && isort . && ruff check . && mypy src/glassbox_rag/ && pytest tests/unit -v
   ```
5. **Commit** with clear messages following Conventional Commits
6. **Push** and open a Pull Request against `main`
7. **Fill out the PR template** with description, testing, and breaking changes

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add embedding LRU cache for query deduplication
fix: use asyncio.get_running_loop() instead of deprecated get_event_loop()
docs: update README with zero-config setup instructions
perf: parallelize HybridRetriever with asyncio.gather()
refactor: switch metrics history to OrderedDict for O(1) eviction
test: add integration tests for Supabase plugin
```

---

## Architecture Overview

```
src/glassbox_rag/
  __init__.py              # Public API + version
  __main__.py              # python -m glassbox_rag support
  config.py                # Pydantic v2 configuration models
  cli.py                   # CLI entry point (11 commands)
  core/
    engine.py              # Main orchestrator
    encoder.py             # Modular encoding layer with LRU cache
    retriever.py           # Adaptive retrieval strategies
    chunker.py             # Text chunking with size monitoring
    reranker.py            # Cross-encoder reranking
    generator.py           # LLM generation with streaming
    hooks.py               # Pipeline hook system with timeouts
    dedup.py               # Document deduplication (exact, fuzzy, semantic)
    tokens.py              # Token counting with context window budgeting
    metrics.py             # Cost and latency tracking
    writeback.py           # Safe write-back operations
    indexing_extras.py     # Lineage enrichment, drift detection, freshness
    strategies/
      query_rewrite.py     # LLM-based query rewriting hook
      multi_query.py       # Multi-query expansion with RRF
      parent_child.py      # Parent-child document retrieval
      context_compression.py  # LLM-based context compression
  evaluation/
    runner.py              # Evaluation runner and report generation
    datasets.py            # Golden dataset loading (YAML/JSON)
    retrieval.py           # Retrieval quality metrics (recall, precision)
    generation.py          # LLM-as-judge metrics (faithfulness, groundedness)
  experiments/
    experiment_tracker.py  # Experiment recording and comparison
  versioning/
    version_manager.py     # Pipeline and prompt versioning
  git/
    manager.py             # Git integration for experiment provenance
  pipelines/
    manager.py             # Multi-pipeline workspace management
  tui/
    app.py                 # Main TUI application (8 tabs)
    client.py              # In-process engine client
    theme.py               # TUI styling
    screens/               # Overview, Pipeline, Debugger, Telemetry,
                           # Config, Ingest, Evaluation, Experiments
    widgets/               # Reusable TUI widgets
  trace/
    tracker.py             # Trace lifecycle management
    visualizer.py          # ASCII trace rendering
    backends.py            # Persistent trace storage (Redis, PostgreSQL)
  plugins/
    base.py                # Abstract plugin interfaces
    sdk.py                 # Plugin registry and developer SDK
    supabase.py            # Supabase (vector store + database)
    vector_stores/         # Qdrant, Chroma, FAISS, Pinecone, Weaviate
    databases/             # PostgreSQL, SQLite, MongoDB, MySQL
  adapters/
    langchain.py           # LangChain retriever and embeddings
    llamaindex.py          # LlamaIndex query engine and retriever
    haystack.py            # Haystack component retriever
    _utils.py              # Sync-to-async bridging utilities
  utils/
    auth.py                # JWT authentication + Redis rate limiting
    multimodal.py          # PDF, image, PPTX extraction
    telemetry.py           # OTel + Prometheus integration
    logging.py             # Structured logging
    helpers.py             # Shared utility functions
```

### Key Extension Points

When adding a new feature, identify which module it belongs to:

| Area | Module | When to use |
|------|--------|-------------|
| New encoder backend | `core/encoder.py` | Adding support for a new embedding provider |
| New retrieval strategy | `core/retriever.py` | Adding a new retrieval algorithm |
| New vector store | `plugins/vector_stores/` | Adding support for a new vector database |
| New database | `plugins/databases/` | Adding support for a new document store |
| New evaluation metric | `evaluation/retrieval.py` or `evaluation/generation.py` | Adding new quality metrics |
| New pipeline hook | `core/strategies/` | Adding pre/post-processing pipeline stages |
| New TUI tab | `tui/screens/` | Adding a new dashboard view |
| New CLI command | `cli.py` | Adding a new command-line operation |

---

## Adding a New Plugin

1. Create a new file in the appropriate `plugins/` subdirectory
2. Inherit from `VectorStorePlugin` or `DatabasePlugin` (from `glassbox_rag.plugins.base`)
3. Implement all abstract methods
4. Register the plugin in `plugins/sdk.py` `register_builtins()` if it is a built-in
5. Add the optional dependency to `pyproject.toml`
6. Add tests in `tests/unit/` and optionally `tests/integration/`
7. Update the README's plugin table

---

## Questions?

Open an issue or start a discussion on GitHub. We are happy to help.
