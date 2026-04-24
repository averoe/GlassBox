# Contributing to GlassBox RAG

Thank you for your interest in contributing to GlassBox RAG! This document covers the development setup, coding standards, and submission process.

## Development Setup

### Prerequisites

- **Python 3.9+** (3.11 or 3.12 recommended)
- **Git**
- (Optional) Docker for integration tests

### Getting Started

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/GlassBox.git
cd GlassBox

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev,embeddings,multimodal,telemetry]"

# 4. Verify installation
python -c "import glassbox_rag; print(glassbox_rag.__version__)"
```

### Running Tests

```bash
# Run all unit tests
pytest tests/unit -v

# Run with coverage
pytest tests/unit --cov=glassbox_rag --cov-report=term-missing

# Run integration tests (requires external services)
pytest tests/integration -v
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

## Coding Standards

### Style

- **Formatter:** Black (100 char line length)
- **Import order:** isort with Black profile
- **Linter:** ruff + flake8
- **Type hints:** Required on all public functions

### Patterns

- All I/O-bound code should be `async`
- Synchronous blocking calls (C extensions, SDK clients) must be wrapped in `asyncio.get_running_loop().run_in_executor()`
- Use `get_logger(__name__)` from `glassbox_rag.utils.logging`
- Never return dummy/fallback data — raise descriptive errors

### Documentation

- All public classes and methods require docstrings (Google style)
- Update `README.md` if adding user-facing features
- Update `CHANGELOG.md` for every release

## Pull Request Process

1. **Create a branch** from `main`: `git checkout -b feat/my-feature`
2. **Make changes** following the coding standards above
3. **Add tests** for new functionality (aim for >80% coverage on new code)
4. **Run the full quality check:** `black . && isort . && ruff check . && pytest`
5. **Commit** with clear messages: `feat: add embedding cache`, `fix: parallelize HybridRetriever`
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
```

## Architecture Overview

```
src/glassbox_rag/
├── __init__.py          # Public API + version
├── config.py            # Pydantic v2 configuration models
├── server.py            # FastAPI REST API
├── cli.py               # CLI entry point
├── core/
│   ├── engine.py        # Main orchestrator
│   ├── encoder.py       # Modular encoding layer
│   ├── retriever.py     # Adaptive retrieval strategies
│   ├── chunker.py       # Text chunking strategies
│   ├── reranker.py      # Cross-encoder reranking
│   ├── metrics.py       # Cost & latency tracking
│   └── writeback.py     # Safe write-back operations
├── trace/
│   ├── tracker.py       # Trace lifecycle management
│   ├── visualizer.py    # ASCII + HTML visualizations
│   └── backends.py      # Persistent trace storage
├── plugins/
│   ├── sdk.py           # Plugin registry & developer SDK
│   ├── base.py          # Abstract plugin interfaces
│   ├── vector_stores/   # Qdrant, Chroma, etc.
│   └── databases/       # PostgreSQL, SQLite, etc.
└── utils/
    ├── auth.py          # JWT + API key auth
    ├── multimodal.py    # PDF, image, PPTX extraction
    ├── telemetry.py     # OTel + Prometheus integration
    └── logging.py       # Structured logging
```

## Questions?

Open an issue or start a discussion on GitHub. We're happy to help!
