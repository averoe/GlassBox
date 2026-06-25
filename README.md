# GlassBox RAG

A production-ready, high-transparency modular RAG (Retrieval-Augmented Generation) framework for enterprise AI/ML applications. GlassBox provides full-pipeline observability, an integrated terminal dashboard, extensible plugin architecture, built-in evaluation and experiment tracking, and real-time telemetry -- all from a single `pip install`.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Terminal Dashboard (TUI)](#terminal-dashboard-tui)
- [Python API Reference](#python-api-reference)
- [Plugin Architecture](#plugin-architecture)
- [Framework Adapters](#framework-adapters)
- [Evaluation System](#evaluation-system)
- [Experiment Tracking](#experiment-tracking)
- [Version Management](#version-management)
- [Advanced Retrieval Strategies](#advanced-retrieval-strategies)
- [Observability](#observability)
- [Production Deployment](#production-deployment)
- [Architecture](#architecture)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Core RAG Pipeline

- **Modular Encoding Layer** -- Swap between OpenAI, Cohere, Google, Ollama, ONNX, and Hugging Face encoders at runtime. Per-query encoder overrides supported.
- **Embedding Cache** -- In-memory LRU cache with SHA-256 keying, configurable max size (default 2048), and batch hit/miss tracking. Eliminates redundant embedding API calls.
- **Adaptive Retrieval** -- Semantic, keyword, and hybrid retrieval strategies with configurable weighting. Heuristic-based strategy selection adapts to query characteristics.
- **Cross-Encoder Reranking** -- Optional reranking pass via cross-encoder, Cohere, or Hugging Face models for improved precision.
- **Advanced Chunking** -- Recursive, sentence-based, semantic, and fixed-size strategies with overlap control and chunk-size monitoring.
- **Document Deduplication** -- Content-hash (SHA-256), SimHash (fuzzy), and semantic embedding-based deduplication with configurable thresholds. Runs automatically during batch ingestion.
- **Write-Back Protection** -- Confidence-gated document updates with configurable modes (read-only, protected, full) and optional human review workflow.

### LLM Generation

- **Streaming Generation** -- Async streaming for real-time token delivery via `engine.stream()`.
- **Multiple Backends** -- OpenAI and Ollama backends with config-driven setup and auto-detection from environment variables.
- **Token Counting** -- Accurate token usage tracking via tiktoken with per-model cost estimation. Supports custom context window mappings for 20+ models.
- **Configurable System Prompts** -- Per-request or config-level system prompt overrides.
- **Inline Evaluation** -- Optional faithfulness, groundedness, and context relevance scoring on every `generate()` call via the `evaluate=True` parameter.

### Terminal Dashboard (TUI)

GlassBox ships with a fully featured terminal dashboard built on Textual. Launch with `glassbox-rag tui` -- no browser or separate frontend required.

- **Overview** -- Live metrics (request count, latency, token usage, cost), component health status, and recent activity feed pulled from real trace data.
- **Pipeline** -- Interactive pipeline visualization with step-by-step execution flow and real-time test execution. Keyboard-driven step reordering.
- **Debugger** -- Trace listing with filtering, detailed step-by-step timing breakdown, anomaly detection, and trace export.
- **Telemetry** -- Latency and throughput charts, cost breakdown by operation, and performance percentile tables (p50/p95/p99) sourced from live metrics.
- **Config and Plugins** -- Interactive configuration editor and plugin documentation with code examples and inline API reference.
- **Ingest** -- File and directory ingestion interface with progress tracking.
- **Evaluation** -- Run evaluation suites against golden datasets and view per-query scoring breakdowns.
- **Experiments** -- Record, list, and compare experiment runs with metrics diff and git commit association.

### Evaluation and Experiment Tracking

- **Golden Datasets** -- Define query-expected document pairs in YAML or JSON format for repeatable evaluation.
- **Retrieval Metrics** -- recall@k, precision@k, and other configurable retrieval quality metrics.
- **Generation Metrics** -- LLM-as-judge scoring for faithfulness, groundedness, and context relevance.
- **Regression Detection** -- Compare evaluation reports with configurable thresholds to catch metric regressions.
- **Experiment Tracker** -- Record pipeline version, dataset version, git commit, and evaluation results. Compare experiments side-by-side.

### Version Management

- **Pipeline Versioning** -- Auto-incrementing pipeline version snapshots stored in `.glassbox/versions/`.
- **Prompt Versioning** -- Track system prompt changes in `.glassbox/prompts/`.
- **Git Integration** -- Optional git commit association and branch tracking for experiment reproducibility.

### Production Infrastructure

- **Rate Limiting** -- Redis-backed distributed rate limiting for production use.
- **Authentication** -- JWT token authentication and API key validation with configurable scoping.
- **OpenTelemetry** -- Optional trace export to OTLP, Jaeger, Zipkin, or console backends.
- **Prometheus Metrics** -- Exportable Prometheus metrics for Grafana integration.
- **Persistent Trace Backends** -- In-memory (default), Redis, or PostgreSQL trace storage with configurable retention.

### Developer Experience

- **Query Pipeline Hooks** -- Register before/after hooks at nine pipeline stages (PRE/POST_RETRIEVE, PRE/POST_GENERATE, PRE/POST_INGEST, PRE/POST_RERANK, ON_ERROR) with priority ordering and timeout enforcement.
- **Plugin System** -- Six plugin types (vector_store, database, encoder, reranker, chunker, processor) with a decorator-based registration API and test harness.
- **Framework Adapters** -- First-class integration with LangChain, LlamaIndex, and Haystack. Safe sync-to-async bridging for use inside running event loops (Jupyter, etc.).
- **CLI** -- 11 commands covering project scaffolding, querying, ingestion, evaluation, experiment tracking, health checks, data export, documentation browsing, and TUI dashboard.
- **Full Type Safety** -- Complete PEP 585/604 type annotations validated with mypy. Ships with `py.typed` marker.

---

## Installation

### From PyPI

```bash
pip install glassbox-rag
```

### With Optional Dependencies

```bash
# Common setup: embeddings + generation + token counting + Qdrant + SQLite
pip install "glassbox-rag[auto]"

# Same as auto but with ChromaDB instead of Qdrant
pip install "glassbox-rag[auto-chroma]"

# Full installation with all plugins
pip install "glassbox-rag[all]"

# Specific extras
pip install "glassbox-rag[embeddings]"          # All embedding providers
pip install "glassbox-rag[vector-stores]"       # All vector store backends
pip install "glassbox-rag[databases]"           # All database backends
pip install "glassbox-rag[multimodal]"          # PDF, image, PPTX extraction
pip install "glassbox-rag[telemetry]"           # OpenTelemetry + Prometheus
pip install "glassbox-rag[auth]"               # Redis rate limiting
pip install "glassbox-rag[generation]"         # LLM generation (OpenAI + tiktoken)
pip install "glassbox-rag[tokens]"             # Token counting (tiktoken)
pip install "glassbox-rag[reranking]"          # Cross-encoder reranking
pip install "glassbox-rag[adapters]"           # LangChain, LlamaIndex, Haystack
```

### From Source

```bash
git clone https://github.com/averoe/GlassBox.git
cd glassbox-rag
pip install -e ".[dev]"
```

---

## Quick Start

### Python API

```python
import asyncio
from glassbox_rag import GlassBoxEngine, GlassBoxConfig, Document

async def main():
    # Initialize -- auto-detects config from environment variables
    config = GlassBoxConfig.from_env()
    engine = GlassBoxEngine(config)
    await engine.initialize()

    # Ingest documents
    documents = [
        {"content": "GlassBox is a modular RAG framework.", "metadata": {"source": "docs"}},
        {"content": "It supports multiple vector stores.", "metadata": {"source": "docs"}},
    ]
    result = await engine.ingest(documents)
    print(f"Ingested {result['chunks_created']} chunks")

    # Retrieve
    results = await engine.retrieve("What is GlassBox?", top_k=5)
    for doc in results.documents:
        print(f"  [{doc.score:.3f}] {doc.content[:80]}")

    # Generate (requires generation backend configured)
    response = await engine.generate("What is GlassBox?", top_k=5)
    print(response.answer)

    # Generate with inline evaluation scoring
    response = await engine.generate("What is GlassBox?", evaluate=True)
    if response.evaluation:
        print(f"Faithfulness: {response.evaluation.faithfulness:.2f}")
        print(f"Groundedness: {response.evaluation.groundedness:.2f}")

    await engine.shutdown()

asyncio.run(main())
```

### Zero-Config Constructor

```python
# No config file needed -- auto-detects from environment variables
engine = GlassBoxEngine.from_env()
await engine.initialize()
```

### Launch the Terminal Dashboard

```bash
glassbox-rag tui --config config/default.yaml

# With custom refresh interval
glassbox-rag tui --config config/default.yaml --refresh 10
```

---

## CLI Reference

GlassBox provides 11 CLI commands accessible via the `glassbox-rag` entry point.

### init -- Scaffold a new project

```bash
glassbox-rag init [directory] [options]

Options:
  --encoder {openai,ollama,cohere,google,onnx}   Default encoder (default: openai)
  --vector-store {qdrant,chroma,faiss,pinecone}   Vector store (default: qdrant)
  --database {sqlite,postgresql}                  Database backend (default: sqlite)
  --force                                         Overwrite existing files
```

Creates `config/default.yaml`, `.env` template, `.gitignore`, and `data/` directory.

### tui -- Launch the terminal dashboard

```bash
glassbox-rag tui [options]

Options:
  --config PATH     Path to configuration file (default: config/default.yaml)
  --refresh INT     Auto-refresh interval in seconds (default: 5)
```

### query -- Run a retrieval query

```bash
glassbox-rag query "your query text" [options]

Options:
  --config PATH     Configuration file path
  --top-k INT       Number of results (default: 5)
  --encoder NAME    Encoder override
  --json            Output as JSON
```

### ingest -- Ingest files or directories

```bash
glassbox-rag ingest <paths...> [options]

Options:
  --config PATH     Configuration file path
  --encoder NAME    Encoder override
  --glob PATTERNS   Comma-separated file patterns (default: *.txt,*.md,*.csv,*.json,*.rst,*.html)
  --meta KEY=VALUE  Metadata pairs (repeatable)
  --json            Output as JSON
```

### health -- Run a health check

```bash
glassbox-rag health [options]

Options:
  --config PATH     Configuration file path
  --json            Output as JSON
```

Reports status of all components: encoder, retriever, vector store, database, telemetry, reranker, and write-back.

### export -- Export metrics, traces, or cache stats

```bash
glassbox-rag export [options]

Options:
  --config PATH                 Configuration file path
  --type {metrics,traces,cache,all}  Data type to export (default: all)
  --output PATH                 Output file path (default: stdout)
  --format {json,text}          Output format (default: json)
```

### check -- Validate configuration

```bash
glassbox-rag check --config config/default.yaml
```

### docs -- Interactive documentation browser

```bash
glassbox-rag docs [topic]
glassbox-rag docs --list

Available topics: config, retrieval, plugins, hooks, api, telemetry
```

### eval -- Evaluation commands

```bash
# Run evaluation on a golden dataset
glassbox-rag eval run --dataset path/to/golden.yaml --config config/default.yaml \
  --metrics "recall@5,precision@5,faithfulness" [--json]

# List evaluation reports
glassbox-rag eval list [--limit 20] [--json]

# Compare two reports for regressions
glassbox-rag eval compare <report_a> <report_b> [--threshold 0.05]
```

### experiment -- Experiment tracking

```bash
# List recorded experiments
glassbox-rag experiment list [--limit 20] [--json]

# Record a new experiment
glassbox-rag experiment record [--version INT] [--dataset STR] [--trace-id STR] [--json]

# Compare two experiments
glassbox-rag experiment compare <exp_a> <exp_b> [--json]
```

### version -- Show version

```bash
glassbox-rag version
```

---

## Configuration

GlassBox uses YAML configuration with `${ENV_VAR}` and `${ENV_VAR:default}` substitution. A zero-config mode is also available via `GlassBoxConfig.from_env()`.

### Configuration Sections

| Section | Description |
|---------|-------------|
| `server` | Host, port, workers settings |
| `logging` | Log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`) and format (`json`/`text`) |
| `trace` | Trace backend (`memory`/`redis`/`postgresql`), retention, sample rate |
| `encoding` | Encoder setup: default encoder, cloud providers, local providers |
| `vector_store` | Vector store backend and connection settings |
| `database` | Document storage backend and connection settings |
| `chunking` | Strategy (`recursive`/`sentence`/`semantic`/`fixed`), size, overlap |
| `retrieval` | Top-k, min score, adaptive strategies, reranking |
| `generation` | LLM backend (`openai`/`ollama`), model, temperature, system prompt |
| `writeback` | Write-back mode (`read-only`/`protected`/`full`), confidence threshold |
| `metrics` | Cost and latency tracking toggles |
| `telemetry` | OpenTelemetry and Prometheus export settings |
| `security` | API key requirements, CORS, rate limiting |
| `auth` | JWT authentication settings |
| `evaluation` | Evaluation backend, metrics, regression threshold |
| `versioning` | Pipeline/prompt versioning directories |
| `git` | Git integration (auto-commit, branch association) |
| `query_rewrite` | LLM-based query rewriting pre-retrieval |
| `context_compression` | LLM-based context compression pre-generation |
| `parent_child` | Parent-child document retrieval |
| `multi_query` | Multi-query expansion with reciprocal rank fusion |
| `indexing_extras` | Drift detection, freshness validation, lineage tracking |
| `plugins` | Custom plugin loading |

### Example Configuration

```yaml
encoding:
  default_encoder: "openai"
  cloud:
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "text-embedding-3-small"
      embedding_dim: 1536

vector_store:
  type: "qdrant"
  qdrant:
    host: "localhost"
    port: 6333
    collection_name: "glassbox_docs"

database:
  type: "sqlite"
  sqlite:
    path: "./data/glassbox.db"

chunking:
  strategy: "recursive"
  chunk_size: 512
  chunk_overlap: 50

retrieval:
  top_k: 5
  min_score: 0.3
  rerank_enabled: false
  adaptive:
    enabled: true
    strategies:
      - name: "semantic"
      - name: "hybrid"
        weight_semantic: 0.6
        weight_keyword: 0.4

generation:
  backend: "openai"
  model: "gpt-4o-mini"
  temperature: 0.7

trace:
  enabled: true
  backend: "memory"

writeback:
  enabled: true
  mode: "protected"
  protected:
    confidence_threshold: 0.8

metrics:
  enabled: true
  track_tokens: true
  track_latency: true
  track_cost: true
```

---

## Terminal Dashboard (TUI)

The TUI is built with [Textual](https://github.com/Textualize/textual) and runs the GlassBox engine in-process -- no separate server required.

Launch with `glassbox-rag tui` after configuring your project.

### Tabs

| Tab | Key | Description |
|-----|-----|-------------|
| **Overview** | `1` | System health, live metrics (requests, latency, tokens, cost), component status. |
| **Pipeline** | `2` | Interactive pipeline visualization. Run test queries and observe step-by-step execution with timing. Supports keyboard-driven step reordering (Alt+Up/Down). |
| **Debugger** | `3` | Browse execution traces, filter by status, inspect individual trace steps with duration breakdown, anomaly detection, and trace export. |
| **Telemetry** | `4` | Latency/throughput charts, per-operation cost breakdown, and performance percentile tables (p50/p95/p99). |
| **Config and Plugins** | `5` | Interactive configuration editor and plugin documentation with code examples and inline API reference for all plugin types. |
| **Ingest** | `6` | File and directory ingestion interface with progress tracking and metadata attachment. |
| **Evaluation** | `7` | Run evaluation suites, view per-query scores, and browse historical evaluation reports. |
| **Experiments** | `8` | Record experiments, list historical runs, and compare experiment metrics side-by-side. |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Refresh all data |
| `1` -- `8` | Switch tabs |

---

## Python API Reference

### Core Engine Methods

| Method | Description |
|--------|-------------|
| `engine.retrieve(query, top_k=5, encoder=None)` | Retrieve relevant documents for a query. Returns `RetrievalResult`. |
| `engine.ingest(documents, encoder=None)` | Ingest documents into the pipeline. Accepts list of dicts with `content` and `metadata` keys. |
| `engine.update(document_id, content, metadata=None, confidence_score=1.0)` | Update a document with write-back protection. Returns `WriteBackResult`. |
| `engine.generate(query, encoder=None, top_k=None, system_prompt=None, temperature=None, max_tokens=None, evaluate=False)` | Retrieve + generate an answer. Returns `GenerateResponse`. Set `evaluate=True` for inline scoring. |
| `engine.stream(query, encoder=None, top_k=None, system_prompt=None, temperature=None, max_tokens=None)` | Async streaming retrieve + generate. Yields `StreamEvent` objects (type: `token`, `metadata`, `done`). |
| `engine.batch_ingest(documents, batch_size=50, encoder=None, on_progress=None, deduplicate=True, max_concurrent_batches=3)` | Concurrent batch ingestion with progress callbacks and automatic deduplication. |

### Observability Methods

| Method | Description |
|--------|-------------|
| `engine.list_traces(limit=100)` | List execution traces. |
| `engine.get_trace(trace_id)` | Get a single trace with full step tree. |
| `engine.get_metrics_summary()` | JSON metrics summary (requests, cost, latency, histograms). |
| `engine.get_telemetry_status()` | OTel and Prometheus configuration and connection status. |
| `engine.get_cache_stats()` | Embedding cache hit/miss/eviction statistics. |
| `engine.get_chunk_report()` | Chunk size distribution report. |
| `engine.get_hook_info()` | Registered lifecycle hooks and their configuration. |
| `engine.get_token_counter()` | Access the engine's token counter for budget allocation. |

### Constructors

| Constructor | Description |
|-------------|-------------|
| `GlassBoxEngine(config)` | Standard constructor from a `GlassBoxConfig` object. |
| `GlassBoxEngine.from_env()` | Zero-config constructor. Auto-detects API keys and backends from environment variables. |
| `GlassBoxConfig.from_env()` | Build configuration purely from environment variables (`OPENAI_API_KEY`, `OLLAMA_BASE_URL`, `GLASSBOX_*`, etc.). |

### Response Types

**GenerateResponse** -- returned by `engine.generate()`:

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `str` | Generated answer text |
| `sources` | `list[dict]` | Source documents used for generation |
| `retrieval` | `dict` | Retrieval metadata (strategy, timing, count) |
| `generation` | `dict` | Generation metadata (model, tokens, timing) |
| `trace_id` | `str` | Trace ID for debugging |
| `pipeline_version` | `int` | Current pipeline version number |
| `evaluation` | `InlineEvaluation or None` | Faithfulness, groundedness, context_relevance scores (only when `evaluate=True`) |

Supports dict-like access for backward compatibility: `response["answer"]` works alongside `response.answer`.

---

## Plugin Architecture

GlassBox supports six plugin types. The plugin registry is thread-safe and supports both built-in and third-party plugins.

### Supported Plugin Types

| Type | Description | Built-in Implementations |
|------|-------------|--------------------------|
| `vector_store` | Vector similarity search | Qdrant, Chroma, FAISS, Pinecone, Weaviate, Supabase |
| `database` | Document storage | PostgreSQL, SQLite, MongoDB, MySQL, Supabase |
| `encoder` | Text embedding providers | OpenAI, Cohere, Google, Ollama, ONNX, Hugging Face |
| `reranker` | Post-retrieval reranking | Cross-encoder, Cohere, Hugging Face |
| `chunker` | Text chunking strategies | Recursive, Sentence, Semantic, Fixed |
| `processor` | Content processors | PDF, Image (OCR), PPTX |

### Writing a Vector Store Plugin

```python
from glassbox_rag.plugins.base import VectorStorePlugin

class CustomVectorStore(VectorStorePlugin):
    async def initialize(self) -> bool:
        # Connect to your backend
        return True

    async def shutdown(self) -> None:
        # Clean up connections
        pass

    async def health_check(self) -> bool:
        return True

    async def add_vectors(self, vectors, ids=None, contents=None, metadata=None):
        # Store vectors and return list of IDs
        return ["id-1", "id-2"]

    async def search(self, query_vector, top_k=5):
        # Return list of (id, score, content, metadata) tuples
        return [("id-1", 0.95, "document text", {"source": "example"})]

    async def get_vector(self, vector_id):
        return {"id": vector_id, "content": "...", "metadata": {}}

    async def delete_vector(self, vector_id):
        return True

    async def count(self):
        return 42
```

### Writing a Database Plugin

```python
from glassbox_rag.plugins.base import DatabasePlugin

class CustomDatabase(DatabasePlugin):
    async def initialize(self) -> bool:
        return True

    async def shutdown(self) -> None:
        pass

    async def health_check(self) -> bool:
        return True

    async def connect(self) -> bool:
        return True

    async def insert(self, table, data):
        return "record-id"

    async def update(self, table, record_id, data):
        return True

    async def delete(self, table, record_id):
        return True

    async def query(self, table, filters=None):
        return [{"id": "1", "content": "..."}]

    async def search_text(self, terms, top_k=10):
        return [{"id": "1", "content": "...", "score": 0.9}]

    async def ensure_tables(self) -> None:
        pass
```

### Registering Custom Plugins

**Decorator-based registration:**

```python
from glassbox_rag.plugins.sdk import register_plugin

@register_plugin("vector_store", "my_store")
class MyStore(VectorStorePlugin):
    ...
```

**Programmatic registration:**

```python
from glassbox_rag.plugins.sdk import plugin_registry

plugin_registry.register("vector_store", "my_store", MyStore)
```

**Configuration-based loading:**

```yaml
plugins:
  custom:
    - type: vector_store
      name: my_store
      module: mypackage.my_vector_store
      class: MyVectorStore
```

### Plugin Test Harness

```python
from glassbox_rag.plugins.sdk import PluginTestHarness

results = await PluginTestHarness.test_vector_store(my_store_instance)
print(results["passed"])   # ["initialize", "health_check", "add_vectors", ...]
print(results["failed"])   # []
```

### Query Pipeline Hooks

```python
from glassbox_rag import GlassBoxEngine, HookPoint

engine = GlassBoxEngine(config)
await engine.initialize()

# Decorator-based registration
@engine.hooks.on(HookPoint.POST_RETRIEVE, priority=10)
async def log_results(context):
    print(f"Retrieved {len(context['documents'])} documents")
    return context

# Direct registration
async def filter_low_scores(context):
    context["documents"] = [d for d in context["documents"] if d.score > 0.5]
    return context

engine.hooks.register(HookPoint.POST_RETRIEVE, filter_low_scores, priority=20)
```

**Available Hook Points:**

| Hook Point | Trigger |
|------------|---------|
| `HookPoint.PRE_RETRIEVE` | Before query encoding and retrieval |
| `HookPoint.POST_RETRIEVE` | After retrieval, before returning results |
| `HookPoint.PRE_GENERATE` | Before LLM generation (RAG context ready) |
| `HookPoint.POST_GENERATE` | After LLM response is generated |
| `HookPoint.PRE_INGEST` | Before document ingestion |
| `HookPoint.POST_INGEST` | After document ingestion (drift detector runs here) |
| `HookPoint.PRE_RERANK` | Before reranking pass |
| `HookPoint.POST_RERANK` | After reranking pass |
| `HookPoint.ON_ERROR` | When any pipeline operation fails (errors swallowed) |

Hooks execute in priority order (lower values run first). Each hook has a configurable timeout (default: 30 seconds).

---

## Framework Adapters

GlassBox integrates with popular LLM frameworks. All adapters include safe sync-to-async bridging for use inside running event loops (Jupyter, Colab, etc.).

### LangChain

```python
from glassbox_rag.adapters.langchain import GlassBoxRetriever, GlassBoxEmbeddings

# As a retriever (LCEL-compatible)
retriever = GlassBoxRetriever(engine=engine, top_k=5, encoder="openai")
docs = await retriever.ainvoke("your query")        # async (primary)
docs = retriever.invoke("your query")               # sync (auto-bridged)

# As an embeddings provider
embeddings = GlassBoxEmbeddings(engine=engine, encoder="openai")
vectors = await embeddings.aembed_documents(["text1", "text2"])
query_vec = await embeddings.aembed_query("your query")
```

Registers as a virtual subclass of `langchain_core.retrievers.BaseRetriever` and `langchain_core.embeddings.Embeddings` when langchain-core is installed.

### LlamaIndex

```python
from glassbox_rag.adapters.llamaindex import GlassBoxQueryEngine, GlassBoxLlamaRetriever

# As a query engine (retrieve + generate)
query_engine = GlassBoxQueryEngine(engine=engine, top_k=5)
response = await query_engine.aquery("your query")  # returns llama_index Response

# As a retriever only
retriever = GlassBoxLlamaRetriever(engine=engine, top_k=5)
nodes = await retriever.aretrieve("your query")     # returns NodeWithScore list
```

### Haystack

```python
from glassbox_rag.adapters.haystack import GlassBoxHaystackRetriever

# As a Haystack component (decorated with @component when haystack-ai is installed)
retriever = GlassBoxHaystackRetriever(engine=engine, top_k=5)
result = await retriever.run(query="your query")
haystack_docs = result["documents"]
```

Install adapters with:

```bash
pip install "glassbox-rag[adapters-langchain]"    # langchain-core
pip install "glassbox-rag[adapters-llamaindex]"    # llama-index-core
pip install "glassbox-rag[adapters-haystack]"      # haystack-ai
pip install "glassbox-rag[adapters]"               # all three
```

---

## Evaluation System

GlassBox includes a built-in evaluation framework for measuring retrieval and generation quality.

### Golden Datasets

Define expected results in YAML:

```yaml
queries:
  - query: "What is GlassBox?"
    expected_doc_ids: ["doc-1", "doc-3"]
    expected_answer_keywords: ["modular", "RAG", "framework"]
  - query: "How does chunking work?"
    expected_doc_ids: ["doc-5"]
```

### Running Evaluations

```python
from glassbox_rag.evaluation import EvaluationRunner, GoldenDataset

dataset = GoldenDataset.from_yaml(".glassbox/datasets/golden.yaml")
runner = EvaluationRunner(pipeline=engine, eval_config=config.evaluation)
report = await runner.run(dataset, metrics=["recall@5", "precision@5", "faithfulness"])
print(report.summary())
```

### Regression Testing

```python
regression = report_a.compare(report_b, threshold=0.05)
print(regression.summary())
```

### CLI Evaluation

```bash
glassbox-rag eval run --dataset golden.yaml --metrics "recall@5,faithfulness"
glassbox-rag eval compare report_a.json report_b.json --threshold 0.05
```

---

## Experiment Tracking

Record and compare pipeline experiments with full provenance tracking.

```python
from glassbox_rag import ExperimentTracker

tracker = ExperimentTracker.from_workspace()

# Record an experiment
record = tracker.record(config_version=1, dataset_version="v2", trace_id="abc123")

# List experiments
experiments = tracker.list(limit=20)

# Compare two experiments
report = tracker.compare("exp-a-id", "exp-b-id")
print(report.summary)
```

Experiments are stored in `.glassbox/experiments/` and include:
- Pipeline version number
- Git commit hash (auto-detected)
- Dataset version identifier
- Evaluation metric results
- Timestamp

---

## Version Management

```python
from glassbox_rag import VersionManager

vm = VersionManager()
vm.snapshot(config)                    # Save current pipeline configuration
current_version = vm.current_version   # Get current version number
```

Versioning state is stored in `.glassbox/versions/` with auto-incrementing version numbers. The pipeline version is automatically attached to every `GenerateResponse`.

---

## Advanced Retrieval Strategies

GlassBox supports configurable retrieval strategies that can be composed as pipeline hooks.

### Query Rewriting

LLM-based query rewriting for improved retrieval quality:

```yaml
query_rewrite:
  enabled: true
  prompt_template: "Rewrite the following user query into a clearer form.\n\nQuery: {query}"
```

### Context Compression

Post-retrieval context compression to reduce noise before generation:

```yaml
context_compression:
  enabled: true
```

### Parent-Child Retrieval

Retrieve child chunks but return parent documents for broader context:

```yaml
parent_child:
  enabled: true
```

### Multi-Query Expansion

Expand a single query into multiple sub-queries with reciprocal rank fusion:

```yaml
multi_query:
  enabled: true
  num_queries: 3
  fusion_method: "rrf"
```

### Indexing Extras

```yaml
indexing_extras:
  drift_detection_enabled: true    # Detect embedding drift across ingestion batches
  drift_threshold: 0.15            # Cosine distance threshold for drift alerts
  freshness_validation: true       # Warn if encoder config changed since last ingestion
  lineage_tracking: true           # Track chunk-to-document lineage metadata
```

---

## Observability

### OpenTelemetry

Enable OTLP export in configuration:

```yaml
telemetry:
  otel_enabled: true
  otel_exporter: "otlp"              # otlp, jaeger, zipkin, console
  otel_endpoint: "http://localhost:4317"
  service_name: "glassbox-rag"
```

### Prometheus Metrics

Metrics can be exported in Prometheus text format via the engine's telemetry module:

```python
content = engine.telemetry.get_prometheus_metrics()
```

Tracked metrics include: total requests, tokens used, estimated cost, per-operation latency percentiles (p50/p95/p99), embedding cache hit rate, active request count, vector and chunk counts.

### Trace Storage

Traces can be stored in memory (default), Redis, or PostgreSQL:

```yaml
trace:
  enabled: true
  backend: "redis"            # or "postgresql" or "memory"
  retention_days: 30
  sample_rate: 1.0            # 0.0 to 1.0
```

---

## Production Deployment

### Docker

```bash
docker build -t glassbox-rag .

docker run \
  -v $(pwd)/config:/app/config:ro \
  -e OPENAI_API_KEY=your_key \
  glassbox-rag
```

### Docker Compose

```bash
# Development (app + Qdrant + PostgreSQL)
docker-compose up -d

# Production with monitoring
docker-compose -f docker-compose.prod.yml --profile monitoring up -d
```

Production services include:

- **GlassBox RAG** -- Main application with health checks and resource limits
- **Qdrant** -- Vector database with persistence
- **PostgreSQL** -- Document storage
- **Redis** -- Rate limiting and trace caching (optional)
- **Prometheus + Grafana** -- Metrics and dashboards (optional, monitoring profile)

---

## Architecture

```
  CLI / Python API
       |
       v
  GlassBox Engine
       |
       +-- Encoding Layer -----> [OpenAI | Cohere | Ollama | ONNX | Google | HF]
       +-- Chunker ------------> [Recursive | Sentence | Semantic | Fixed]
       +-- Deduplicator -------> [Content Hash | SimHash | Semantic Embedding]
       +-- Retriever ----------> [Semantic | Keyword | Hybrid | Adaptive]
       +-- Reranker -----------> [Cross-Encoder | Cohere | HuggingFace]
       +-- Generator ----------> [OpenAI | Ollama] (with streaming)
       +-- Vector Store -------> [Qdrant | Chroma | FAISS | Pinecone | Weaviate | Supabase]
       +-- Database -----------> [PostgreSQL | SQLite | MongoDB | MySQL | Supabase]
       +-- Write-Back Manager -> [Protected | Full | Read-Only]
       +-- Hook Manager -------> [Pre/Post Retrieve | Pre/Post Generate | ...]
       |
       +-- Trace Tracker ------> [Memory | Redis | PostgreSQL]
       +-- Metrics Tracker ----> [Prometheus | JSON]
       +-- Telemetry ----------> [OpenTelemetry | Console]
       |
       +-- Evaluation ----------> [Retrieval Metrics | Generation Metrics | Regression]
       +-- Experiments ---------> [Record | Compare | Git Association]
       +-- Versioning ----------> [Pipeline Snapshots | Prompt Versioning]
       |
       +-- Strategies ----------> [Query Rewrite | Multi-Query | Parent-Child | Context Compression]
       +-- Indexing Extras -----> [Drift Detection | Freshness Validation | Lineage Tracking]
       |
       v
  Terminal Dashboard (TUI)
       |
       +-- Overview (live metrics + health)
       +-- Pipeline (interactive visualization)
       +-- Debugger (trace inspection)
       +-- Telemetry (charts + tables)
       +-- Config & Plugins (editor + docs)
       +-- Ingest (file ingestion)
       +-- Evaluation (golden dataset scoring)
       +-- Experiments (experiment management)
```

### Project Structure

```
glassbox-rag/
  src/glassbox_rag/
    __init__.py              # Public API and version
    __main__.py              # python -m glassbox_rag support
    config.py                # Pydantic v2 configuration models
    cli.py                   # CLI entry point (11 commands)
    core/
      engine.py              # Main orchestrator
      retriever.py           # Adaptive retrieval strategies
      chunker.py             # Text chunking with size monitoring
      encoder.py             # Modular encoding layer with cache
      generator.py           # LLM generation with streaming
      hooks.py               # Pipeline hook system with timeouts
      dedup.py               # Document deduplication (exact, fuzzy, semantic)
      tokens.py              # Token counting with context window budgeting
      writeback.py           # Write-back protection
      metrics.py             # Cost and latency tracking
      reranker.py            # Cross-encoder reranking
      indexing_extras.py     # Lineage, drift detection, freshness validation
      strategies/
        query_rewrite.py     # LLM-based query rewriting hook
        multi_query.py       # Multi-query expansion with RRF
        parent_child.py      # Parent-child document retrieval
        context_compression.py  # LLM-based context compression
    trace/
      tracker.py             # Concurrency-safe trace system
      visualizer.py          # ASCII trace rendering
      backends.py            # Persistent trace storage (Redis, PostgreSQL)
    evaluation/
      runner.py              # Evaluation runner and reporting
      datasets.py            # Golden dataset loader
      retrieval.py           # Retrieval quality metrics
      generation.py          # LLM-as-judge generation metrics
    experiments/
      experiment_tracker.py  # Experiment recording and comparison
    versioning/
      version_manager.py     # Pipeline and prompt versioning
    git/
      manager.py             # Git integration for experiment provenance
    pipelines/
      manager.py             # Multi-pipeline workspace management
    tui/
      app.py                 # Main TUI application
      client.py              # In-process engine client
      theme.py               # TUI styling
      screens/               # Overview, Pipeline, Debugger, Telemetry,
                             # Config, Ingest, Evaluation, Experiments
      widgets/               # Reusable TUI widgets (metric cards,
                             # trace tree, pipeline view, etc.)
    plugins/
      base.py                # Abstract plugin interfaces
      sdk.py                 # Plugin registry and developer SDK
      vector_stores/         # Qdrant, Chroma, FAISS, Pinecone, Weaviate
      databases/             # PostgreSQL, SQLite, MongoDB, MySQL
      supabase.py            # Supabase (vector store + database)
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
  config/
    default.yaml             # Default configuration
  examples/                  # Example scripts
  tests/                     # Unit and integration test suites
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=glassbox_rag --cov-report=html

# Run specific categories
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run integration tests that require external services
pytest tests/integration/ -v -m integration_real
```

---

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and type checks succeed
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

Licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## Links

- **Repository**: [github.com/averoe/GlassBox](https://github.com/averoe/GlassBox)
- **Issues**: [github.com/averoe/GlassBox/issues](https://github.com/averoe/GlassBox/issues)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)