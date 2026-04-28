# GlassBox RAG

A production-ready, high-transparency modular RAG (Retrieval-Augmented Generation) framework for enterprise AI/ML applications. GlassBox provides full-pipeline observability, an integrated web dashboard, extensible plugin architecture, and real-time telemetry -- all from a single `pip install`.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI](#cli)
- [Configuration](#configuration)
- [Web Dashboard](#web-dashboard)
- [API Reference](#api-reference)
- [Plugin Architecture](#plugin-architecture)
- [Framework Adapters](#framework-adapters)
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
- **Embedding Cache** -- In-memory LRU cache eliminates redundant embedding API calls.
- **Adaptive Retrieval** -- Semantic, keyword, and hybrid retrieval strategies with configurable weighting.
- **Cross-Encoder Reranking** -- Optional reranking pass for improved precision.
- **Advanced Chunking** -- Recursive, sentence-based, semantic, and fixed-size strategies with overlap control and chunk-size monitoring.
- **Document Deduplication** -- Content-hash and SimHash-based deduplication with configurable thresholds. Runs automatically during batch ingestion.
- **Write-Back Protection** -- Confidence-gated document updates with optional human review workflow.

### LLM Generation

- **Streaming Generation** -- Server-Sent Events (SSE) endpoint for real-time token streaming.
- **Multiple Backends** -- OpenAI and Ollama backends with auto-detection from environment variables.
- **Token Counting** -- Accurate token usage tracking via tiktoken with per-model cost estimation.
- **Configurable System Prompts** -- Per-request or config-level system prompt overrides.

### Integrated Web Dashboard

The dashboard is served directly by the FastAPI server at the root URL -- no separate frontend build or deployment required.

- **Overview** -- Live metrics (request count, latency, token usage, cost), component health status, and recent activity feed pulled from real trace data.
- **Pipeline Visualization** -- Interactive SVG pipeline diagram with step-by-step execution flow and real-time test execution.
- **Debugger** -- Trace listing with filtering, detailed step-by-step timing breakdown, and trace visualization.
- **Telemetry** -- Latency and throughput charts, cost breakdown by operation, and performance percentile tables (p50/p95/p99) -- all sourced from live metrics.
- **Plugins** -- Interactive documentation for all plugin types with code examples.

### Production Infrastructure

- **Rate Limiting** -- Memory-based (single worker) or Redis-backed (multi-worker) rate limiting middleware.
- **Authentication** -- JWT token authentication and API key validation with configurable scoping.
- **CORS** -- Configurable cross-origin resource sharing.
- **OpenTelemetry** -- Optional trace export to OTLP, Jaeger, or console backends.
- **Prometheus Metrics** -- Scrapable `/metrics/prometheus` endpoint for Grafana integration.
- **Persistent Trace Backends** -- In-memory (default), Redis, or PostgreSQL trace storage with configurable retention.

### Developer Experience

- **Query Pipeline Hooks** -- Register before/after hooks at any pipeline stage (pre-retrieval, post-retrieval, pre-generation, post-generation) for logging, transformation, or validation.
- **Plugin System** -- Four plugin types (Encoder, Vector Store, Database, Multimodal) with a registration API.
- **Framework Adapters** -- First-class integration with LangChain, LlamaIndex, and Haystack.
- **CLI** -- Project scaffolding, server management, and config validation from the command line.
- **Full Type Safety** -- Complete type annotations validated with mypy. Ships with `py.typed` marker.

---

## Installation

### From PyPI

```bash
pip install glassbox-rag
```

### With Optional Dependencies

```bash
# Common setup: embeddings + generation + token counting
pip install "glassbox-rag[auto]"

# Full installation with all plugins
pip install "glassbox-rag[all]"

# Specific extras
pip install "glassbox-rag[embeddings]"          # All embedding providers
pip install "glassbox-rag[vector-stores]"       # All vector store backends
pip install "glassbox-rag[databases]"           # All database backends
pip install "glassbox-rag[telemetry]"           # OpenTelemetry + Prometheus
pip install "glassbox-rag[auth]"               # JWT authentication
pip install "glassbox-rag[generation]"         # LLM generation
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
        Document(content="GlassBox is a modular RAG framework.", metadata={"source": "docs"}),
        Document(content="It supports multiple vector stores.", metadata={"source": "docs"}),
    ]
    result = await engine.ingest(documents)
    print(f"Ingested {result['chunks_created']} chunks")

    # Retrieve
    results = await engine.retrieve("What is GlassBox?", top_k=5)
    for r in results:
        print(f"  [{r.score:.3f}] {r.content[:80]}")

    # Generate (requires generation backend configured)
    response = await engine.generate("What is GlassBox?", top_k=5)
    print(response["answer"])

    await engine.shutdown()

asyncio.run(main())
```

### Start the Server

```bash
# Using the CLI
glassbox-rag serve --config config/default.yaml

# Or with uvicorn directly
uvicorn glassbox_rag.server:create_app --factory --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` for the dashboard, or `http://localhost:8000/docs` for the OpenAPI explorer.

---

## CLI

```bash
# Initialize a new project with scaffolded config
glassbox-rag init --encoder openai --vector-store qdrant --database sqlite

# Start the server
glassbox-rag serve --config config/default.yaml --port 8000

# Validate configuration
glassbox-rag check --config config/default.yaml

# Show version
glassbox-rag version
```

---

## Configuration

GlassBox uses YAML configuration with `${ENV_VAR}` and `${ENV_VAR:default}` substitution. A zero-config mode is also available via `GlassBoxConfig.from_env()`.

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

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
  adaptive:
    enabled: true
    strategies:
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

telemetry:
  prometheus_enabled: true
  service_name: "glassbox-rag"

security:
  api_key_required: false
  cors:
    enabled: true
    origins: ["*"]

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

## Web Dashboard

The dashboard is bundled inside the package and served automatically when the server starts. No separate build step is needed.

Access it at `http://localhost:8000` after starting the server.

### Pages

| Page | Description |
|------|-------------|
| **Overview** | System health, live metrics (requests, latency, tokens, cost), component status, and recent activity feed sourced from real trace data. |
| **Pipeline** | Interactive SVG visualization of the RAG pipeline. Run test queries and observe step-by-step execution with timing. |
| **Debugger** | Browse execution traces, filter by status, inspect individual trace steps with duration breakdown. |
| **Telemetry** | Latency/throughput charts, per-operation cost breakdown, and performance percentile tables (p50/p95/p99). |
| **Plugins** | Interactive documentation for Encoder, Vector Store, Database, and Multimodal plugin types with code examples. |

All dashboard data is fetched from the same FastAPI backend via `/health`, `/metrics`, and `/traces` endpoints -- no mock data.

---

## API Reference

### Core Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check with per-component status |
| POST | `/retrieve` | Retrieve relevant documents for a query |
| POST | `/retrieve/stream` | Streaming retrieval + generation via SSE |
| POST | `/ingest` | Ingest documents into the pipeline |
| PUT | `/update` | Update documents with write-back protection |

### Traces and Debugging

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/traces` | List execution traces (supports `?limit=N`) |
| GET | `/traces/{trace_id}` | Get a single trace with full step tree |
| GET | `/traces/{trace_id}/visualize` | Visual debugger with ASCII/HTML output |
| GET | `/traces/compare?trace_a=...&trace_b=...` | Side-by-side trace comparison |
| GET | `/traces/anomalies` | Detect anomalous steps across recent traces |

### Metrics and Telemetry

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/metrics` | JSON metrics summary (requests, cost, latency) |
| GET | `/metrics/prometheus` | Prometheus-scrapable text format |
| GET | `/metrics/chunks` | Chunk size distribution report |

### Dashboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves the web dashboard |
| GET | `/dashboard` | Alias for `/` |
| GET | `/static/*` | Static assets (CSS, JS) |

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/token` | Generate a JWT token |

---

## Plugin Architecture

GlassBox supports four plugin types. Register plugins via configuration or the Python API.

### Encoder Plugins

```python
from glassbox_rag.core.encoder import BaseEncoder
import numpy as np

class CustomEncoder(BaseEncoder):
    async def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = await self.compute_embeddings(texts)
        return np.array(embeddings)

    async def encode_query(self, query: str) -> np.ndarray:
        return await self.encode([query])
```

### Vector Store Plugins

```python
from glassbox_rag.plugins.base import VectorStorePlugin

class CustomVectorStore(VectorStorePlugin):
    async def add_vectors(self, vectors, metadata):
        return await self.store(vectors, metadata)

    async def search(self, query_vector, top_k):
        return await self.perform_search(query_vector, top_k)
```

### Database Plugins

```python
from glassbox_rag.plugins.base import DatabasePlugin

class CustomDatabase(DatabasePlugin):
    async def store_document(self, doc_id, content, metadata):
        await self.insert(doc_id, content, metadata)

    async def get_document(self, doc_id):
        return await self.fetch(doc_id)
```

### Query Pipeline Hooks

```python
from glassbox_rag import GlassBoxEngine, HookPoint

engine = GlassBoxEngine(config)
await engine.initialize()

# Register a hook that runs after retrieval
@engine.hooks.register(HookPoint.POST_RETRIEVAL)
async def log_results(context):
    print(f"Retrieved {len(context['results'])} documents")
    return context
```

---

## Framework Adapters

GlassBox integrates with popular LLM frameworks:

### LangChain

```python
from glassbox_rag.adapters.langchain import GlassBoxRetriever

retriever = GlassBoxRetriever(engine=engine, top_k=5)
docs = await retriever.aget_relevant_documents("your query")
```

### LlamaIndex

```python
from glassbox_rag.adapters.llamaindex import GlassBoxQueryEngine

query_engine = GlassBoxQueryEngine(engine=engine)
response = await query_engine.aquery("your query")
```

### Haystack

```python
from glassbox_rag.adapters.haystack import GlassBoxHaystackRetriever

retriever = GlassBoxHaystackRetriever(engine=engine)
```

Install adapters with:

```bash
pip install "glassbox-rag[adapters-langchain]"
pip install "glassbox-rag[adapters-llamaindex]"
pip install "glassbox-rag[adapters-haystack]"
pip install "glassbox-rag[adapters]"   # all three
```

---

## Observability

### Prometheus Metrics

Scrape `http://localhost:8000/metrics/prometheus` with your Prometheus instance:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: "glassbox-rag"
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: "/metrics/prometheus"
```

### OpenTelemetry

Enable OTLP export in configuration:

```yaml
telemetry:
  otel_enabled: true
  otel_exporter: "otlp"
  otel_endpoint: "http://localhost:4317"
  service_name: "glassbox-rag"
```

### Trace Storage

Traces can be stored in memory (default), Redis, or PostgreSQL:

```yaml
trace:
  enabled: true
  backend: "redis"        # or "postgresql" or "memory"
  retention_days: 30
  sample_rate: 1.0
```

---

## Production Deployment

### Docker

```bash
docker build -t glassbox-rag .

docker run -p 8000:8000 \
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
Client Request
      |
      v
  FastAPI Server
      |
      +-- Authentication (JWT / API Key)
      +-- Rate Limiting (Memory / Redis)
      |
      v
  GlassBox Engine
      |
      +-- Encoding Layer ----> [OpenAI | Cohere | Ollama | ONNX | Google | HF]
      +-- Chunker -----------> [Recursive | Sentence | Semantic | Fixed]
      +-- Deduplicator ------> [Content Hash | SimHash]
      +-- Retriever ---------> [Semantic | Keyword | Hybrid | Adaptive]
      +-- Reranker ----------> [Cross-Encoder]
      +-- Generator ---------> [OpenAI | Ollama] (with streaming)
      +-- Vector Store ------> [Qdrant | Chroma | FAISS | Pinecone | Weaviate]
      +-- Database ----------> [PostgreSQL | SQLite | MongoDB | MySQL | Supabase]
      +-- Write-Back Manager -> [Protected | Full | Read-Only]
      +-- Hook Manager ------> [Pre/Post Retrieval | Pre/Post Generation]
      |
      +-- Trace Tracker -----> [Memory | Redis | PostgreSQL]
      +-- Metrics Tracker ---> [Prometheus | JSON]
      +-- Telemetry ---------> [OpenTelemetry | Console]
      |
      v
  Web Dashboard (served at /)
      |
      +-- Overview (live metrics + health)
      +-- Pipeline (interactive visualization)
      +-- Debugger (trace inspection)
      +-- Telemetry (charts + tables)
      +-- Plugins (documentation)
```

### Project Structure

```
glassbox-rag/
  src/glassbox_rag/
    __init__.py          # Public API and version
    server.py            # FastAPI app factory with dashboard serving
    config.py            # Pydantic v2 configuration models
    cli.py               # CLI entry point
    core/
      engine.py          # Main orchestrator
      retriever.py       # Retrieval strategies
      chunker.py         # Text chunking
      encoder.py         # Embedding layer with cache
      generator.py       # LLM generation with streaming
      hooks.py           # Pipeline hook system
      dedup.py           # Document deduplication
      tokens.py          # Token counting
      writeback.py       # Write-back protection
    trace/
      tracker.py         # Concurrency-safe trace system
      visualizer.py      # ASCII/HTML trace rendering
      backends.py        # Persistent trace storage
    plugins/             # Plugin implementations
    adapters/            # LangChain, LlamaIndex, Haystack
    ui/
      static/
        app.js           # Dashboard JavaScript
        style.css        # Dashboard styles
      templates/
        dashboard.html   # Dashboard HTML
  config/
    default.yaml         # Default configuration
  tests/                 # Test suite
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
```

---

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and type checks succeed
5. Submit a pull request

---

## License

Licensed under the Apache License 2.0. See the LICENSE file for details.

---

## Links

- **Repository**: [github.com/averoe/GlassBox](https://github.com/averoe/GlassBox)
- **Issues**: [github.com/averoe/GlassBox/issues](https://github.com/averoe/GlassBox/issues)
- **API Docs**: `http://localhost:8000/docs` (when server is running)
- **Dashboard**: `http://localhost:8000` (when server is running)