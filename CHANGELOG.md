# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-04-23

### Added — Feature Backlog Implementation (14 features)

#### Performance & Optimization
- **Embedding LRU Cache** — Thread-safe `EmbeddingCache` with SHA-256 keying, configurable `max_size` (default 2048), and batch hit/miss tracking. Integrated into `ModularEncodingLayer.encode()` and `encode_single()` with cache stats in metadata
- **Document Deduplication** — `DocumentDeduplicator` with three strategies: `exact` (SHA-256), `fuzzy` (SimHash), and `semantic` (cosine similarity). Configurable threshold, stateful `is_duplicate()` for streaming, and automatic dedup in `batch_ingest()`
- **Async Batch Ingest with Progress** — `engine.batch_ingest()` with configurable `batch_size`, `on_progress` callback `(batch_num, total, result)`, and automatic deduplication. Reports aggregated results

#### Framework Integration
- **LangChain Adapter** — `GlassBoxRetriever` (BaseRetriever interface) and `GlassBoxEmbeddings` (Embeddings interface) for drop-in use in LangChain chains and agents
- **LlamaIndex Adapter** — `GlassBoxLlamaRetriever` (returns `NodeWithScore`) and `GlassBoxQueryEngine` (returns `Response`) for integration with LlamaIndex pipelines
- **Haystack Adapter** — `GlassBoxHaystackRetriever` component with `run()` method returning Haystack `Document` objects
- **Streaming Retrieve + Generate** — `engine.stream()` async iterator that retrieves context documents, then streams LLM tokens via `AsyncIterator[str]`

#### Runtime Features
- **First-class LLM Generation** — `LLMGenerator` with OpenAI and Ollama backends, `GenerationConfig`, automatic context window budgeting, and `engine.generate()` for end-to-end RAG
- **Query Pipeline Hooks** — `HookManager` with decorator-based `@hooks.on(HookPoint.PRE_RETRIEVE)` registration at 9 pipeline points: `PRE/POST_RETRIEVE`, `PRE/POST_GENERATE`, `PRE/POST_INGEST`, `PRE/POST_RERANK`, `ON_ERROR`
- **Token Counter & Context Window** — `TokenCounter` with tiktoken (accurate) and regex (fallback) backends, `TokenBudget` allocation, `truncate_to_budget()`, and known context windows for 20+ models

#### Developer Experience
- **`GlassBoxEngine.from_env()`** — Zero-config constructor that auto-detects `OPENAI_API_KEY`, `COHERE_API_KEY`, `GOOGLE_API_KEY`, `OLLAMA_BASE_URL`, `QDRANT_HOST`, `PGHOST`, etc.
- **`GlassBoxConfig.from_env()`** — Environment-based configuration builder with intelligent defaults
- **`glassbox-rag init` CLI** — Project scaffolding command generating config YAML, `.env` template, `.gitignore`, and data directory. Supports `--encoder`, `--vector-store`, `--database` options
- **`glassbox-rag check` CLI** — Configuration validator that reports server, encoder, vector store, database, and chunking settings
- **`pip install glassbox-rag[auto]`** — Meta-extra that installs embeddings + generation + tokens + qdrant + sqlite for instant setup
- **`pip install glassbox-rag[adapters]`** — Installs all three framework adapters (LangChain, LlamaIndex, Haystack)
- **`pip install glassbox-rag[generation]`** — Installs OpenAI + tiktoken for LLM generation
- **`pip install glassbox-rag[tokens]`** — Installs tiktoken for accurate token counting

#### Security & Governance
- **`SecretStr` for API Keys** — `SecurityConfig.api_keys` and `AuthConfig.jwt_secret` now use Pydantic `SecretStr` to prevent key leakage in logs, serialization, and tracebacks
- **Rate Limiter Multi-Worker Warning** — Server startup logs a warning when `workers > 1` with memory-based rate limiting, recommending Redis backend

#### Code Quality
- **Type Annotation Modernization** — `from __future__ import annotations` added to all 17+ modules for PEP 604/585 modern syntax support

### Fixed (from 0.3.x)
- Version mismatch — `__init__.__version__` synced to `0.4.0`
- `HybridRetriever.retrieve()` runs in parallel via `asyncio.gather()`
- `engine.retrieve()` returns `RetrievalResult` directly
- `AdaptiveRetriever` reads strategies from config regardless of `adaptive.enabled`
- `CohereEncoder.initialize()` no longer blocks the event loop
- `GoogleEncoder.encode()` reuses a single `ThreadPoolExecutor`
- `MetricsTracker._request_history` uses `OrderedDict` for O(1) eviction
- `SentenceChunker` regex improved for abbreviations
- `RecursiveChunker` char position tracking fixed
- `ImageProcessor` OCR runs in executor
- Prometheus version info updated from stale `0.2.0`

### Changed
- `ModularEncodingLayer.__init__` accepts `cache_size` parameter (default 2048)
- `GlassBoxEngine.__init__` creates `HookManager`, `DocumentDeduplicator`, and `TokenCounter`
- `encode()` and `encode_single()` return cache hit/miss stats in metadata
- Server auth comparisons use `SecretStr.get_secret_value()` for secure comparison
- JWT auth initialization extracts secret via `.get_secret_value()`

## [0.3.1] - 2026-04-05

### Added
- Persistent trace backends (Redis, PostgreSQL)
- Supabase plugin (vector store + database)
- Plugin Developer SDK with decorator-based registration
- Telemetry dashboard integration in `VisualDebugger`
- Prometheus metrics endpoint (`/metrics/prometheus`)
- OpenTelemetry trace export

## [0.3.0] - 2026-04-04

### Added
- Modular encoding layer with OpenAI, Ollama, ONNX, HuggingFace, Cohere, Google encoders
- Adaptive retrieval with strategy selection heuristics
- Cross-encoder reranking (Cohere, sentence-transformers, HuggingFace)
- Multimodal support (PDF, image OCR, PPTX)
- Write-back protection (read-only, protected, full modes)
- JWT + API key authentication
- Redis rate limiting

## [0.2.0] - 2026-04-03

### Added
- Initial public release
- FastAPI server with REST API
- Basic text chunking (fixed, sentence, recursive)
- Qdrant and Chroma vector store plugins
- SQLite and PostgreSQL database plugins
- Metrics tracking with latency histograms
- Trace system with ASCII visualization
