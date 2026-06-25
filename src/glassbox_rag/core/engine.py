"""
Main GlassBox RAG Engine — orchestrates all components.

Coordinates the encoding layer, chunker, retriever, write-back,
metrics, trace, telemetry, multimodal, reranker, hooks, dedup,
and LLM generation systems.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union
from uuid import uuid4

from glassbox_rag.config import GlassBoxConfig
from glassbox_rag.core.chunker import (
    Chunk,
    ChunkSizeMonitor,
    ChunkingStats,
    create_chunker,
)
from glassbox_rag.core.dedup import DocumentDeduplicator
from glassbox_rag.core.encoder import (
    EmbeddingCache,
    EncoderError,
    EncoderNotAvailableError,
    ModularEncodingLayer,
)
from glassbox_rag.core.hooks import HookManager, HookPoint
from glassbox_rag.core.metrics import MetricsTracker, OperationType
from glassbox_rag.core.retriever import (
    AdaptiveRetriever,
    Document,
    RetrievalResult,
    RetrieverError,
)
from glassbox_rag.core.tokens import TokenCounter
from glassbox_rag.core.writeback import (
    WriteBackManager,
    WriteBackRequest,
    WriteBackResult,
    WriteBackError,
)
from glassbox_rag.core.indexing_extras import (
    DriftDetector,
    FreshnessValidator,
    LineageEnricher,
)
from glassbox_rag.plugins.base import DatabasePlugin, VectorStorePlugin
from glassbox_rag.trace.tracker import TraceTracker
from glassbox_rag.utils.logging import get_logger
from glassbox_rag.utils.multimodal import MultimodalHandler, MultimodalContent
from glassbox_rag.utils.telemetry import TelemetryHub

logger = get_logger(__name__)


class EngineError(Exception):
    """Raised when an engine operation fails."""


@dataclass
class InlineEvaluation:
    """Inline evaluation scores attached to a GenerateResponse."""

    faithfulness: float = 0.0
    groundedness: float = 0.0
    context_relevance: float = 0.0


@dataclass
class GenerateResponse:
    """
    Response from GlassBoxEngine.generate().

    Backward-compatible with dict access: response["answer"] still works.
    The .evaluation attribute is only populated when evaluate=True;
    otherwise it is None.
    """

    answer: str
    sources: list[dict[str, Any]]
    retrieval: dict[str, Any]
    generation: dict[str, Any]
    trace_id: str = ""
    pipeline_version: int = 0
    evaluation: InlineEvaluation | None = None

    # ── Dict-like interface for backward compatibility ────────

    def __getitem__(self, key: str) -> Any:
        return self._to_dict()[key]

    def __contains__(self, key: str) -> bool:
        return key in self._to_dict()

    def keys(self):
        return self._to_dict().keys()

    def get(self, key: str, default: Any = None) -> Any:
        return self._to_dict().get(key, default)

    def _to_dict(self) -> dict[str, Any]:
        d = {
            "answer": self.answer,
            "sources": self.sources,
            "retrieval": self.retrieval,
            "generation": self.generation,
            "trace_id": self.trace_id,
            "pipeline_version": self.pipeline_version,
        }
        if self.evaluation is not None:
            d["evaluation"] = {
                "faithfulness": self.evaluation.faithfulness,
                "groundedness": self.evaluation.groundedness,
                "context_relevance": self.evaluation.context_relevance,
            }
        return d


class GlassBoxEngine:
    """
    Main RAG engine that orchestrates all components.

    Provides high-level API for:
    - Document ingestion (text + multimodal, with chunking)
    - Retrieval with adaptive strategies + reranking
    - Safe write-back operations
    - Complete observability via tracing, metrics, and telemetry

    Must be initialized with `await engine.initialize()` before use.
    """

    def __init__(self, config: GlassBoxConfig):
        """Initialize GlassBox engine (lightweight — call initialize() to start)."""
        self.config = config

        # Core components
        self.encoding_layer = ModularEncodingLayer(config)
        self.metrics_tracker = MetricsTracker(config)
        self.trace_tracker = TraceTracker(config)
        self.chunker = create_chunker(config.chunking)
        self.chunk_monitor = ChunkSizeMonitor(target_size=config.chunking.chunk_size)
        self.multimodal_handler = MultimodalHandler()

        # New: Pipeline hooks
        self.hooks = HookManager()

        # New: Document deduplication
        self.deduplicator = DocumentDeduplicator(strategy="exact")

        # New: Token counter
        self.token_counter = TokenCounter()

        # Indexing extras (separated concerns)
        ie = config.indexing_extras
        self._lineage_enricher = LineageEnricher(enabled=ie.lineage_tracking)
        self._drift_detector = DriftDetector(
            threshold=ie.drift_threshold,
            enabled=ie.drift_detection_enabled,
        )
        self._freshness_validator = FreshnessValidator()

        # Telemetry hub
        self.telemetry = TelemetryHub(
            otel_enabled=config.telemetry.otel_enabled,
            otel_endpoint=config.telemetry.otel_endpoint,
            otel_exporter=config.telemetry.otel_exporter,
            prometheus_enabled=config.telemetry.prometheus_enabled,
            service_name=config.telemetry.service_name,
        )

        # Plugins (set during initialize)
        self.vector_store: VectorStorePlugin | None = None
        self.database: DatabasePlugin | None = None

        # Wired up after plugins load
        self.retriever: AdaptiveRetriever | None = None
        self.writeback_manager: WriteBackManager | None = None
        self.reranker: Any | None = None

        # LLM generator (optional — initialized on first use or via config)
        self._generator = None
        self._generator_lock = asyncio.Lock()

        self._initialized = False

    # ── Class constructors ───────────────────────────────────────

    @classmethod
    def from_env(cls) -> "GlassBoxEngine":
        """
        Zero-config constructor — derive all settings from environment.

        Usage:
            engine = GlassBoxEngine.from_env()
            await engine.initialize()
        """
        return cls(GlassBoxConfig.from_env())

    # ── Lifecycle ─────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Async initialization — loads plugins, encoders, reranker, and wires everything."""
        logger.info("Initializing GlassBox engine…")

        # 0. Validate config before doing anything expensive
        self._validate_config()

        # 0. Trace backend (Redis/PostgreSQL persistence)
        await self.trace_tracker.init_backend()

        # 1. Plugin registry + custom plugins
        from glassbox_rag.plugins.sdk import plugin_registry
        plugin_registry.register_builtins()
        if self.config.plugins.custom:
            plugin_registry.load_custom_plugins(self.config.plugins.custom)

        # 2. Plugins
        await self._init_plugins()

        # 3. Encoders
        await self.encoding_layer.initialize()

        # 4. Retriever
        self.retriever = AdaptiveRetriever(
            self.config,
            vector_store=self.vector_store,
            database=self.database,
        )

        # 5. Reranker
        if self.config.retrieval.rerank_enabled:
            await self._init_reranker()

        # 6. Writeback
        self.writeback_manager = WriteBackManager(
            self.config, database=self.database,
        )

        # 7. Register retrieval strategy hooks (Build Directive v3)
        await self._register_strategy_hooks()

        # 8. Register DriftDetector as POST_INGEST hook
        if self.config.indexing_extras.drift_detection_enabled:
            self.hooks.register(
                HookPoint.POST_INGEST, self._drift_detector, priority=10,
            )
            logger.info("✓ Drift detector registered as POST_INGEST hook")

        # 9. Freshness validation — warn if encoder config changed since last run
        if self.config.indexing_extras.freshness_validation:
            enc_cfg = self.config.encoding
            default_enc = enc_cfg.default_encoder
            # Resolve model/dim from cloud or local config
            enc_model = ""
            enc_dim = 0
            cloud_cfg = getattr(enc_cfg.cloud, default_enc, None)
            if cloud_cfg and hasattr(cloud_cfg, "model"):
                enc_model = cloud_cfg.model
                enc_dim = getattr(cloud_cfg, "embedding_dim", 0)
            freshness_report = self._freshness_validator.check(
                encoder_name=default_enc,
                encoder_model=enc_model,
                embedding_dim=enc_dim,
            )
            if not freshness_report["fresh"]:
                logger.warning(
                    "Freshness check: %s", freshness_report["recommendation"],
                )

        self._initialized = True
        logger.info(
            "GlassBox engine ready  "
            "(encoder=%s, vector_store=%s, database=%s, reranker=%s, "
            "otel=%s, prometheus=%s)",
            "yes" if self.encoding_layer.encoders else "none",
            type(self.vector_store).__name__ if self.vector_store else "none",
            type(self.database).__name__ if self.database else "none",
            type(self.reranker).__name__ if self.reranker else "none",
            self.config.telemetry.otel_enabled,
            self.config.telemetry.prometheus_enabled,
        )

    async def shutdown(self) -> None:
        """Gracefully shut down all components."""
        logger.info("Shutting down GlassBox engine…")
        await self.encoding_layer.shutdown()
        if self.vector_store:
            await self.vector_store.shutdown()
        if self.database:
            await self.database.shutdown()
        await self.trace_tracker.shutdown()
        self.telemetry.shutdown()
        self._initialized = False
        logger.info("GlassBox engine shut down")

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise EngineError(
                "Engine not initialized. Call `await engine.initialize()` first."
            )

    def _validate_config(self) -> None:
        """Validate configuration at initialize() time to catch errors early."""
        # Check vector store config
        vs_type = self.config.vector_store.type
        vs_config = getattr(self.config.vector_store, vs_type, None)
        if vs_type and not vs_config:
            raise EngineError(
                f"Vector store type is '{vs_type}' but no '{vs_type}' configuration "
                f"block was found. Add a '{vs_type}:' block under 'vector_store:' in "
                f"your config, or remove 'vector_store.type' to skip vector storage."
            )

        # Check database config
        db_type = self.config.database.type
        db_config = getattr(self.config.database, db_type, None)
        if db_type and not db_config:
            raise EngineError(
                f"Database type is '{db_type}' but no '{db_type}' configuration "
                f"block was found. Add a '{db_type}:' block under 'database:' in "
                f"your config, or remove 'database.type' to skip database."
            )

    # ── Plugin Loading ────────────────────────────────────────────

    async def _init_plugins(self) -> None:
        vs_type = self.config.vector_store.type
        vs_config = getattr(self.config.vector_store, vs_type, None)
        if vs_config:
            self.vector_store = await self._load_plugin("vector_store", vs_type, vs_config)

        db_type = self.config.database.type
        db_config = getattr(self.config.database, db_type, None)
        if db_config:
            self.database = await self._load_plugin("database", db_type, db_config)

    async def _load_plugin(self, plugin_type: str, name: str, config: Dict) -> Any | None:
        # Try plugin registry first
        from glassbox_rag.plugins.sdk import plugin_registry
        registered = plugin_registry.get(plugin_type, name)
        if registered:
            try:
                instance = registered(config)
                if await instance.initialize():
                    logger.info("✓ Loaded %s plugin: %s (via registry)", plugin_type, name)
                    return instance
            except Exception as e:
                logger.warning("Plugin %s/%s from registry failed: %s", plugin_type, name, e)

        # Fallback to direct import map
        plugin_map = {
            "vector_store": {
                "qdrant": ("glassbox_rag.plugins.vector_stores.qdrant", "QdrantVectorStore"),
                "chroma": ("glassbox_rag.plugins.vector_stores.chroma", "ChromaVectorStore"),
                "supabase": ("glassbox_rag.plugins.supabase", "SupabaseVectorStore"),
            },
            "database": {
                "postgresql": ("glassbox_rag.plugins.databases.postgresql", "PostgreSQLDatabase"),
                "sqlite": ("glassbox_rag.plugins.databases.sqlite", "SQLiteDatabase"),
                "supabase": ("glassbox_rag.plugins.supabase", "SupabaseDatabase"),
            },
        }
        if plugin_type not in plugin_map or name not in plugin_map[plugin_type]:
            logger.warning("Unknown plugin %s.%s — skipping", plugin_type, name)
            return None
        module_name, class_name = plugin_map[plugin_type][name]
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            instance = cls(config)
            if await instance.initialize():
                logger.info("✓ Loaded %s plugin: %s", plugin_type, name)
                return instance
            logger.warning("✗ Plugin %s.%s init returned False", plugin_type, name)
            return None
        except ImportError as e:
            logger.warning("✗ Could not import %s plugin '%s': %s", plugin_type, name, e)
            return None
        except Exception as e:
            logger.error("✗ Failed to load %s plugin '%s': %s", plugin_type, name, e)
            return None

    async def _register_strategy_hooks(self) -> None:
        """Register Build Directive v3 retrieval strategy hooks."""
        # These hooks use the generator lazily — they store a reference
        # but only call it when enabled and the generator is ready.
        generator = self._generator  # may be None until first generate()

        if self.config.query_rewrite.enabled:
            from glassbox_rag.core.strategies.query_rewrite import QueryRewriteHook
            hook = QueryRewriteHook(self.config.query_rewrite, generator)
            self.hooks.register(HookPoint.PRE_RETRIEVE, hook, priority=-10)
            logger.info("✓ Query rewrite hook registered")

        if self.config.parent_child.enabled:
            from glassbox_rag.core.strategies.parent_child import ParentChildResolver
            hook = ParentChildResolver(self.config.parent_child, self.database)
            self.hooks.register(HookPoint.POST_RETRIEVE, hook, priority=10)
            logger.info("✓ Parent-child resolver hook registered")

        if self.config.context_compression.enabled:
            from glassbox_rag.core.strategies.context_compression import ContextCompressor
            hook = ContextCompressor(self.config.context_compression, generator)
            self.hooks.register(HookPoint.PRE_GENERATE, hook, priority=10)
            logger.info("✓ Context compression hook registered")

    async def _init_reranker(self) -> None:
        """Initialize the configured reranker."""
        try:
            from glassbox_rag.core.reranker import create_reranker
            self.reranker = await create_reranker(
                self.config.retrieval.reranker_type,
                self.config.retrieval.reranker_config,
            )
            logger.info("✓ Reranker initialized: %s", self.config.retrieval.reranker_type)
        except Exception as e:
            logger.warning("✗ Reranker initialization failed: %s", e)
            self.reranker = None

    # ── Retrieve ──────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        encoder: str | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """
        Retrieve documents: encode → retrieve → rerank (optional).

        Returns:
            RetrievalResult with documents, strategy, timing, and trace_id.
        """
        self._ensure_initialized()

        request_id = str(uuid4())
        self.metrics_tracker.start_request(request_id)
        trace = self.trace_tracker.start_trace(request_id)

        if self.telemetry.prometheus:
            self.telemetry.prometheus.inc_active()

        root = self.trace_tracker.start_step(
            "retrieve", input_data={"query": query, "top_k": top_k},
        )

        try:
            # 0. Pre-retrieve hook
            ctx = await self.hooks.run(HookPoint.PRE_RETRIEVE, {
                "query": query, "encoder": encoder, "top_k": top_k,
            })
            query = ctx.get("query", query)
            encoder = ctx.get("encoder", encoder)
            top_k = ctx.get("top_k", top_k)

            # 1. Encode
            enc_step = self.trace_tracker.start_step("encode_query")
            enc_op = self.metrics_tracker.start_operation(OperationType.ENCODE)

            embedding, enc_info = await self.encoding_layer.encode_single(
                query, encoder_name=encoder,
            )

            self.metrics_tracker.end_operation(enc_op)
            self.telemetry.on_operation_complete("encode", enc_op.get_latency_ms() / 1000)
            self.trace_tracker.end_step(enc_step.step_id, output_data={
                "embedding_dim": len(embedding), "encoder": enc_info["encoder"],
            })

            # 2. Retrieve
            ret_step = self.trace_tracker.start_step("adaptive_retrieve")
            ret_op = self.metrics_tracker.start_operation(OperationType.RETRIEVE)

            result = await self.retriever.retrieve(
                query=query, query_embedding=embedding, top_k=top_k,
            )

            self.metrics_tracker.end_operation(ret_op)
            self.telemetry.on_operation_complete("retrieve", ret_op.get_latency_ms() / 1000)
            self.trace_tracker.end_step(ret_step.step_id, output_data={
                "strategy": result.strategy, "num_results": result.total_results,
            })

            # 3. Rerank (optional)
            if self.reranker and result.documents:
                rr_step = self.trace_tracker.start_step("rerank")
                rr_op = self.metrics_tracker.start_operation(OperationType.RERANK)

                result.documents = await self.reranker.rerank(
                    query=query,
                    documents=result.documents,
                    top_k=top_k or self.config.retrieval.top_k,
                )
                result.total_results = len(result.documents)

                self.metrics_tracker.end_operation(rr_op)
                self.telemetry.on_operation_complete("rerank", rr_op.get_latency_ms() / 1000)
                self.trace_tracker.end_step(rr_step.step_id, output_data={
                    "reranked_count": result.total_results,
                })

            # 4. Post-retrieve hook
            ctx = await self.hooks.run(HookPoint.POST_RETRIEVE, {
                "query": query, "result": result, "documents": result.documents,
            })
            result.documents = ctx.get("documents", result.documents)
            result.total_results = len(result.documents)

            # Finalize
            self.trace_tracker.end_step(root.step_id)
            req_metrics = self.metrics_tracker.end_request(request_id)
            self.trace_tracker.end_trace(trace.trace_id)

            self.telemetry.on_trace_complete(trace)
            self.telemetry.on_request_complete(
                "retrieve", req_metrics.total_latency_ms / 1000,
                req_metrics.total_tokens, req_metrics.total_cost_usd,
            )

            if self.telemetry.prometheus:
                self.telemetry.prometheus.dec_active()

            # Attach trace_id to result for easy downstream access
            result.trace_id = trace.trace_id

            return result

        except Exception as e:
            self.trace_tracker.end_step(root.step_id, error=str(e))
            self.trace_tracker.end_trace(trace.trace_id)
            self.metrics_tracker.end_request(request_id)
            if self.telemetry.prometheus:
                self.telemetry.prometheus.dec_active()

            # Fire error hooks
            await self.hooks.run(HookPoint.ON_ERROR, {
                "operation": "retrieve", "query": query, "error": e,
            })

            logger.error("Retrieve failed: %s", e)
            raise

    # ── Generate (LLM) ────────────────────────────────────────────

    async def generate(
        self,
        query: str,
        *,
        encoder: str | None = None,
        top_k: int | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        evaluate: bool = False,
    ) -> GenerateResponse:
        """
        Retrieve + Generate: end-to-end RAG pipeline.

        Retrieves relevant documents, then generates a response
        using the configured LLM backend.

        Args:
            evaluate: If True, run inline faithfulness and groundedness
                      scoring before returning. Scores are included in
                      response.evaluation.

        Returns:
            GenerateResponse with 'answer', 'sources', 'retrieval',
            'generation' details, and optional 'evaluation' scores.
            Supports dict-like access for backward compatibility.
        """
        self._ensure_initialized()
        await self._ensure_generator()

        # Retrieve
        retrieval_result = await self.retrieve(query, encoder=encoder, top_k=top_k)

        # Extract document texts for context
        context_docs = [doc.content for doc in retrieval_result.documents]

        # Pre-generate hook
        ctx = await self.hooks.run(HookPoint.PRE_GENERATE, {
            "query": query,
            "context_documents": context_docs,
            "retrieval_result": retrieval_result,
        })
        context_docs = ctx.get("context_documents", context_docs)

        # Generate
        gen_result = await self._generator.generate(
            query, context_docs,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Post-generate hook
        ctx = await self.hooks.run(HookPoint.POST_GENERATE, {
            "query": query,
            "answer": gen_result.text,
            "generation_result": gen_result,
            "retrieval_result": retrieval_result,
        })

        answer = ctx.get("answer", gen_result.text)

        # Inline evaluation
        eval_result = None
        if evaluate and self._generator:
            eval_result = await self._run_inline_evaluation(
                query, answer, context_docs,
            )

        # Pipeline version
        pipeline_version = self._get_pipeline_version()

        return GenerateResponse(
            answer=answer,
            sources=[d.to_dict() for d in retrieval_result.documents],
            retrieval=retrieval_result.to_dict(),
            generation=gen_result.to_dict(),
            trace_id=retrieval_result.trace_id,
            pipeline_version=pipeline_version,
            evaluation=eval_result,
        )

    async def stream(
        self,
        query: str,
        *,
        encoder: str | None = None,
        top_k: int | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[Any]:
        """
        Streaming retrieve + generate with structured events.

        Retrieves documents, then streams the LLM response as StreamEvent objects.

        Usage:
            async for event in engine.stream("What is RAG?"):
                if event.type == "token":
                    print(event.content, end="", flush=True)
                elif event.type == "metadata":
                    print(f"Sources: {event.metadata}")
                elif event.type == "done":
                    print(f"\nDone: {event.metadata}")
        """
        self._ensure_initialized()
        await self._ensure_generator()

        from glassbox_rag.core.generator import StreamEvent

        retrieval_result = await self.retrieve(query, encoder=encoder, top_k=top_k)
        context_docs = [doc.content for doc in retrieval_result.documents]

        # Pre-generate hook
        ctx = await self.hooks.run(HookPoint.PRE_GENERATE, {
            "query": query,
            "context_documents": context_docs,
            "retrieval_result": retrieval_result,
        })
        context_docs = ctx.get("context_documents", context_docs)

        # Emit retrieval metadata event
        yield StreamEvent(
            type="metadata",
            metadata={
                "trace_id": retrieval_result.trace_id,
                "strategy": retrieval_result.strategy,
                "num_sources": len(retrieval_result.documents),
                "sources": [d.to_dict() for d in retrieval_result.documents],
            },
        )

        # Stream LLM tokens with trace step
        gen_step = self.trace_tracker.start_step(
            "generate",
            input_data={"query": query, "num_context_docs": len(context_docs)},
        )
        token_count = 0
        async for event in self._generator.stream(
            query, context_docs,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if event.type == "token":
                token_count += 1
            elif event.type == "done":
                self.trace_tracker.end_step(gen_step.step_id, output_data={
                    "tokens_streamed": token_count,
                    "model": event.metadata.get("model") if event.metadata else None,
                })
            yield event

    async def _ensure_generator(self) -> None:
        """Lazily initialize the LLM generator from config (preferred) or env vars."""
        if self._generator is not None:
            return

        async with self._generator_lock:
            # Double-check after acquiring lock
            if self._generator is not None:
                return

            from glassbox_rag.core.generator import LLMGenerator, GenerationConfig

            gen_cfg = self.config.generation
            backend_type = gen_cfg.backend
            backend_config: dict[str, Any] = {}

            # Config-driven (preferred)
            if backend_type and gen_cfg.api_key:
                backend_config["api_key"] = gen_cfg.api_key.get_secret_value()
                backend_config["model"] = gen_cfg.model
                if gen_cfg.base_url:
                    backend_config["base_url"] = gen_cfg.base_url
            elif backend_type == "ollama":
                backend_config["base_url"] = gen_cfg.base_url or "http://localhost:11434"
                backend_config["model"] = gen_cfg.model
            else:
                # Fallback: auto-detect from env vars
                import os
                if os.getenv("OPENAI_API_KEY"):
                    backend_type = "openai"
                    backend_config["api_key"] = os.environ["OPENAI_API_KEY"]
                elif os.getenv("OLLAMA_BASE_URL"):
                    backend_type = "ollama"
                    backend_config["base_url"] = os.environ["OLLAMA_BASE_URL"]
                else:
                    raise EngineError(
                        "No LLM backend configured for generation. "
                        "Set 'generation.backend' in config, or set "
                        "OPENAI_API_KEY / OLLAMA_BASE_URL environment variable."
                    )

            generation_config = GenerationConfig(
                model=gen_cfg.model,
                temperature=gen_cfg.temperature,
                max_tokens=gen_cfg.max_tokens,
                system_prompt=gen_cfg.system_prompt,
            )

            self._generator = LLMGenerator(
                backend_type=backend_type,
                backend_config=backend_config,
                generation_config=generation_config,
            )
            await self._generator.initialize()

    # ── Ingest ────────────────────────────────────────────────────

    async def ingest(
        self,
        documents: list[dict[str, Any]],
        encoder: str | None = None,
    ) -> dict[str, Any]:
        """
        Ingest documents (text or multimodal): extract → chunk → encode → store.

        Supports documents with:
        - {"content": "text…", "metadata": {…}}                — text
        - {"content": <bytes>, "content_type": "pdf", …}       — multimodal
        - {"content": <bytes>, "content_type": "image", …}     — multimodal
        """
        self._ensure_initialized()

        if not documents:
            raise EngineError("No documents provided for ingestion")

        request_id = str(uuid4())
        trace = self.trace_tracker.start_trace(request_id)
        self.metrics_tracker.start_request(request_id)

        if self.telemetry.prometheus:
            self.telemetry.prometheus.inc_active()

        ingest_step = self.trace_tracker.start_step(
            "ingest", input_data={"num_documents": len(documents)},
        )

        try:
            # Step 0: Multimodal extraction (if needed) — run concurrently
            extract_step = self.trace_tracker.start_step("extract_content")
            text_docs: list[dict[str, Any]] = []

            async def _extract_doc(i: int, doc: dict[str, Any]) -> list[dict[str, Any]]:
                content_type = doc.get("content_type", "text")
                if content_type != "text":
                    mm_content = MultimodalContent(
                        content_type=content_type,
                        content=doc.get("content", ""),
                        metadata=doc.get("metadata", {}),
                    )
                    extracted_texts = await self.multimodal_handler.process(mm_content)
                    return [{
                        "content": text,
                        "metadata": {
                            **doc.get("metadata", {}),
                            "source_type": content_type,
                        },
                    } for text in extracted_texts]
                else:
                    content = doc.get("content", "")
                    if isinstance(content, str) and content.strip():
                        return [doc]
                    elif not isinstance(content, str):
                        return [{
                            "content": content.decode("utf-8", errors="replace")
                            if isinstance(content, bytes) else str(content),
                            "metadata": doc.get("metadata", {}),
                        }]
                    else:
                        raise EngineError(
                            f"Document at index {i} is missing 'content' field"
                        )

            # Run all extractions concurrently
            nested = await asyncio.gather(
                *[_extract_doc(i, d) for i, d in enumerate(documents)]
            )
            text_docs = [doc for group in nested for doc in group]

            self.trace_tracker.end_step(extract_step.step_id, output_data={
                "text_documents": len(text_docs),
                "multimodal_converted": len(text_docs) - sum(
                    1 for d in documents if d.get("content_type", "text") == "text"
                ),
            })

            # Step 1: Chunk
            chunk_step = self.trace_tracker.start_step("chunk_documents")
            chunk_op = self.metrics_tracker.start_operation(OperationType.CHUNK)

            all_chunks: list[Chunk] = []
            all_texts: list[str] = []
            all_meta: list[Dict] = []

            for doc in text_docs:
                meta = doc.get("metadata", {})
                chunks, stats = self.chunker.chunk_with_stats(doc["content"], metadata=meta)
                self.chunk_monitor.record(chunks)
                for c in chunks:
                    c.metadata.update(meta)
                    all_chunks.append(c)
                    all_texts.append(c.text)
                    all_meta.append(c.metadata)

            # Lineage enrichment (runs at chunker layer, not inline in engine)
            for doc in text_docs:
                source_content = doc.get("content", "")
                doc_meta = doc.get("metadata", {})
                doc_chunks = [
                    c for c in all_chunks
                    if c.metadata.get("source") == doc_meta.get("source", "")
                ]
                if doc_chunks:
                    self._lineage_enricher.enrich(
                        doc_chunks,
                        source_content=source_content,
                        parent_doc_id=doc_meta.get("source", ""),
                    )

            self.metrics_tracker.end_operation(chunk_op)
            self.telemetry.on_operation_complete("chunk", chunk_op.get_latency_ms() / 1000)
            self.trace_tracker.end_step(chunk_step.step_id, output_data={
                "total_chunks": len(all_chunks),
                "avg_size": round(sum(c.size for c in all_chunks) / max(len(all_chunks), 1)),
            })

            # Step 2: Encode
            enc_step = self.trace_tracker.start_step("encode_chunks")
            enc_op = self.metrics_tracker.start_operation(OperationType.ENCODE)

            embeddings, enc_info = await self.encoding_layer.encode(
                all_texts, encoder_name=encoder,
            )

            self.metrics_tracker.end_operation(enc_op)
            self.telemetry.on_operation_complete("encode", enc_op.get_latency_ms() / 1000)
            self.trace_tracker.end_step(enc_step.step_id, output_data=enc_info)

            # Step 3: Store
            store_step = self.trace_tracker.start_step("store_vectors")
            store_op = self.metrics_tracker.start_operation(OperationType.INGEST)

            doc_ids: list[str] = []
            if self.vector_store:
                doc_ids = await self.vector_store.add_vectors(
                    vectors=embeddings.tolist(),
                    contents=all_texts,
                    metadata=all_meta,
                )
            else:
                logger.warning("No vector store — embeddings not persisted")
                doc_ids = [str(uuid4()) for _ in all_chunks]

            if self.database:
                for chunk, doc_id in zip(all_chunks, doc_ids):
                    await self.database.insert("documents", {
                        "id": doc_id, "content": chunk.text, "metadata": chunk.metadata,
                    })

            self.metrics_tracker.end_operation(store_op)
            self.telemetry.on_operation_complete("store", store_op.get_latency_ms() / 1000)
            self.trace_tracker.end_step(store_step.step_id, output_data={
                "vectors_stored": len(doc_ids),
            })

            if self.telemetry.prometheus:
                self.telemetry.prometheus.set_chunks_count(len(all_chunks))
                if self.vector_store:
                    count = await self.vector_store.count()
                    self.telemetry.prometheus.set_vectors_count(count)

            # POST_INGEST hook (DriftDetector runs here if registered)
            import numpy as _np
            await self.hooks.run(HookPoint.POST_INGEST, {
                "embeddings": _np.array(embeddings) if not isinstance(embeddings, _np.ndarray) else embeddings,
                "chunk_count": len(all_chunks),
                "document_ids": doc_ids,
            })

            # Finalize
            self.trace_tracker.end_step(ingest_step.step_id)
            req_metrics = self.metrics_tracker.end_request(request_id)
            self.trace_tracker.end_trace(trace.trace_id)

            self.telemetry.on_trace_complete(trace)
            self.telemetry.on_request_complete(
                "ingest", req_metrics.total_latency_ms / 1000,
            )
            if self.telemetry.prometheus:
                self.telemetry.prometheus.dec_active()

            result = {
                "success": True,
                "documents_ingested": len(documents),
                "chunks_created": len(all_chunks),
                "document_ids": doc_ids,
                "trace_id": trace.trace_id,
                "chunk_monitor": self.chunk_monitor.get_report(),
            }
            logger.info(
                "Ingested %d docs → %d chunks → %d vectors",
                len(documents), len(all_chunks), len(doc_ids),
            )
            return result

        except Exception as e:
            self.trace_tracker.end_step(ingest_step.step_id, error=str(e))
            self.trace_tracker.end_trace(trace.trace_id)
            self.metrics_tracker.end_request(request_id)
            if self.telemetry.prometheus:
                self.telemetry.prometheus.dec_active()
            await self.hooks.run(HookPoint.ON_ERROR, {
                "operation": "ingest", "error": e,
            })
            logger.error("Ingest failed: %s", e)
            raise

    # ── Batch Ingest (with progress) ──────────────────────────────

    async def batch_ingest(
        self,
        documents: list[dict[str, Any]],
        *,
        batch_size: int = 50,
        encoder: str | None = None,
        on_progress: Optional[
            Union[
                Callable[[int, int, dict[str, Any]], None],
                Callable[[int, int, dict[str, Any]], Any],  # async callable
            ]
        ] = None,
        deduplicate: bool = True,
        max_concurrent_batches: int = 3,
    ) -> dict[str, Any]:
        """
        Async batch ingest with progress reporting.

        Splits documents into batches and reports progress after each batch.

        Args:
            documents: Full list of documents to ingest.
            batch_size: Number of documents per batch.
            encoder: Encoder name override.
            on_progress: Callback(batch_num, total_batches, batch_result).
            deduplicate: Whether to run deduplication before ingest.

        Returns:
            Aggregated ingest result.
        """
        self._ensure_initialized()

        if not documents:
            raise EngineError("No documents provided for batch ingestion")

        # Deduplicate
        dupes: list = []
        if deduplicate:
            unique_docs, dupes = self.deduplicator.deduplicate(documents)
            if dupes:
                logger.info(
                    "Batch ingest: removed %d duplicates, %d unique documents remain",
                    len(dupes), len(unique_docs),
                )
            documents = unique_docs

        total_batches = math.ceil(len(documents) / batch_size)
        all_results: list[dict[str, Any]] = []
        total_chunks = 0
        total_doc_ids: list[str] = []

        # Use semaphore for concurrent batch processing
        sem = asyncio.Semaphore(max_concurrent_batches)

        async def run_batch(batch_num: int, batch: list[dict[str, Any]]) -> dict[str, Any]:
            async with sem:
                result = await self.ingest(batch, encoder=encoder)
                if on_progress:
                    if inspect.iscoroutinefunction(on_progress):
                        await on_progress(batch_num + 1, total_batches, result)
                    else:
                        on_progress(batch_num + 1, total_batches, result)
                logger.info(
                    "Batch %d/%d ingested: %d docs → %d chunks",
                    batch_num + 1, total_batches,
                    len(batch), result.get("chunks_created", 0),
                )
                return result

        batches = [
            documents[i:i + batch_size]
            for i in range(0, len(documents), batch_size)
        ]

        results = await asyncio.gather(
            *[run_batch(i, b) for i, b in enumerate(batches)]
        )

        for result in results:
            all_results.append(result)
            total_chunks += result.get("chunks_created", 0)
            total_doc_ids.extend(result.get("document_ids", []))

        return {
            "success": True,
            "documents_ingested": len(documents),
            "chunks_created": total_chunks,
            "document_ids": total_doc_ids,
            "batches": total_batches,
            "duplicates_removed": len(dupes) if deduplicate else 0,
        }

    # ── Update ────────────────────────────────────────────────────

    async def update(
        self,
        document_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        confidence_score: float = 1.0,
    ) -> WriteBackResult:
        """Update a document with write-back protection."""
        self._ensure_initialized()

        request_id = str(uuid4())
        trace = self.trace_tracker.start_trace(request_id)
        self.metrics_tracker.start_request(request_id)

        step = self.trace_tracker.start_step("update", input_data={
            "document_id": document_id, "confidence_score": confidence_score,
        })

        try:
            wb_request = WriteBackRequest(
                operation="update", document_id=document_id,
                content=content, metadata=metadata or {},
                confidence_score=confidence_score,
            )

            wb_step = self.trace_tracker.start_step("execute_writeback")
            result = await self.writeback_manager.execute_write(wb_request)
            self.trace_tracker.end_step(wb_step.step_id, output_data={
                "success": result.success, "requires_review": result.requires_review,
            })

            self.trace_tracker.end_step(step.step_id)
            req_metrics = self.metrics_tracker.end_request(request_id)
            self.trace_tracker.end_trace(trace.trace_id)

            self.telemetry.on_trace_complete(trace)
            self.telemetry.on_request_complete(
                "update", req_metrics.total_latency_ms / 1000,
            )
            return result

        except Exception as e:
            self.trace_tracker.end_step(step.step_id, error=str(e))
            self.trace_tracker.end_trace(trace.trace_id)
            self.metrics_tracker.end_request(request_id)
            logger.error("Update failed for doc %s: %s", document_id, e)
            raise

    # ── Observability ─────────────────────────────────────────────

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        trace = self.trace_tracker.get_trace(trace_id)
        return trace.to_dict() if trace else None

    def list_traces(self, limit: int = 100) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.trace_tracker.list_traces(limit)]

    def get_metrics_summary(self) -> dict[str, Any]:
        return self.metrics_tracker.get_summary()

    def get_chunk_report(self) -> dict[str, Any]:
        return self.chunk_monitor.get_report()

    def get_telemetry_status(self) -> dict[str, Any]:
        return self.telemetry.get_telemetry_status()

    def get_cache_stats(self) -> dict[str, Any]:
        """Return embedding cache statistics."""
        return self.encoding_layer.cache.stats()

    def get_hook_info(self) -> dict[str, Any]:
        """Return registered hook information."""
        return {
            "total_hooks": self.hooks.total_hooks,
            "hooks": self.hooks.list_hooks(),
        }

    def get_token_counter(self) -> TokenCounter:
        """Return the engine's token counter."""
        return self.token_counter

    # ── Build Directive v3 helpers ────────────────────────────────

    async def _run_inline_evaluation(
        self,
        query: str,
        answer: str,
        context_docs: list[str],
    ) -> InlineEvaluation:
        """Run inline faithfulness and groundedness scoring."""
        from glassbox_rag.evaluation.generation import (
            faithfulness,
            groundedness,
            context_relevance,
        )

        faith_score = 0.0
        ground_score = 0.0
        cr_score = 0.0

        try:
            faith_score = await faithfulness(answer, context_docs, self._generator)
        except Exception as e:
            logger.warning("Inline faithfulness eval failed: %s", e)

        try:
            ground_score = await groundedness(answer, context_docs, self._generator)
        except Exception as e:
            logger.warning("Inline groundedness eval failed: %s", e)

        try:
            cr_score = await context_relevance(query, context_docs, self._generator)
        except Exception as e:
            logger.warning("Inline context_relevance eval failed: %s", e)

        return InlineEvaluation(
            faithfulness=faith_score,
            groundedness=ground_score,
            context_relevance=cr_score,
        )

    def _get_pipeline_version(self) -> int:
        """Get current pipeline version from .glassbox/versions/."""
        import json as json_mod
        current_file = Path(".glassbox/versions/current.json")
        if current_file.exists():
            try:
                data = json_mod.loads(current_file.read_text(encoding="utf-8"))
                return data.get("version", 0)
            except Exception:
                pass
        return 0

