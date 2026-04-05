"""
Main GlassBox RAG Engine — orchestrates all components.

Coordinates the encoding layer, chunker, retriever, write-back,
metrics, trace, telemetry, multimodal, and reranker systems.
"""

import importlib
from typing import Any, Dict, List, Optional
from uuid import uuid4

from glassbox_rag.config import GlassBoxConfig
from glassbox_rag.core.chunker import (
    Chunk,
    ChunkSizeMonitor,
    ChunkingStats,
    create_chunker,
)
from glassbox_rag.core.encoder import (
    EncoderError,
    EncoderNotAvailableError,
    ModularEncodingLayer,
)
from glassbox_rag.core.metrics import MetricsTracker, OperationType
from glassbox_rag.core.retriever import (
    AdaptiveRetriever,
    Document,
    RetrievalResult,
    RetrieverError,
)
from glassbox_rag.core.writeback import (
    WriteBackManager,
    WriteBackRequest,
    WriteBackResult,
    WriteBackError,
)
from glassbox_rag.plugins.base import DatabasePlugin, VectorStorePlugin
from glassbox_rag.trace.tracker import TraceTracker
from glassbox_rag.utils.logging import get_logger
from glassbox_rag.utils.multimodal import MultimodalHandler, MultimodalContent
from glassbox_rag.utils.telemetry import TelemetryHub

logger = get_logger(__name__)


class EngineError(Exception):
    """Raised when an engine operation fails."""


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

        # Telemetry hub
        self.telemetry = TelemetryHub(
            otel_enabled=config.telemetry.otel_enabled,
            otel_endpoint=config.telemetry.otel_endpoint,
            otel_exporter=config.telemetry.otel_exporter,
            prometheus_enabled=config.telemetry.prometheus_enabled,
            service_name=config.telemetry.service_name,
        )

        # Plugins (set during initialize)
        self.vector_store: Optional[VectorStorePlugin] = None
        self.database: Optional[DatabasePlugin] = None

        # Wired up after plugins load
        self.retriever: Optional[AdaptiveRetriever] = None
        self.writeback_manager: Optional[WriteBackManager] = None
        self.reranker: Optional[Any] = None

        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Async initialization — loads plugins, encoders, reranker, and wires everything."""
        logger.info("Initializing GlassBox engine…")

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

    async def _load_plugin(self, plugin_type: str, name: str, config: Dict) -> Optional[Any]:
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
        encoder: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> tuple:
        """
        Retrieve documents: encode → retrieve → rerank (optional).

        Returns:
            Tuple of (RetrievalResult, trace_data dict).
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

            return result, trace.to_dict()

        except Exception as e:
            self.trace_tracker.end_step(root.step_id, error=str(e))
            self.trace_tracker.end_trace(trace.trace_id)
            self.metrics_tracker.end_request(request_id)
            if self.telemetry.prometheus:
                self.telemetry.prometheus.dec_active()
            logger.error("Retrieve failed: %s", e)
            raise

    # ── Ingest ────────────────────────────────────────────────────

    async def ingest(
        self,
        documents: List[Dict[str, Any]],
        encoder: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            # Step 0: Multimodal extraction (if needed)
            extract_step = self.trace_tracker.start_step("extract_content")
            text_docs: List[Dict[str, Any]] = []

            for i, doc in enumerate(documents):
                content_type = doc.get("content_type", "text")

                if content_type != "text":
                    # Multimodal — extract text
                    mm_content = MultimodalContent(
                        content_type=content_type,
                        content=doc.get("content", ""),
                        metadata=doc.get("metadata", {}),
                    )
                    extracted_texts = await self.multimodal_handler.process(mm_content)
                    for text in extracted_texts:
                        text_docs.append({
                            "content": text,
                            "metadata": {
                                **doc.get("metadata", {}),
                                "source_type": content_type,
                            },
                        })
                else:
                    content = doc.get("content", "")
                    if isinstance(content, str) and content.strip():
                        text_docs.append(doc)
                    elif not isinstance(content, str):
                        # Might be bytes
                        text_docs.append({
                            "content": content.decode("utf-8", errors="replace")
                            if isinstance(content, bytes) else str(content),
                            "metadata": doc.get("metadata", {}),
                        })
                    else:
                        raise EngineError(
                            f"Document at index {i} is missing 'content' field"
                        )

            self.trace_tracker.end_step(extract_step.step_id, output_data={
                "text_documents": len(text_docs),
                "multimodal_converted": len(text_docs) - sum(
                    1 for d in documents if d.get("content_type", "text") == "text"
                ),
            })

            # Step 1: Chunk
            chunk_step = self.trace_tracker.start_step("chunk_documents")
            chunk_op = self.metrics_tracker.start_operation(OperationType.CHUNK)

            all_chunks: List[Chunk] = []
            all_texts: List[str] = []
            all_meta: List[Dict] = []

            for doc in text_docs:
                meta = doc.get("metadata", {})
                chunks, stats = self.chunker.chunk_with_stats(doc["content"], metadata=meta)
                self.chunk_monitor.record(chunks)
                for c in chunks:
                    c.metadata.update(meta)
                    all_chunks.append(c)
                    all_texts.append(c.text)
                    all_meta.append(c.metadata)

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

            doc_ids: List[str] = []
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
            logger.error("Ingest failed: %s", e)
            raise

    # ── Update ────────────────────────────────────────────────────

    async def update(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
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

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        trace = self.trace_tracker.get_trace(trace_id)
        return trace.to_dict() if trace else None

    def list_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.trace_tracker.list_traces(limit)]

    def get_metrics_summary(self) -> Dict[str, Any]:
        return self.metrics_tracker.get_summary()

    def get_chunk_report(self) -> Dict[str, Any]:
        return self.chunk_monitor.get_report()

    def get_telemetry_status(self) -> Dict[str, Any]:
        return self.telemetry.get_telemetry_status()
