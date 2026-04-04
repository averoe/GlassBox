"""
Main GlassBox RAG Engine - orchestrates all components.

Coordinates the encoding layer, retriever, write-back, metrics, and trace system.
"""

from typing import List, Optional, Dict, Any
from uuid import uuid4
import importlib

from glassbox_rag.config import GlassBoxConfig
from glassbox_rag.core.encoder import ModularEncodingLayer
from glassbox_rag.core.retriever import AdaptiveRetriever, RetrievalResult, Document
from glassbox_rag.core.writeback import WriteBackManager, WriteBackRequest, WriteBackResult
from glassbox_rag.core.metrics import MetricsTracker, OperationType
from glassbox_rag.trace.tracker import TraceTracker
from glassbox_rag.plugins.base import VectorStorePlugin, DatabasePlugin


class GlassBoxEngine:
    """
    Main RAG engine that orchestrates all components.

    Provides high-level API for:
    - Document ingestion
    - Retrieval with adaptive strategies
    - Semantic response generation (with LLM integration)
    - Safe write-back operations
    - Complete observability
    """

    def __init__(self, config: GlassBoxConfig):
        """Initialize GlassBox engine."""
        self.config = config
        
        # Initialize core components
        self.encoding_layer = ModularEncodingLayer(config)
        self.retriever = AdaptiveRetriever(config)
        self.writeback_manager = WriteBackManager(config)
        self.metrics_tracker = MetricsTracker(config)
        self.trace_tracker = TraceTracker(config)
        
        # Initialize plugins
        self.vector_store: Optional[VectorStorePlugin] = None
        self.database: Optional[DatabasePlugin] = None
        self._init_plugins()

    def _init_plugins(self):
        """Initialize configured plugins."""
        # Load vector store plugin
        vector_store_type = self.config.vector_store.type
        if vector_store_type:
            try:
                vector_store_config = getattr(self.config.vector_store, vector_store_type, {})
                # Only load if config is not None
                if vector_store_config:
                    self.vector_store = self._load_plugin(
                        "vector_store",
                        vector_store_type,
                        vector_store_config,
                    )
                    if self.vector_store:
                        print(f"✓ Initialized vector store: {vector_store_type}")
            except Exception as e:
                print(f"✗ Failed to initialize vector store: {e}")
        
        # Load database plugin
        database_type = self.config.database.type
        if database_type:
            try:
                database_config = getattr(self.config.database, database_type, {})
                # Only load if config is not None
                if database_config:
                    self.database = self._load_plugin(
                        "database",
                        database_type,
                        database_config,
                    )
                    if self.database:
                        print(f"✓ Initialized database: {database_type}")
            except Exception as e:
                print(f"✗ Failed to initialize database: {e}")

    def _load_plugin(
        self,
        plugin_type: str,
        plugin_name: str,
        config: Dict[str, Any],
    ) -> Optional[Any]:
        """
        Dynamically load a plugin by name.

        Args:
            plugin_type: Type of plugin (vector_store, database, embedder)
            plugin_name: Name of the specific plugin
            config: Configuration for the plugin

        Returns:
            Initialized plugin instance or None if failed.
        """
        try:
            # Map plugin names to modules
            plugin_map = {
                "vector_store": {
                    "qdrant": ("glassbox_rag.plugins.vector_stores.qdrant", "QdrantVectorStore"),
                    "chroma": ("glassbox_rag.plugins.vector_stores.chroma", "ChromaVectorStore"),
                },
                "database": {
                    "postgresql": ("glassbox_rag.plugins.databases.postgresql", "PostgreSQLDatabase"),
                    "sqlite": ("glassbox_rag.plugins.databases.sqlite", "SQLiteDatabase"),
                },
            }

            if plugin_type not in plugin_map or plugin_name not in plugin_map[plugin_type]:
                raise ValueError(f"Unknown {plugin_type}: {plugin_name}")

            module_name, class_name = plugin_map[plugin_type][plugin_name]
            
            # Import module
            module = importlib.import_module(module_name)
            plugin_class = getattr(module, class_name)
            
            # Instantiate plugin
            return plugin_class(config)
        except Exception as e:
            print(f"Failed to load plugin {plugin_type}.{plugin_name}: {e}")
            return None

    async def retrieve(
        self,
        query: str,
        encoder: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> tuple[RetrievalResult, Dict[str, Any]]:
        """
        Retrieve documents based on a query.

        Args:
            query: Query text.
            encoder: Optional encoder to use (defaults to configured default).
            top_k: Optional number of results (defaults to config).

        Returns:
            Tuple of (RetrievalResult, trace_data).
        """
        request_id = str(uuid4())
        
        # Start tracing and metrics
        metrics = self.metrics_tracker.start_request(request_id)
        trace = self.trace_tracker.start_trace(request_id)

        trace_root = self.trace_tracker.start_step(
            name="retrieve",
            input_data={"query": query, "encoder": encoder, "top_k": top_k},
        )

        try:
            # Step 1: Encode query
            encode_step = self.trace_tracker.start_step(
                name="encode_query",
                input_data={"query": query},
            )
            
            encode_metrics = self.metrics_tracker.start_operation(OperationType.ENCODE)
            
            query_embedding, encode_info = await self.encoding_layer.encode_single(
                query,
                encoder_name=encoder,
            )
            
            self.metrics_tracker.end_operation(encode_metrics)
            self.trace_tracker.end_step(
                encode_step.step_id,
                output_data={
                    "embedding_dim": len(query_embedding),
                    "encoder": encode_info["encoder"],
                },
            )

            # Step 2: Retrieve documents
            retrieve_step = self.trace_tracker.start_step(
                name="adaptive_retrieve",
                input_data={"top_k": top_k or self.config.retrieval.top_k},
            )
            
            retrieve_metrics = self.metrics_tracker.start_operation(OperationType.RETRIEVE)
            
            result = await self.retriever.retrieve(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
            )
            
            self.metrics_tracker.end_operation(retrieve_metrics)
            self.trace_tracker.end_step(
                retrieve_step.step_id,
                output_data={
                    "strategy": result.strategy,
                    "num_results": result.total_results,
                    "execution_time_ms": result.execution_time_ms,
                },
            )

            # Finalize
            self.trace_tracker.end_step(trace_root.step_id)
            self.metrics_tracker.end_request(request_id)
            self.trace_tracker.end_trace(trace.trace_id)

            return result, trace.to_dict()

        except Exception as e:
            # Record error
            self.trace_tracker.end_step(
                trace_root.step_id,
                error=str(e),
            )
            self.trace_tracker.end_trace(trace.trace_id)
            raise

    async def ingest(
        self,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Ingest documents into the vector store.

        Args:
            documents: List of document dictionaries with 'content' and optional 'metadata'.

        Returns:
            Ingestion result with document IDs and statistics.
        """
        request_id = str(uuid4())
        
        # Start tracing
        trace = self.trace_tracker.start_trace(request_id)
        ingest_step = self.trace_tracker.start_step(
            name="ingest",
            input_data={"num_documents": len(documents)},
        )

        try:
            # Encode all documents
            encode_step = self.trace_tracker.start_step(
                name="encode_documents",
                input_data={"num_documents": len(documents)},
            )
            
            contents = [doc.get("content", "") for doc in documents]
            embeddings, encode_info = await self.encoding_layer.encode(contents)
            
            self.trace_tracker.end_step(
                encode_step.step_id,
                output_data={"embeddings_shape": embeddings.shape},
            )

            # TODO: Store embeddings in vector store and documents in database
            
            self.trace_tracker.end_step(
                ingest_step.step_id,
                output_data={
                    "documents_ingested": len(documents),
                    "embeddings_shape": embeddings.shape,
                },
            )
            
            self.trace_tracker.end_trace(trace.trace_id)

            return {
                "success": True,
                "documents_ingested": len(documents),
                "trace_id": trace.trace_id,
            }

        except Exception as e:
            self.trace_tracker.end_step(ingest_step.step_id, error=str(e))
            self.trace_tracker.end_trace(trace.trace_id)
            raise

    async def update(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        confidence_score: float = 1.0,
    ) -> WriteBackResult:
        """
        Update an existing document with write-back protection.

        Args:
            document_id: ID of document to update.
            content: New content.
            metadata: Optional metadata.
            confidence_score: Confidence in the update (0-1).

        Returns:
            WriteBackResult with operation status.
        """
        request_id = str(uuid4())
        
        # Start tracing
        trace = self.trace_tracker.start_trace(request_id)
        update_step = self.trace_tracker.start_step(
            name="update",
            input_data={
                "document_id": document_id,
                "confidence_score": confidence_score,
            },
        )

        try:
            # Create write-back request
            wb_request = WriteBackRequest(
                operation="update",
                document_id=document_id,
                content=content,
                metadata=metadata or {},
                confidence_score=confidence_score,
            )

            # Execute write-back
            wb_step = self.trace_tracker.start_step(
                name="execute_writeback",
                input_data={
                    "operation": wb_request.operation,
                    "confidence_score": confidence_score,
                },
            )
            
            result = await self.writeback_manager.execute_write(wb_request)
            
            self.trace_tracker.end_step(
                wb_step.step_id,
                output_data={
                    "success": result.success,
                    "requires_review": result.requires_review,
                },
            )

            self.trace_tracker.end_step(update_step.step_id)
            self.trace_tracker.end_trace(trace.trace_id)

            return result

        except Exception as e:
            self.trace_tracker.end_step(update_step.step_id, error=str(e))
            self.trace_tracker.end_trace(trace.trace_id)
            raise

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a trace by ID."""
        trace = self.trace_tracker.get_trace(trace_id)
        return trace.to_dict() if trace else None

    def list_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List recent traces."""
        traces = self.trace_tracker.list_traces(limit)
        return [t.to_dict() for t in traces]
