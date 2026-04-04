"""
FastAPI Server - REST API for GlassBox RAG.

Exposes endpoints for retrieval, ingestion, updates, and trace inspection.
"""

from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from glassbox_rag.config import load_config, GlassBoxConfig
from glassbox_rag.core.engine import GlassBoxEngine
from glassbox_rag.trace.visualizer import VisualDebugger


# Request/Response Models
class DocumentResult(BaseModel):
    """A single document result."""

    id: str
    content: str
    score: float
    metadata: dict = {}


class RetrieveRequest(BaseModel):
    """Request to retrieve documents."""

    query: str
    encoder: str = "openai"
    top_k: int = 5


class RetrieveResponse(BaseModel):
    """Response with retrieved documents."""

    success: bool
    query: str
    num_results: int
    strategy: str
    execution_time_ms: float
    documents: List[DocumentResult]
    trace_id: str


class IngestRequest(BaseModel):
    """Request to ingest documents."""

    documents: List[dict]


class IngestResponse(BaseModel):
    """Response to ingest request."""

    success: bool
    documents_ingested: int
    trace_id: str


class UpdateRequest(BaseModel):
    """Request to update a document."""

    document_id: str
    content: str
    metadata: dict = {}
    confidence_score: float = 0.0


class UpdateResponse(BaseModel):
    """Response to update request."""

    success: bool
    operation_id: str
    message: str
    requires_review: bool
    review_url: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    components: dict


# Global engine instance
_engine: Optional[GlassBoxEngine] = None


def get_engine() -> GlassBoxEngine:
    """Get the global engine instance."""
    global _engine
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine


def create_app(config: GlassBoxConfig) -> FastAPI:
    """Create FastAPI application."""
    global _engine
    
    # Initialize engine
    _engine = GlassBoxEngine(config)
    
    # Create FastAPI app
    app = FastAPI(
        title="GlassBox RAG",
        description="A transparent, modular RAG framework for AI/ML applications",
        version="0.1.0",
    )

    # Add CORS middleware
    if config.security.cors.get("enabled", True):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.security.cors.get("origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Routes
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            components={
                "encoder": "operational",
                "retriever": "operational",
                "writeback": "operational",
                "trace": "operational",
            },
        )

    @app.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve(request: RetrieveRequest):
        """Retrieve documents based on query."""
        try:
            engine = get_engine()
            result, trace_data = await engine.retrieve(
                query=request.query,
                encoder=request.encoder,
                top_k=request.top_k,
            )

            # Convert documents to response models
            documents = [
                DocumentResult(
                    id=doc.id,
                    content=doc.content,
                    score=doc.score or 0.0,
                    metadata=doc.metadata,
                )
                for doc in result.documents
            ]

            return RetrieveResponse(
                success=True,
                query=request.query,
                num_results=result.total_results,
                strategy=result.strategy,
                execution_time_ms=result.execution_time_ms,
                documents=documents,
                trace_id=trace_data["trace_id"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(request: IngestRequest):
        """Ingest documents into the vector store."""
        try:
            engine = get_engine()
            result = await engine.ingest(documents=request.documents)

            return IngestResponse(
                success=result["success"],
                documents_ingested=result["documents_ingested"],
                trace_id=result["trace_id"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.put("/update", response_model=UpdateResponse)
    async def update(request: UpdateRequest):
        """Update a document with write-back protection."""
        try:
            engine = get_engine()
            result = await engine.update(
                document_id=request.document_id,
                content=request.content,
                metadata=request.metadata,
                confidence_score=request.confidence_score,
            )

            return UpdateResponse(
                success=result.success,
                operation_id=result.operation_id,
                message=result.message,
                requires_review=result.requires_review,
                review_url=result.review_url,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/traces/{trace_id}")
    async def get_trace(trace_id: str):
        """Get trace details by ID."""
        try:
            engine = get_engine()
            trace = engine.get_trace(trace_id)

            if not trace:
                raise HTTPException(status_code=404, detail="Trace not found")

            return trace
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/traces")
    async def list_traces(limit: int = Query(100, le=1000)):
        """List recent traces."""
        try:
            engine = get_engine()
            traces = engine.list_traces(limit)
            return {"traces": traces, "count": len(traces)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/traces/{trace_id}/visualize")
    async def visualize_trace(trace_id: str):
        """Get ASCII visualization of a trace."""
        try:
            engine = get_engine()
            trace_obj = engine.trace_tracker.get_trace(trace_id)

            if not trace_obj:
                raise HTTPException(status_code=404, detail="Trace not found")

            visualization = VisualDebugger.format_trace_ascii(trace_obj)
            return {"trace_id": trace_id, "visualization": visualization}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # UI Routes
    ui_path = Path(__file__).parent.parent.parent / "ui"
    
    # Mount static files
    if (ui_path / "static").exists():
        app.mount("/static", StaticFiles(directory=str(ui_path / "static")), name="static")
    
    @app.get("/")
    async def serve_ui():
        """Serve the dashboard UI."""
        ui_file = ui_path / "templates" / "index.html"
        if ui_file.exists():
            return FileResponse(ui_file)
        return {"message": "GlassBox RAG API. Visit /docs for API documentation."}

    return app


def run_server(
    config_path: Path = Path("config/default.yaml"),
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 4,
):
    """Run the FastAPI server."""
    # Load configuration
    config = load_config(config_path)

    # Create app
    app = create_app(config)

    # Determine if we should use reload
    use_reload = reload or config.dev.reload

    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=use_reload,
        workers=1 if use_reload else workers,
        log_level=config.logging.level.lower(),
    )


# For production use
app = create_app(GlassBoxConfig())  # Default config
