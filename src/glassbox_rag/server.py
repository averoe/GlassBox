"""
FastAPI Server — REST API for GlassBox RAG.

Lazy initialization, JWT+API-key auth, Redis/memory rate limiting,
Prometheus /metrics endpoint, OTel trace export, and integrated
visual debugger with telemetry.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from glassbox_rag.config import GlassBoxConfig, load_config
from glassbox_rag.core.engine import GlassBoxEngine
from glassbox_rag.trace.visualizer import VisualDebugger
from glassbox_rag.utils.logging import get_logger, setup_logging
from glassbox_rag import __version__

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Request / Response Models
# ═══════════════════════════════════════════════════════════════════

class DocumentResult(BaseModel):
    id: str
    content: str
    score: float
    metadata: Dict = Field(default_factory=dict)

class RetrieveRequest(BaseModel):
    query: str
    encoder: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)

class RetrieveResponse(BaseModel):
    success: bool
    query: str
    num_results: int
    strategy: str
    execution_time_ms: float
    documents: list[DocumentResult]
    trace_id: str

class IngestRequest(BaseModel):
    documents: list[dict]

class IngestResponse(BaseModel):
    success: bool
    documents_ingested: int
    chunks_created: int
    trace_id: str

class UpdateRequest(BaseModel):
    document_id: str
    content: str
    metadata: Dict = Field(default_factory=dict)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

class UpdateResponse(BaseModel):
    success: bool
    operation_id: str
    message: str
    requires_review: bool
    review_url: str | None = None

class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict
    warnings: list[str] = []

class GenerateRequest(BaseModel):
    query: str
    encoder: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

class TokenRequest(BaseModel):
    subject: str
    expires_in: int = Field(default=3600, ge=60)


# ═══════════════════════════════════════════════════════════════════
#  App Factory
# ═══════════════════════════════════════════════════════════════════

def create_app(config: GlassBoxConfig | None = None) -> FastAPI:
    """Create FastAPI app with all integrations."""
    if config is None:
        config = GlassBoxConfig()

    setup_logging(level=config.logging.level, log_format=config.logging.format)

    app = FastAPI(
        title="GlassBox RAG",
        description="A transparent, modular RAG framework with integrated telemetry",
        version=__version__,
    )

    engine: GlassBoxEngine | None = None
    redis_rate_limiter = None
    memory_rate_limiter_data: dict[str, list] = defaultdict(list)
    jwt_auth = None

    # ── CORS ──────────────────────────────────────────────────
    if config.security.cors.get("enabled", True):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.security.cors.get("origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ── Lifecycle ─────────────────────────────────────────────

    @app.on_event("startup")
    async def startup():
        nonlocal engine, redis_rate_limiter, jwt_auth

        logger.info("Starting GlassBox server…")
        engine = GlassBoxEngine(config)
        await engine.initialize()

        # Redis rate limiter (if configured)
        if config.security.rate_limit_backend == "redis":
            from glassbox_rag.utils.auth import RedisRateLimiter
            redis_rate_limiter = RedisRateLimiter(
                redis_url=config.security.redis_url,
                max_requests=config.security.rate_limit_rpm,
            )
            if not await redis_rate_limiter.initialize():
                logger.warning("Redis rate limiter unavailable, falling back to memory")
                redis_rate_limiter = None

        # JWT auth (if configured)
        if config.auth.enabled and config.auth.jwt_secret:
            from glassbox_rag.utils.auth import JWTAuth
            jwt_auth = JWTAuth(
                secret=config.auth.jwt_secret.get_secret_value(),
                issuer=config.auth.jwt_issuer,
                audience=config.auth.jwt_audience,
            )
            logger.info("JWT authentication enabled")

        # Multi-worker rate-limiter warning
        if config.server.workers > 1 and config.security.rate_limit_backend == "memory":
            logger.warning(
                "⚠ Memory-based rate limiter is per-process — with %d workers, "
                "effective rate limit is %dx the configured %d RPM. "
                "Use 'rate_limit_backend: redis' for accurate multi-worker limiting.",
                config.server.workers,
                config.server.workers,
                config.security.rate_limit_rpm,
            )

        logger.info("GlassBox server ready on %s:%d", config.server.host, config.server.port)

    @app.on_event("shutdown")
    async def shutdown():
        nonlocal engine, redis_rate_limiter
        if engine:
            await engine.shutdown()
        if redis_rate_limiter:
            await redis_rate_limiter.shutdown()
        logger.info("GlassBox server stopped")

    def get_engine() -> GlassBoxEngine:
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not ready")
        return engine

    # ── Middleware ─────────────────────────────────────────────

    @app.middleware("http")
    async def auth_and_rate_limit(request: Request, call_next):
        # Skip for health, docs, metrics
        skip_paths = {"/health", "/docs", "/openapi.json", "/redoc", "/metrics/prometheus"}
        if request.url.path in skip_paths:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"

        # Rate limiting
        if redis_rate_limiter:
            if not await redis_rate_limiter.is_allowed(client_ip):
                return JSONResponse(status_code=429, content={
                    "detail": "Rate limit exceeded",
                })
        else:
            # Memory rate limiter
            now = time.time()
            window_start = now - 60
            memory_rate_limiter_data[client_ip] = [
                t for t in memory_rate_limiter_data[client_ip] if t > window_start
            ]
            if len(memory_rate_limiter_data[client_ip]) >= config.security.rate_limit_rpm:
                return JSONResponse(status_code=429, content={
                    "detail": "Rate limit exceeded",
                })
            memory_rate_limiter_data[client_ip].append(now)

        # Authentication
        if config.auth.enabled and jwt_auth:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                from glassbox_rag.utils.auth import AuthError
                try:
                    claims = jwt_auth.validate_token(auth_header[7:])
                    request.state.user = claims
                except AuthError as e:
                    return JSONResponse(status_code=401, content={
                        "detail": f"Authentication failed: {e}",
                    })
            elif config.security.api_key_required:
                api_key = request.headers.get("X-API-Key", "")
                if not any(k.get_secret_value() == api_key for k in config.security.api_keys):
                    return JSONResponse(status_code=401, content={
                        "detail": "Invalid or missing authentication",
                    })
            else:
                return JSONResponse(status_code=401, content={
                    "detail": "Authentication required",
                })
        elif config.security.api_key_required:
            api_key = request.headers.get("X-API-Key", "")
            if not any(k.get_secret_value() == api_key for k in config.security.api_keys):
                return JSONResponse(status_code=401, content={
                    "detail": "Invalid API key",
                })

        return await call_next(request)

    # ── Core Routes ───────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        eng = get_engine()
        components: dict[str, str] = {
            "encoder": "operational" if eng.encoding_layer.encoders else "no_encoders",
            "retriever": "operational",
            "writeback": "operational",
            "trace": "operational",
            "telemetry_otel": "enabled" if config.telemetry.otel_enabled else "disabled",
            "telemetry_prometheus": "enabled" if config.telemetry.prometheus_enabled else "disabled",
            "reranker": "operational" if eng.reranker else "disabled",
        }
        if eng.vector_store:
            components["vector_store"] = "operational" if await eng.vector_store.health_check() else "unhealthy"
        else:
            components["vector_store"] = "not_configured"
        if eng.database:
            components["database"] = "operational" if await eng.database.health_check() else "unhealthy"
        else:
            components["database"] = "not_configured"

        all_ok = all(
            v in ("operational", "not_configured", "enabled", "disabled")
            for v in components.values()
        )

        warnings = []
        if config.server.workers > 1 and config.security.rate_limit_backend == "memory":
            warnings.append(
                f"Memory-based rate limiter is per-process — with {config.server.workers} "
                f"workers, effective rate limit is {config.server.workers}x the configured "
                f"{config.security.rate_limit_rpm} RPM. Use 'rate_limit_backend: redis'."
            )

        return HealthResponse(
            status="healthy" if all_ok else "degraded",
            version=__version__, components=components,
            warnings=warnings,
        )

    @app.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve(request: RetrieveRequest):
        try:
            eng = get_engine()
            result = await eng.retrieve(
                query=request.query, encoder=request.encoder, top_k=request.top_k,
            )
            return RetrieveResponse(
                success=True, query=request.query,
                num_results=result.total_results, strategy=result.strategy,
                execution_time_ms=result.execution_time_ms,
                documents=[
                    DocumentResult(id=d.id, content=d.content, score=d.score or 0.0, metadata=d.metadata)
                    for d in result.documents
                ],
                trace_id=result.trace_id,
            )
        except Exception as e:
            logger.error("Retrieve error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(request: IngestRequest):
        try:
            eng = get_engine()
            result = await eng.ingest(documents=request.documents)
            return IngestResponse(
                success=result["success"],
                documents_ingested=result["documents_ingested"],
                chunks_created=result.get("chunks_created", 0),
                trace_id=result["trace_id"],
            )
        except Exception as e:
            logger.error("Ingest error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/stream")
    async def stream_generate(request: GenerateRequest):
        """SSE endpoint for streaming retrieve + generate."""
        import json as json_module

        eng = get_engine()

        async def event_generator():
            try:
                async for event in eng.stream(
                    request.query,
                    encoder=request.encoder,
                    top_k=request.top_k,
                    system_prompt=request.system_prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ):
                    data = json_module.dumps({
                        "type": event.type,
                        "content": event.content,
                        "metadata": event.metadata,
                    })
                    yield f"data: {data}\n\n"
            except Exception as e:
                error_data = json_module.dumps({"type": "error", "content": str(e)})
                yield f"data: {error_data}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.put("/update", response_model=UpdateResponse)
    async def update(request: UpdateRequest):
        try:
            eng = get_engine()
            result = await eng.update(
                document_id=request.document_id, content=request.content,
                metadata=request.metadata, confidence_score=request.confidence_score,
            )
            return UpdateResponse(
                success=result.success, operation_id=result.operation_id,
                message=result.message, requires_review=result.requires_review,
                review_url=result.review_url,
            )
        except Exception as e:
            logger.error("Update error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    # ── Trace & Debug Routes ──────────────────────────────────

    @app.get("/traces/{trace_id}")
    async def get_trace(trace_id: str):
        eng = get_engine()
        trace = eng.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace

    @app.get("/traces")
    async def list_traces(limit: int = Query(100, ge=1, le=1000)):
        eng = get_engine()
        return {"traces": eng.list_traces(limit), "count": len(eng.list_traces(limit))}

    @app.get("/traces/{trace_id}/visualize")
    async def visualize_trace(trace_id: str, format: str = Query("json", regex="^(json|html)$")):
        """Integrated visual debugger with telemetry data."""
        eng = get_engine()
        trace_obj = eng.trace_tracker.get_trace(trace_id)
        if not trace_obj:
            raise HTTPException(status_code=404, detail="Trace not found")

        if format == "html":
            from fastapi.responses import HTMLResponse
            html = VisualDebugger.format_telemetry_dashboard(
                trace_obj,
                telemetry_status=eng.get_telemetry_status(),
                metrics_summary=eng.get_metrics_summary(),
                as_html=True,
            )
            return HTMLResponse(content=html)

        return {
            "trace_id": trace_id,
            "ascii": VisualDebugger.format_trace_ascii(trace_obj),
            "waterfall": VisualDebugger.format_waterfall(trace_obj),
            "cost_breakdown": VisualDebugger.format_cost_breakdown(trace_obj),
            "summary": VisualDebugger.get_trace_summary(trace_obj),
            "telemetry": eng.get_telemetry_status(),
        }

    @app.get("/traces/compare")
    async def compare_traces(trace_a: str, trace_b: str):
        """Compare two traces side-by-side."""
        eng = get_engine()
        a = eng.trace_tracker.get_trace(trace_a)
        b = eng.trace_tracker.get_trace(trace_b)
        if not a or not b:
            raise HTTPException(status_code=404, detail="One or both traces not found")
        return VisualDebugger.compare_traces(a, b)

    @app.get("/traces/anomalies")
    async def detect_anomalies(
        threshold: float = Query(2.0, ge=1.0),
        limit: int = Query(100, ge=10, le=1000),
    ):
        """Detect anomalous steps across recent traces."""
        eng = get_engine()
        traces = eng.trace_tracker.list_traces(limit)
        anomalies = VisualDebugger.detect_anomalies(traces, threshold)
        return {"anomalies": anomalies, "traces_analyzed": len(traces)}

    # ── Metrics & Telemetry Routes ────────────────────────────

    @app.get("/metrics")
    async def get_metrics():
        eng = get_engine()
        summary = eng.get_metrics_summary()
        summary["telemetry"] = eng.get_telemetry_status()
        return summary

    @app.get("/metrics/chunks")
    async def get_chunk_metrics():
        return get_engine().get_chunk_report()

    @app.get("/metrics/prometheus")
    async def prometheus_metrics():
        """Prometheus-scrapable metrics endpoint."""
        eng = get_engine()
        content = eng.telemetry.get_prometheus_metrics()
        content_type = eng.telemetry.get_prometheus_content_type()
        return Response(content=content, media_type=content_type)

    # ── Auth Routes ───────────────────────────────────────────

    @app.post("/auth/token")
    async def create_token(request: TokenRequest):
        """Generate a JWT token (requires auth to be configured)."""
        if not jwt_auth:
            raise HTTPException(status_code=501, detail="JWT auth not configured")
        token = jwt_auth.create_token(
            subject=request.subject, expires_in=request.expires_in,
        )
        return {"token": token, "expires_in": request.expires_in}

    return app


# ═══════════════════════════════════════════════════════════════════
#  Server Runner
# ═══════════════════════════════════════════════════════════════════

def run_server(
    config_path: Path | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 4,
) -> None:
    if config_path and config_path.exists():
        config = load_config(config_path)
    else:
        config = GlassBoxConfig()
        if config_path:
            logger.warning("Config file %s not found, using defaults", config_path)

    setup_logging(level=config.logging.level, log_format=config.logging.format)
    use_reload = reload or config.dev.reload

    logger.info(
        "Starting server: host=%s, port=%d, workers=%d, reload=%s",
        host, port, workers, use_reload,
    )
    uvicorn.run(
        "glassbox_rag.server:create_app",
        host=host, port=port,
        reload=use_reload,
        workers=1 if use_reload else workers,
        log_level=config.logging.level.lower(),
        factory=True,
    )
