"""Configuration loading and validation for GlassBox RAG.

Uses Pydantic v2 for strict validation, env-var substitution,
and sensible defaults for all configuration sections.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ConfigDict, SecretStr, field_validator


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = False
    workers: int = Field(default=4, ge=1, le=32)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "text"  # json or text

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"Logging level must be one of {allowed}")
        return v.upper()

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v not in ("json", "text"):
            raise ValueError("Logging format must be 'json' or 'text'")
        return v


class TraceConfig(BaseModel):
    """Trace and observability configuration."""

    enabled: bool = True
    backend: str = "memory"  # memory, redis, or postgresql
    retention_days: int = Field(default=30, ge=1)
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = {"memory", "redis", "postgresql"}
        if v not in allowed:
            raise ValueError(f"Trace backend must be one of {allowed}")
        return v


class OpenAIEncoderConfig(BaseModel):
    api_key: SecretStr | None = None
    model: str = "text-embedding-3-small"
    embedding_dim: int = 1536


class CohereEncoderConfig(BaseModel):
    api_key: SecretStr | None = None
    model: str = "embed-english-v3.0"
    embedding_dim: int = 1024
    input_type: str = "search_document"


class GoogleEncoderConfig(BaseModel):
    api_key: SecretStr | None = None
    model: str = "models/text-embedding-004"
    embedding_dim: int = 768


class CloudEncodersConfig(BaseModel):
    """Cloud embedding providers configuration."""
    
    openai: OpenAIEncoderConfig | None = None
    cohere: CohereEncoderConfig | None = None
    google: GoogleEncoderConfig | None = None
    huggingface: dict[str, Any] | None = None


class EncodingConfig(BaseModel):
    """Modular encoding layer configuration."""

    default_encoder: str = "openai"
    local: dict[str, Any] = Field(default_factory=dict)
    cloud: CloudEncodersConfig = Field(default_factory=CloudEncodersConfig)


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""

    type: str = "qdrant"
    qdrant: dict[str, Any] | None = None
    chroma: dict[str, Any] | None = None
    pgvector: dict[str, Any] | None = None
    supabase: dict[str, Any] | None = None
    faiss: dict[str, Any] | None = None
    pinecone: dict[str, Any] | None = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"qdrant", "chroma", "pgvector", "supabase", "faiss", "pinecone"}
        if v not in allowed:
            raise ValueError(f"Vector store type must be one of {allowed}")
        return v


class DatabaseConfig(BaseModel):
    """Database configuration."""

    type: str = "sqlite"
    postgresql: dict[str, Any] | None = None
    mysql: dict[str, Any] | None = None
    sqlite: dict[str, Any] | None = Field(
        default_factory=lambda: {"path": "./data/glassbox.db"}
    )
    mongodb: dict[str, Any] | None = None
    supabase: dict[str, Any] | None = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"postgresql", "mysql", "sqlite", "mongodb", "supabase"}
        if v not in allowed:
            raise ValueError(f"Database type must be one of {allowed}")
        return v


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""

    strategy: str = "recursive"  # fixed, sentence, recursive, semantic
    chunk_size: int = Field(default=512, ge=64, le=8192)
    chunk_overlap: int = Field(default=50, ge=0)
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )
    track_sizes: bool = True

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"fixed", "sentence", "recursive", "semantic"}
        if v not in allowed:
            raise ValueError(f"Chunking strategy must be one of {allowed}")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info: Any) -> int:
        # info.data may not have chunk_size yet during construction
        return v


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    adaptive: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=5, ge=1, le=100)
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)
    weight_semantic: float = Field(default=0.6, ge=0.0, le=1.0)
    weight_keyword: float = Field(default=0.4, ge=0.0, le=1.0)
    rerank_enabled: bool = False
    reranker_type: str = "cross-encoder"  # cohere, cross-encoder, huggingface
    reranker_config: dict[str, Any] = Field(default_factory=dict)


class MultimodalConfig(BaseModel):
    """Multimodal configuration."""

    enabled: bool = True
    types: list[str] = Field(default_factory=lambda: ["text", "image", "pdf"])
    image: dict[str, Any] = Field(default_factory=dict)
    pdf: dict[str, Any] = Field(default_factory=dict)


class WritebackConfig(BaseModel):
    """Write-back configuration."""

    enabled: bool = True
    mode: str = "protected"  # read-only, protected, or full
    protected: dict[str, Any] | None = None

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = {"read-only", "protected", "full"}
        if v not in allowed:
            raise ValueError(f"Writeback mode must be one of {allowed}")
        return v


class MetricsConfig(BaseModel):
    """Cost and metrics tracking configuration."""

    enabled: bool = True
    track_tokens: bool = True
    track_latency: bool = True
    track_cost: bool = True
    costs: dict[str, float] = Field(default_factory=dict)


class SecurityConfig(BaseModel):
    """Security and authentication configuration."""

    api_key_required: bool = False
    api_keys: list[SecretStr] = Field(default_factory=list)
    cors: dict[str, Any] = Field(default_factory=dict)
    rate_limit_rpm: int = Field(default=60, ge=0)
    rate_limit_backend: str = "memory"  # memory or redis
    redis_url: str = "redis://localhost:6379"


class AuthConfig(BaseModel):
    """Authentication configuration."""

    enabled: bool = False
    jwt_secret: SecretStr | None = None
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    jwt_expiry_seconds: int = Field(default=3600, ge=60)


class TelemetryConfig(BaseModel):
    """Telemetry and observability export configuration."""

    otel_enabled: bool = False
    otel_exporter: str = "console"  # console, otlp, jaeger
    otel_endpoint: str | None = None
    prometheus_enabled: bool = False
    service_name: str = "glassbox-rag"


class GenerationSection(BaseModel):
    """LLM generation configuration."""

    backend: str = ""  # openai, ollama — empty means auto-detect from env
    model: str = "gpt-4o-mini"
    api_key: SecretStr | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    system_prompt: str = (
        "You are a helpful assistant. Answer questions using the provided context. "
        "If you cannot find the answer in the context, say so clearly."
    )

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = {"", "openai", "ollama"}
        if v not in allowed:
            raise ValueError(f"Generation backend must be one of {allowed}")
        return v


class PluginConfig(BaseModel):
    """Plugin registry configuration."""

    vector_stores: list[dict[str, str]] = Field(default_factory=list)
    embedders: list[dict[str, str]] = Field(default_factory=list)
    databases: list[dict[str, str]] = Field(default_factory=list)
    custom: list[dict[str, Any]] = Field(default_factory=list)


class DevConfig(BaseModel):
    """Development settings."""

    reload: bool = True
    trace_all_requests: bool = True


class GlassBoxConfig(BaseModel):
    """Main GlassBox RAG configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    trace: TraceConfig = Field(default_factory=TraceConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)
    writeback: WritebackConfig = Field(default_factory=WritebackConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    generation: GenerationSection = Field(default_factory=GenerationSection)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    dev: DevConfig = Field(default_factory=DevConfig)

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_env(cls) -> "GlassBoxConfig":
        """
        Zero-config constructor — build configuration purely from environment variables.

        Reads GLASSBOX_* environment variables and maps them to config sections:
            GLASSBOX_SERVER_HOST, GLASSBOX_SERVER_PORT,
            GLASSBOX_LOG_LEVEL, GLASSBOX_ENCODER, GLASSBOX_VECTOR_STORE,
            GLASSBOX_DATABASE, OPENAI_API_KEY, COHERE_API_KEY, GOOGLE_API_KEY, etc.

        Returns:
            GlassBoxConfig with settings derived from the environment.
        """
        config_dict: dict[str, Any] = {}

        # Server
        server: dict[str, Any] = {}
        if os.getenv("GLASSBOX_SERVER_HOST"):
            server["host"] = os.environ["GLASSBOX_SERVER_HOST"]
        if os.getenv("GLASSBOX_SERVER_PORT"):
            server["port"] = int(os.environ["GLASSBOX_SERVER_PORT"])
        if os.getenv("GLASSBOX_SERVER_WORKERS"):
            server["workers"] = int(os.environ["GLASSBOX_SERVER_WORKERS"])
        if server:
            config_dict["server"] = server

        # Logging
        if os.getenv("GLASSBOX_LOG_LEVEL"):
            config_dict["logging"] = {"level": os.environ["GLASSBOX_LOG_LEVEL"]}

        # Encoder — auto-detect from available API keys
        encoding: dict[str, Any] = {"cloud": {}, "local": {}}
        default_encoder = os.getenv("GLASSBOX_ENCODER", "")

        if os.getenv("OPENAI_API_KEY"):
            encoding["cloud"]["openai"] = {
                "api_key": os.environ["OPENAI_API_KEY"],
                "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            }
            default_encoder = default_encoder or "openai"

        if os.getenv("COHERE_API_KEY"):
            encoding["cloud"]["cohere"] = {
                "api_key": os.environ["COHERE_API_KEY"],
                "model": os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0"),
            }
            default_encoder = default_encoder or "cohere"

        if os.getenv("GOOGLE_API_KEY"):
            encoding["cloud"]["google"] = {
                "api_key": os.environ["GOOGLE_API_KEY"],
                "model": os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004"),
            }
            default_encoder = default_encoder or "google"

        if os.getenv("HF_API_KEY"):
            encoding["cloud"]["huggingface"] = {
                "api_key": os.environ["HF_API_KEY"],
            }
            default_encoder = default_encoder or "huggingface"

        if os.getenv("OLLAMA_BASE_URL"):
            encoding["local"]["ollama"] = {
                "base_url": os.environ["OLLAMA_BASE_URL"],
                "model": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            }
            default_encoder = default_encoder or "ollama"

        if encoding["cloud"] or encoding["local"]:
            encoding["default_encoder"] = default_encoder or "openai"
            config_dict["encoding"] = encoding

        # Generation — auto-detect from available API keys
        gen: dict[str, Any] = {}
        if os.getenv("OPENAI_API_KEY"):
            gen["backend"] = "openai"
            gen["api_key"] = os.environ["OPENAI_API_KEY"]
            gen["model"] = os.getenv("GLASSBOX_LLM_MODEL", "gpt-4o-mini")
        elif os.getenv("OLLAMA_BASE_URL"):
            gen["backend"] = "ollama"
            gen["base_url"] = os.environ["OLLAMA_BASE_URL"]
            gen["model"] = os.getenv("OLLAMA_LLM_MODEL", "llama3")
        if gen:
            config_dict["generation"] = gen

        # Vector store
        vs_type = os.getenv("GLASSBOX_VECTOR_STORE", "")
        if vs_type:
            vs: dict[str, Any] = {"type": vs_type}
            if vs_type == "qdrant":
                vs["qdrant"] = {
                    "host": os.getenv("QDRANT_HOST", "localhost"),
                    "port": int(os.getenv("QDRANT_PORT", "6333")),
                    "collection_name": os.getenv("QDRANT_COLLECTION", "glassbox_docs"),
                }
            elif vs_type == "chroma":
                vs["chroma"] = {
                    "path": os.getenv("CHROMA_PATH", "./data/chroma"),
                    "collection_name": os.getenv("CHROMA_COLLECTION", "glassbox_docs"),
                }
            elif vs_type == "pinecone":
                vs["pinecone"] = {
                    "api_key": os.getenv("PINECONE_API_KEY", ""),
                    "environment": os.getenv("PINECONE_ENVIRONMENT", ""),
                    "index_name": os.getenv("PINECONE_INDEX", "glassbox"),
                }
            config_dict["vector_store"] = vs

        # Database
        db_type = os.getenv("GLASSBOX_DATABASE", "")
        if db_type:
            db: dict[str, Any] = {"type": db_type}
            if db_type == "postgresql":
                db["postgresql"] = {
                    "host": os.getenv("PGHOST", "localhost"),
                    "port": int(os.getenv("PGPORT", "5432")),
                    "database": os.getenv("PGDATABASE", "glassbox_db"),
                    "user": os.getenv("PGUSER", "glassbox"),
                    "password": os.getenv("PGPASSWORD", ""),
                }
            elif db_type == "sqlite":
                db["sqlite"] = {"path": os.getenv("SQLITE_PATH", "./data/glassbox.db")}
            config_dict["database"] = db

        # Telemetry — pick up standard OTel env vars
        telem: dict[str, Any] = {}
        otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otel_endpoint:
            telem["otel_enabled"] = True
            telem["otel_endpoint"] = otel_endpoint
            telem["otel_exporter"] = "otlp"
        if telem:
            config_dict["telemetry"] = telem

        # Security
        if os.getenv("GLASSBOX_API_KEY"):
            config_dict["security"] = {
                "api_key_required": True,
                "api_keys": [os.environ["GLASSBOX_API_KEY"]],
            }

        return cls(**config_dict)


def load_config(config_path: str | Path) -> GlassBoxConfig:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        GlassBoxConfig instance.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f) or {}

    # Replace environment variables
    config_dict = _replace_env_vars(config_dict)

    return GlassBoxConfig(**config_dict)


# Regex for ${VAR_NAME} or ${VAR_NAME:default_value}
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def _replace_env_vars(obj: Any) -> Any:
    """
    Recursively replace ${VAR_NAME} and ${VAR_NAME:default} patterns
    with environment variable values.

    Supports:
      - ${VAR_NAME}       — replaced with env var or kept as-is if missing
      - ${VAR_NAME:default} — replaced with env var or default if missing
      - Inline substitution: "prefix_${VAR}_suffix"

    Args:
        obj: Object to process (dict, list, str, etc.)

    Returns:
        Object with environment variables replaced.
    """
    if isinstance(obj, str):

        def _replace_match(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)
            env_val = os.getenv(var_name)
            if env_val is not None:
                return env_val
            if default is not None:
                return default
            return match.group(0)  # Keep original if no env var and no default

        return _ENV_VAR_PATTERN.sub(_replace_match, obj)
    elif isinstance(obj, dict):
        return {k: _replace_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_replace_env_vars(item) for item in obj]
    else:
        return obj
