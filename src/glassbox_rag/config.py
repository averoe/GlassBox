"""Configuration loading and validation for GlassBox RAG.

Uses Pydantic v2 for strict validation, env-var substitution,
and sensible defaults for all configuration sections.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ConfigDict, field_validator


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


class EncodingConfig(BaseModel):
    """Modular encoding layer configuration."""

    default_encoder: str = "openai"
    local: Dict[str, Any] = Field(default_factory=dict)
    cloud: Dict[str, Any] = Field(default_factory=dict)


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""

    type: str = "qdrant"
    qdrant: Optional[Dict[str, Any]] = None
    chroma: Optional[Dict[str, Any]] = None
    pgvector: Optional[Dict[str, Any]] = None
    supabase: Optional[Dict[str, Any]] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"qdrant", "chroma", "pgvector", "supabase"}
        if v not in allowed:
            raise ValueError(f"Vector store type must be one of {allowed}")
        return v


class DatabaseConfig(BaseModel):
    """Database configuration."""

    type: str = "sqlite"
    postgresql: Optional[Dict[str, Any]] = None
    mysql: Optional[Dict[str, Any]] = None
    sqlite: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"path": "./data/glassbox.db"}
    )
    mongodb: Optional[Dict[str, Any]] = None
    supabase: Optional[Dict[str, Any]] = None

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
    separators: List[str] = Field(
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

    adaptive: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=5, ge=1, le=100)
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)
    rerank_enabled: bool = False
    reranker_type: str = "cross-encoder"  # cohere, cross-encoder, huggingface
    reranker_config: Dict[str, Any] = Field(default_factory=dict)


class MultimodalConfig(BaseModel):
    """Multimodal configuration."""

    enabled: bool = True
    types: List[str] = Field(default_factory=lambda: ["text", "image", "pdf"])
    image: Dict[str, Any] = Field(default_factory=dict)
    pdf: Dict[str, Any] = Field(default_factory=dict)


class WritebackConfig(BaseModel):
    """Write-back configuration."""

    enabled: bool = True
    mode: str = "protected"  # read-only, protected, or full
    protected: Optional[Dict[str, Any]] = None

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
    costs: Dict[str, float] = Field(default_factory=dict)


class SecurityConfig(BaseModel):
    """Security and authentication configuration."""

    api_key_required: bool = False
    api_keys: List[str] = Field(default_factory=list)
    cors: Dict[str, Any] = Field(default_factory=dict)
    rate_limit_rpm: int = Field(default=60, ge=0)
    rate_limit_backend: str = "memory"  # memory or redis
    redis_url: str = "redis://localhost:6379"


class AuthConfig(BaseModel):
    """Authentication configuration."""

    enabled: bool = False
    jwt_secret: Optional[str] = None
    jwt_issuer: Optional[str] = None
    jwt_audience: Optional[str] = None
    jwt_expiry_seconds: int = Field(default=3600, ge=60)


class TelemetryConfig(BaseModel):
    """Telemetry and observability export configuration."""

    otel_enabled: bool = False
    otel_exporter: str = "console"  # console, otlp, jaeger
    otel_endpoint: Optional[str] = None
    prometheus_enabled: bool = False
    service_name: str = "glassbox-rag"


class PluginConfig(BaseModel):
    """Plugin registry configuration."""

    vector_stores: List[Dict[str, str]] = Field(default_factory=list)
    embedders: List[Dict[str, str]] = Field(default_factory=list)
    databases: List[Dict[str, str]] = Field(default_factory=list)
    custom: List[Dict[str, Any]] = Field(default_factory=list)


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
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    dev: DevConfig = Field(default_factory=DevConfig)

    model_config = ConfigDict(extra="allow")


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
