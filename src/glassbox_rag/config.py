"""Configuration loading and validation for GlassBox RAG."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, ConfigDict


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"  # json or text


class TraceConfig(BaseModel):
    """Trace and observability configuration."""

    enabled: bool = True
    backend: str = "memory"  # memory, redis, or postgresql
    retention_days: int = 30
    sample_rate: float = 1.0


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


class DatabaseConfig(BaseModel):
    """Database configuration."""

    type: str = "postgresql"
    postgresql: Optional[Dict[str, Any]] = None
    mysql: Optional[Dict[str, Any]] = None
    sqlite: Optional[Dict[str, Any]] = None
    mongodb: Optional[Dict[str, Any]] = None


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    adaptive: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 5
    min_score: float = 0.3
    rerank_enabled: bool = False


class MultimodalConfig(BaseModel):
    """Multimodal configuration."""

    enabled: bool = True
    types: list = Field(default_factory=lambda: ["text", "image", "pdf"])
    image: Dict[str, Any] = Field(default_factory=dict)
    pdf: Dict[str, Any] = Field(default_factory=dict)


class WritebackConfig(BaseModel):
    """Write-back configuration."""

    enabled: bool = True
    mode: str = "protected"  # read-only, protected, or full
    protected: Optional[Dict[str, Any]] = None


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
    api_keys: list = Field(default_factory=list)
    cors: Dict[str, Any] = Field(default_factory=dict)


class PluginConfig(BaseModel):
    """Plugin registry configuration."""

    vector_stores: list = Field(default_factory=list)
    embedders: list = Field(default_factory=list)
    databases: list = Field(default_factory=list)


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
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)
    writeback: WritebackConfig = Field(default_factory=WritebackConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    dev: DevConfig = Field(default_factory=DevConfig)

    model_config = ConfigDict(extra="allow")  # Allow extra fields for extension


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


def _replace_env_vars(obj: Any) -> Any:
    """
    Recursively replace ${VAR_NAME} with environment variables.

    Args:
        obj: Object to process (dict, list, str, etc.)

    Returns:
        Object with environment variables replaced.
    """
    import os

    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.getenv(var_name, obj)
        return obj
    elif isinstance(obj, dict):
        return {k: _replace_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_replace_env_vars(item) for item in obj]
    else:
        return obj
