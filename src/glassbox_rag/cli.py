"""CLI entry point for GlassBox RAG.

No module-level side effects — imports server lazily.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    """Main CLI entry point."""
    # Ensure stdout can handle Unicode on Windows (cp1252 → utf-8)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        prog="glassbox-rag",
        description="GlassBox RAG — A transparent modular RAG framework",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # ── tui ───────────────────────────────────────────────────
    tui_parser = subparsers.add_parser(
        "tui", help="Launch the terminal dashboard (TUI)"
    )
    tui_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to configuration file",
    )
    tui_parser.add_argument(
        "--refresh",
        type=int,
        default=5,
        help="Auto-refresh interval in seconds (default: 5)",
    )

    # ── init ──────────────────────────────────────────────────
    init_parser = subparsers.add_parser(
        "init", help="Initialize a new GlassBox RAG project"
    )
    init_parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to initialize (default: current directory)",
    )
    init_parser.add_argument(
        "--encoder",
        choices=["openai", "ollama", "cohere", "google", "onnx"],
        default="openai",
        help="Default encoder to configure",
    )
    init_parser.add_argument(
        "--vector-store",
        choices=["qdrant", "chroma", "faiss", "pinecone"],
        default="qdrant",
        help="Vector store to configure",
    )
    init_parser.add_argument(
        "--database",
        choices=["sqlite", "postgresql"],
        default="sqlite",
        help="Database backend to configure",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files without prompting (CI-safe)",
    )

    # ── version ───────────────────────────────────────────────
    subparsers.add_parser("version", help="Show version")

    # ── check ─────────────────────────────────────────────────
    check_parser = subparsers.add_parser("check", help="Validate configuration")
    check_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to configuration file",
    )

    # ── docs ──────────────────────────────────────────────────
    docs_parser = subparsers.add_parser(
        "docs", help="Interactive documentation topic browser"
    )
    docs_parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="Topic to display (e.g. config, retrieval, plugins, hooks, api)",
    )
    docs_parser.add_argument(
        "--list",
        action="store_true",
        dest="list_topics",
        help="List all available topics",
    )

    # ── query ─────────────────────────────────────────────────
    query_parser = subparsers.add_parser(
        "query", help="Run a retrieval query from the CLI"
    )
    query_parser.add_argument(
        "text",
        help="The query text to search for",
    )
    query_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to configuration file",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    query_parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Encoder to use (default: configured default)",
    )
    query_parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON",
    )

    # ── ingest ────────────────────────────────────────────────
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest files or directories into the pipeline"
    )
    ingest_parser.add_argument(
        "paths",
        nargs="+",
        help="File or directory paths to ingest",
    )
    ingest_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to configuration file",
    )
    ingest_parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Encoder to use (default: configured default)",
    )
    ingest_parser.add_argument(
        "--glob",
        type=str,
        default="*.txt,*.md,*.csv,*.json,*.rst,*.html",
        help="Comma-separated glob patterns for directory scanning (default: *.txt,*.md,*.csv,*.json,*.rst,*.html)",
    )
    ingest_parser.add_argument(
        "--meta",
        type=str,
        action="append",
        default=[],
        help="Metadata key=value pairs (repeatable, e.g. --meta source=web --meta author=me)",
    )
    ingest_parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON",
    )

    # ── health ────────────────────────────────────────────────
    health_parser = subparsers.add_parser(
        "health", help="Run a health check on the engine and its components"
    )
    health_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to configuration file",
    )
    health_parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON",
    )

    # ── export ────────────────────────────────────────────────
    export_parser = subparsers.add_parser(
        "export", help="Export engine metrics, traces, or cache stats"
    )
    export_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to configuration file",
    )
    export_parser.add_argument(
        "--type",
        choices=["metrics", "traces", "cache", "all"],
        default="all",
        dest="export_type",
        help="Type of data to export (default: all)",
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: stdout)",
    )
    export_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        dest="output_format",
        help="Output format (default: json)",
    )

    # ── eval ──────────────────────────────────────────────────
    eval_parser = subparsers.add_parser(
        "eval", help="Evaluation commands (run, list, compare)"
    )
    eval_sub = eval_parser.add_subparsers(dest="eval_command", help="Evaluation subcommands")

    # eval run
    eval_run_parser = eval_sub.add_parser("run", help="Run evaluation on a golden dataset")
    eval_run_parser.add_argument(
        "--dataset", type=Path, required=True,
        help="Path to golden dataset YAML/JSON file",
    )
    eval_run_parser.add_argument(
        "--config", type=Path, default=Path("config/default.yaml"),
        help="Path to configuration file",
    )
    eval_run_parser.add_argument(
        "--metrics", type=str, default="recall@5,precision@5,faithfulness",
        help="Comma-separated metrics (default: recall@5,precision@5,faithfulness)",
    )
    eval_run_parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output results as JSON",
    )

    # eval list
    eval_list_parser = eval_sub.add_parser("list", help="List evaluation reports")
    eval_list_parser.add_argument(
        "--limit", type=int, default=20,
        help="Max reports to show (default: 20)",
    )
    eval_list_parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output results as JSON",
    )

    # eval compare
    eval_cmp_parser = eval_sub.add_parser("compare", help="Compare two evaluation reports")
    eval_cmp_parser.add_argument("report_a", help="First report file")
    eval_cmp_parser.add_argument("report_b", help="Second report file")
    eval_cmp_parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Regression threshold (default: 0.05)",
    )

    # ── experiment ────────────────────────────────────────────
    exp_parser = subparsers.add_parser(
        "experiment", help="Experiment commands (list, record, compare)"
    )
    exp_sub = exp_parser.add_subparsers(dest="exp_command", help="Experiment subcommands")

    # experiment list
    exp_list_parser = exp_sub.add_parser("list", help="List experiments")
    exp_list_parser.add_argument(
        "--limit", type=int, default=20,
        help="Max experiments to show (default: 20)",
    )
    exp_list_parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output results as JSON",
    )

    # experiment record
    exp_rec_parser = exp_sub.add_parser("record", help="Record a new experiment")
    exp_rec_parser.add_argument(
        "--version", type=int, default=0,
        help="Pipeline version (default: auto-detect)",
    )
    exp_rec_parser.add_argument(
        "--dataset", type=str, default="",
        help="Dataset version identifier",
    )
    exp_rec_parser.add_argument(
        "--trace-id", type=str, default="",
        help="Trace ID from a pipeline run",
    )
    exp_rec_parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output results as JSON",
    )

    # experiment compare
    exp_cmp_parser = exp_sub.add_parser("compare", help="Compare two experiments")
    exp_cmp_parser.add_argument("exp_a", help="First experiment ID")
    exp_cmp_parser.add_argument("exp_b", help="Second experiment ID")
    exp_cmp_parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    if args.command == "tui":
        # Lazy import to avoid module-level side effects
        from glassbox_rag.tui.app import run_tui

        run_tui(
            config_path=str(args.config),
            refresh_interval=args.refresh,
        )

    elif args.command == "init":
        _init_project(
            directory=args.directory,
            encoder=args.encoder,
            vector_store=args.vector_store,
            database=args.database,
            force=args.force,
        )

    elif args.command == "version":
        from glassbox_rag import __version__
        print(f"glassbox-rag {__version__}")

    elif args.command == "check":
        _check_config(args.config)

    elif args.command == "docs":
        _docs_browser(topic=args.topic, list_topics=args.list_topics)

    elif args.command == "query":
        _run_query(
            text=args.text,
            config_path=args.config,
            top_k=args.top_k,
            encoder=args.encoder,
            output_json=args.output_json,
        )

    elif args.command == "ingest":
        _run_ingest(
            paths=args.paths,
            config_path=args.config,
            encoder=args.encoder,
            glob_patterns=args.glob,
            meta_pairs=args.meta,
            output_json=args.output_json,
        )

    elif args.command == "health":
        _run_health(
            config_path=args.config,
            output_json=args.output_json,
        )

    elif args.command == "export":
        _run_export(
            config_path=args.config,
            export_type=args.export_type,
            output_path=args.output,
            output_format=args.output_format,
        )

    elif args.command == "eval":
        _handle_eval(args)

    elif args.command == "experiment":
        _handle_experiment(args)

    else:
        parser.print_help()
        sys.exit(1)



def _init_project(
    directory: str,
    encoder: str,
    vector_store: str,
    database: str,
    force: bool = False,
) -> None:
    """Scaffold a new GlassBox RAG project."""
    project_dir = Path(directory).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    config_dir = project_dir / "config"
    config_dir.mkdir(exist_ok=True)

    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Generate config YAML
    config_content = _generate_config(encoder, vector_store, database)
    config_path = config_dir / "default.yaml"
    if config_path.exists() and not force:
        print(f"⚠ Config file already exists: {config_path}")
        response = input("  Overwrite? [y/N] ").strip().lower()
        if response != "y":
            print("  Skipped config file.")
        else:
            config_path.write_text(config_content)
            print(f"✓ Updated config: {config_path}")
    else:
        config_path.write_text(config_content)
        print(f"✓ Created config: {config_path}")

    # Generate .env template
    env_path = project_dir / ".env"
    if not env_path.exists():
        env_content = _generate_env(encoder)
        env_path.write_text(env_content)
        print(f"✓ Created .env template: {env_path}")
    else:
        print(f"  .env already exists — skipped")

    # Generate .gitignore
    gitignore_path = project_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text(
            "# GlassBox RAG\n"
            ".env\n"
            "__pycache__/\n"
            "*.pyc\n"
            "data/\n"
            "dist/\n"
            ".pytest_cache/\n"
            "*.egg-info/\n"
        )
        print(f"✓ Created .gitignore: {gitignore_path}")

    print(f"\n✓ GlassBox RAG project initialized in {project_dir}")
    print(f"  Encoder: {encoder}")
    print(f"  Vector Store: {vector_store}")
    print(f"  Database: {database}")
    print(f"\nNext steps:")
    print(f"  1. Edit .env with your API keys")
    print(f"  2. Run: glassbox-rag tui --config config/default.yaml")


def _generate_config(encoder: str, vector_store: str, database: str) -> str:
    """Generate a project-specific YAML configuration."""
    config = f"""# GlassBox RAG Configuration
# Generated by `glassbox-rag init`

logging:
  level: "INFO"
  format: "text"

trace:
  enabled: true
  backend: "memory"

encoding:
  default_encoder: "{encoder}"
  cloud:
"""

    if encoder == "openai":
        config += """    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "text-embedding-3-small"
      embedding_dim: 1536
"""
    elif encoder == "cohere":
        config += """    cohere:
      api_key: "${COHERE_API_KEY}"
      model: "embed-english-v3.0"
      embedding_dim: 1024
"""
    elif encoder == "google":
        config += """    google:
      api_key: "${GOOGLE_API_KEY}"
      model: "models/text-embedding-004"
      embedding_dim: 768
"""

    if encoder in ("ollama", "onnx"):
        config += f"""  local:
    {encoder}:
"""
        if encoder == "ollama":
            config += """      base_url: "http://localhost:11434"
      model: "nomic-embed-text"
"""
        elif encoder == "onnx":
            config += """      model_path: "models/embeddings/all-MiniLM-L6-v2"
      device: "cpu"
"""

    config += f"""
vector_store:
  type: "{vector_store}"
"""

    if vector_store == "qdrant":
        config += """  qdrant:
    host: "localhost"
    port: 6333
    collection_name: "glassbox_docs"
"""
    elif vector_store == "chroma":
        config += """  chroma:
    path: "./data/chroma"
    collection_name: "glassbox_docs"
"""

    config += f"""
database:
  type: "{database}"
"""

    if database == "sqlite":
        config += """  sqlite:
    path: "./data/glassbox.db"
"""
    elif database == "postgresql":
        config += """  postgresql:
    host: "${PGHOST:localhost}"
    port: 5432
    database: "${PGDATABASE:glassbox_db}"
    user: "${PGUSER:glassbox}"
    password: "${PGPASSWORD}"
"""

    config += """
retrieval:
  top_k: 5
  min_score: 0.3
  adaptive:
    enabled: true
    strategies:
      - name: "semantic"
      - name: "hybrid"
        weight_semantic: 0.6
        weight_keyword: 0.4

chunking:
  strategy: "recursive"
  chunk_size: 512
  chunk_overlap: 50

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
"""

    return config


def _generate_env(encoder: str) -> str:
    """Generate a .env template."""
    lines = [
        "# GlassBox RAG Environment Variables",
        "# Fill in your API keys below",
        "",
    ]

    if encoder == "openai":
        lines.append("OPENAI_API_KEY=sk-your-key-here")
    elif encoder == "cohere":
        lines.append("COHERE_API_KEY=your-key-here")
    elif encoder == "google":
        lines.append("GOOGLE_API_KEY=your-key-here")
    elif encoder == "ollama":
        lines.append("OLLAMA_BASE_URL=http://localhost:11434")

    lines.extend([
        "",
        "# Optional: Database",
        "# PGHOST=localhost",
        "# PGDATABASE=glassbox_db",
        "# PGUSER=glassbox",
        "# PGPASSWORD=",
        "",
        "# Optional: API Security",
        "# GLASSBOX_API_KEY=your-api-key",
        "",
    ])

    return "\n".join(lines)


def _check_config(config_path: Path) -> None:
    """Validate a config file and report issues."""
    from glassbox_rag.config import load_config

    try:
        config = load_config(config_path)
        print(f"✓ Configuration valid: {config_path}")
        print(f"  Encoder:      {config.encoding.default_encoder}")
        print(f"  Vector Store: {config.vector_store.type}")
        print(f"  Database:     {config.database.type}")
        print(f"  Chunking:     {config.chunking.strategy} (size={config.chunking.chunk_size})")
        print(f"  Trace:        {'enabled' if config.trace.enabled else 'disabled'}")
        print(f"  Writeback:    {config.writeback.mode}")
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Invalid configuration: {e}")
        sys.exit(1)


# ── Shared Helpers ────────────────────────────────────────────────


def _init_engine(config_path: Path):
    """Load config and initialise the GlassBox engine synchronously.

    Returns (engine, config) or prints an error and exits.
    """
    import asyncio
    from glassbox_rag.config import load_config
    from glassbox_rag.core.engine import GlassBoxEngine

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Invalid configuration: {e}")
        sys.exit(1)

    engine = GlassBoxEngine(config)
    try:
        asyncio.get_event_loop().run_until_complete(engine.initialize())
    except RuntimeError:
        # No running event loop — create one
        asyncio.run(engine.initialize())
    return engine, config


# ── docs ──────────────────────────────────────────────────────────


_DOCS_TOPICS = {
    "config": """\
GlassBox RAG -- Configuration Guide

  Configuration is stored in YAML files (default: config/default.yaml).
  Supports ${ENV_VAR} and ${ENV_VAR:default} substitution.

  Top-level sections:
    encoding           Encoder setup (model, dimensions, batch size)
    retrieval          Retrieval strategy (adaptive, semantic, hybrid, keyword)
    vector_store       Vector store backend (qdrant, chroma, faiss, pinecone, weaviate)
    database           Document store backend (postgresql, sqlite, mongodb, mysql)
    chunking           Text chunking strategy (recursive, sentence, semantic, fixed)
    generation         LLM backend configuration (openai, ollama)
    writeback          Write-back protection (read-only, protected, full)
    telemetry          OpenTelemetry and Prometheus configuration
    security           Rate limiting and API key settings
    auth               JWT authentication and token settings
    trace              Trace backend (memory, redis, postgresql)
    logging            Log level and format
    metrics            Cost and latency tracking configuration
    evaluation         Evaluation backend, metrics, regression threshold
    versioning         Pipeline and prompt versioning directories
    git                Git integration (auto-commit, branch association)
    query_rewrite      LLM-based query rewriting pre-retrieval
    multi_query        Multi-query expansion with reciprocal rank fusion
    parent_child       Parent-child document retrieval
    context_compression  LLM-based context compression pre-generation
    indexing_extras    Drift detection, freshness validation, lineage tracking
    plugins            Custom plugin loading

  Quick start:
    glassbox-rag init              Create a new project with default config
    glassbox-rag check --config .  Validate your config file
""",
    "retrieval": """\
GlassBox RAG -- Retrieval Strategies

  GlassBox uses an Adaptive Retriever that selects the best strategy
  based on query characteristics.

  Available strategies:
    semantic       Pure vector similarity search
    hybrid         Weighted combination of semantic + keyword (RRF)
    keyword        Full-text keyword search

  Strategy selection heuristics (when adaptive.enabled is true):
    - Short queries (3 words or less) use keyword retrieval
    - Natural language questions use semantic retrieval
    - Ambiguous queries use hybrid retrieval
    - When adaptive is disabled, defaults to hybrid (if available)

  Configuration example:
    retrieval:
      top_k: 5
      min_score: 0.3
      rerank_enabled: false
      adaptive:
        enabled: true
        strategies:
          - name: "semantic"
          - name: "hybrid"
            weight_semantic: 0.6
            weight_keyword: 0.4

  Reranking:
    Set rerank_enabled: true to apply post-retrieval reranking
    via cross-encoder, Cohere, or HuggingFace models.

  Advanced strategies (registered as pipeline hooks):
    query_rewrite          LLM-based query rewriting before retrieval
    multi_query            Multi-query expansion with reciprocal rank fusion
    parent_child           Retrieve child chunks, return parent documents
    context_compression    LLM-based context compression before generation
""",
    "plugins": """\
GlassBox RAG -- Plugin System

  GlassBox has a plugin-based architecture for extensibility.
  The plugin registry supports six plugin types.

  Built-in plugin types:
    vector_store   Similarity search (qdrant, chroma, faiss, pinecone, weaviate, supabase)
    database       Document storage (postgresql, sqlite, mongodb, mysql, supabase)
    encoder        Text embedding providers (openai, ollama, cohere, google, onnx, huggingface)
    reranker       Post-retrieval reranking (cross-encoder, cohere, huggingface)
    chunker        Text chunking strategies (recursive, sentence, semantic, fixed)
    processor      Content processors (PDF, image OCR, PPTX)

  Creating a custom plugin:
    1. Inherit from VectorStorePlugin or DatabasePlugin (from glassbox_rag.plugins.base)
    2. Implement all abstract methods (initialize, shutdown, health_check, ...)
    3. Register via decorator, code, or config:

  Decorator registration:
    from glassbox_rag.plugins.sdk import register_plugin
    @register_plugin("vector_store", "my_store")
    class MyStore(VectorStorePlugin): ...

  Programmatic registration:
    from glassbox_rag.plugins.sdk import plugin_registry
    plugin_registry.register("vector_store", "my_store", MyStoreClass)

  Config-based loading:
    plugins:
      custom:
        - type: vector_store
          name: my_store
          module: mypackage.my_vector_store
          class: MyVectorStore

  Testing:
    from glassbox_rag.plugins.sdk import PluginTestHarness
    results = await PluginTestHarness.test_vector_store(my_instance)
""",
    "hooks": """\
GlassBox RAG -- Lifecycle Hooks

  Hooks let you inject custom logic at key pipeline points.
  Multiple hooks per point execute in priority order (lower runs first).
  Each hook has a configurable timeout (default: 30 seconds).

  Available hook points:
    PRE_RETRIEVE     Before query encoding and retrieval
    POST_RETRIEVE    After retrieval, before returning results
    PRE_GENERATE     Before LLM generation (RAG context ready)
    POST_GENERATE    After LLM response is generated
    PRE_INGEST       Before document ingestion
    POST_INGEST      After document ingestion (drift detector runs here)
    PRE_RERANK       Before reranking pass
    POST_RERANK      After reranking pass
    ON_ERROR         When any pipeline operation fails (errors swallowed)

  Registering hooks (decorator):
    @engine.hooks.on(HookPoint.PRE_RETRIEVE, priority=10)
    async def my_hook(context: dict) -> dict:
        context["query"] = context["query"].lower()
        return context

  Registering hooks (direct):
    engine.hooks.register(HookPoint.POST_RETRIEVE, my_fn, priority=20)

  Removing hooks:
    engine.hooks.unregister(HookPoint.POST_RETRIEVE, my_fn)

  Hook context:
    Each hook receives a dict with operation-specific data
    and must return the (potentially modified) dict.
    Returning None leaves context unchanged (with a warning).
""",
    "api": """\
GlassBox RAG -- API Reference

  Core Engine Methods:

    engine.retrieve(query, top_k=5, encoder=None)
      Execute a retrieval query against the pipeline.
      Returns: RetrievalResult with .documents, .strategy, .execution_time_ms

    engine.ingest(documents, encoder=None)
      Ingest documents into the pipeline.
      Accepts: list of {"content": "...", "metadata": {...}}

    engine.batch_ingest(documents, batch_size=50, encoder=None,
                        on_progress=None, deduplicate=True,
                        max_concurrent_batches=3)
      Concurrent batch ingestion with progress callbacks.

    engine.generate(query, encoder=None, top_k=None, system_prompt=None,
                    temperature=None, max_tokens=None, evaluate=False)
      Retrieve + Generate: end-to-end RAG pipeline.
      Returns: GenerateResponse with .answer, .sources, .evaluation
      Set evaluate=True for inline faithfulness/groundedness scoring.

    engine.stream(query, encoder=None, top_k=None, system_prompt=None,
                  temperature=None, max_tokens=None)
      Streaming retrieve + generate.
      Yields: StreamEvent objects (type: token, metadata, done)

    engine.update(document_id, content, metadata=None, confidence_score=1.0)
      Update a document with write-back protection.
      Returns: WriteBackResult

  Observability Methods:

    engine.get_metrics_summary()    Requests, tokens, cost, histograms
    engine.get_cache_stats()        Embedding cache hit/miss/eviction stats
    engine.get_hook_info()          Registered hooks and their configuration
    engine.get_chunk_report()       Chunk size distribution report
    engine.get_telemetry_status()   OTel and Prometheus status
    engine.list_traces(limit=100)   List execution traces
    engine.get_trace(trace_id)      Get a single trace with full step tree
    engine.get_token_counter()      Access the token counter instance

  Constructors:

    GlassBoxEngine(config)          Standard constructor
    GlassBoxEngine.from_env()       Zero-config (auto-detects env vars)
    GlassBoxConfig.from_env()       Build config from environment variables

  CLI Commands:
    glassbox-rag tui               Launch terminal dashboard
    glassbox-rag query TEXT        Run retrieval from CLI
    glassbox-rag ingest PATH       Ingest files/directories
    glassbox-rag health            Engine health check
    glassbox-rag export            Export metrics/traces
    glassbox-rag docs              This documentation browser
    glassbox-rag eval              Evaluation commands (run, list, compare)
    glassbox-rag experiment        Experiment tracking (list, record, compare)
    glassbox-rag init              Scaffold a new project
    glassbox-rag check             Validate configuration
    glassbox-rag version           Show version
""",
    "telemetry": """\
GlassBox RAG -- Telemetry and Observability

  GlassBox provides built-in observability via OpenTelemetry and Prometheus.

  OpenTelemetry:
    telemetry:
      otel_enabled: true
      otel_exporter: "otlp"          # otlp, jaeger, zipkin, console
      otel_endpoint: "http://localhost:4317"
      service_name: "glassbox-rag"

  Prometheus:
    telemetry:
      prometheus_enabled: true

  Traces:
    Every retrieval, ingest, and generation operation creates
    a detailed trace with step-by-step timing, input/output data,
    and error information.

    trace:
      enabled: true
      backend: "memory"       # memory, redis, postgresql
      retention_days: 30
      sample_rate: 1.0        # 0.0 to 1.0

  Metrics tracked:
    - Total requests, tokens used, estimated cost
    - Per-operation latency percentiles (P50/P95/P99)
    - Embedding cache hit rate
    - Active request count
    - Vector and chunk counts
""",
}


def _docs_browser(topic: str | None = None, list_topics: bool = False) -> None:
    """Interactive documentation topic browser."""
    if list_topics or (topic is None and not list_topics):
        print("GlassBox RAG -- Documentation Topics\n")
        print("  Available topics:\n")
        for key in _DOCS_TOPICS:
            # First non-empty content line as description
            first_line = ""
            for line in _DOCS_TOPICS[key].splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("GlassBox RAG"):
                    first_line = stripped
                    break
            print(f"    {key:<14} {first_line}")
        print(f"\n  Usage: glassbox-rag docs <topic>")
        print(f"  Example: glassbox-rag docs config\n")

        if topic is None and not list_topics:
            # Interactive mode: prompt user to pick a topic
            try:
                choice = input("  Enter a topic (or 'q' to quit): ").strip().lower()
                if choice in _DOCS_TOPICS:
                    print()
                    print(_DOCS_TOPICS[choice])
                elif choice and choice != "q":
                    print(f"\n  ✗ Unknown topic: '{choice}'")
                    print(f"  Available: {', '.join(_DOCS_TOPICS.keys())}")
            except (EOFError, KeyboardInterrupt):
                print()
        return

    key = topic.lower().strip()
    if key in _DOCS_TOPICS:
        print(_DOCS_TOPICS[key])
    else:
        print(f"✗ Unknown topic: '{key}'")
        print(f"  Available topics: {', '.join(_DOCS_TOPICS.keys())}")
        print(f"  Run: glassbox-rag docs --list")
        sys.exit(1)


# ── query ─────────────────────────────────────────────────────────


def _run_query(
    text: str,
    config_path: Path,
    top_k: int,
    encoder: str | None,
    output_json: bool,
) -> None:
    """Run a retrieval query from the CLI."""
    import asyncio
    import json

    engine, config = _init_engine(config_path)

    async def _do_query():
        try:
            result = await engine.retrieve(query=text, top_k=top_k, encoder=encoder)
            return result
        finally:
            await engine.shutdown()

    try:
        result = asyncio.get_event_loop().run_until_complete(_do_query())
    except RuntimeError:
        result = asyncio.run(_do_query())

    if output_json:
        data = {
            "query": text,
            "strategy": result.strategy,
            "total_results": result.total_results,
            "execution_time_ms": result.execution_time_ms,
            "trace_id": result.trace_id,
            "documents": [
                {
                    "id": d.id,
                    "content": d.content[:500],
                    "score": d.score,
                    "metadata": d.metadata if hasattr(d, "metadata") else {},
                }
                for d in result.documents
            ],
        }
        print(json.dumps(data, indent=2, default=str))
    else:
        print(f"\nGlassBox RAG -- Query Results\n")
        print(f"  Query:     {text}")
        print(f"  Strategy:  {result.strategy}")
        print(f"  Results:   {result.total_results}")
        print(f"  Time:      {result.execution_time_ms:.1f}ms")
        if result.trace_id:
            print(f"  Trace ID:  {result.trace_id[:12]}…")
        print()

        if not result.documents:
            print("  📭  No documents found.\n")
        else:
            for i, doc in enumerate(result.documents, 1):
                score_str = f"{doc.score:.4f}" if doc.score else "N/A"
                print(f"  ── Result {i} ── (score: {score_str})")
                # Show first 300 chars of content
                preview = doc.content[:300].replace("\n", " ")
                if len(doc.content) > 300:
                    preview += "…"
                print(f"     {preview}")
                print()


# ── ingest ────────────────────────────────────────────────────────


def _run_ingest(
    paths: list[str],
    config_path: Path,
    encoder: str | None,
    glob_patterns: str,
    meta_pairs: list[str],
    output_json: bool,
) -> None:
    """Ingest files or directories into the pipeline."""
    import asyncio
    import json

    # Parse metadata
    metadata: dict[str, str] = {}
    for pair in meta_pairs:
        if "=" in pair:
            k, v = pair.split("=", 1)
            metadata[k.strip()] = v.strip()
        else:
            print(f"⚠ Ignoring invalid metadata (expected key=value): {pair}")

    # Collect files
    patterns = [p.strip() for p in glob_patterns.split(",") if p.strip()]
    documents: list[dict] = []
    file_count = 0

    for path_str in paths:
        p = Path(path_str)
        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                documents.append({
                    "content": content,
                    "metadata": {
                        **metadata,
                        "source": str(p),
                        "filename": p.name,
                    },
                })
                file_count += 1
                if not output_json:
                    print(f"  📄 {p.name} ({len(content):,} chars)")
            except Exception as e:
                print(f"  ✗ Could not read {p}: {e}")
        elif p.is_dir():
            for pattern in patterns:
                for fp in p.rglob(pattern):
                    if fp.is_file():
                        try:
                            content = fp.read_text(encoding="utf-8", errors="replace")
                            documents.append({
                                "content": content,
                                "metadata": {
                                    **metadata,
                                    "source": str(fp),
                                    "filename": fp.name,
                                },
                            })
                            file_count += 1
                            if not output_json:
                                print(f"  📄 {fp.relative_to(p)} ({len(content):,} chars)")
                        except Exception:
                            pass
        else:
            print(f"  ✗ Path not found: {path_str}")

    if not documents:
        print("✗ No files found to ingest.")
        sys.exit(1)

    if not output_json:
        print(f"\n  Ingesting {file_count} file(s)…")

    engine, config = _init_engine(config_path)

    async def _do_ingest():
        try:
            result = await engine.ingest(documents=documents, encoder=encoder)
            return result
        finally:
            await engine.shutdown()

    try:
        result = asyncio.get_event_loop().run_until_complete(_do_ingest())
    except RuntimeError:
        result = asyncio.run(_do_ingest())

    if output_json:
        # Serialise result (doc_ids may be large, truncate if needed)
        export_data = {
            "success": result.get("success", False),
            "documents_ingested": result.get("documents_ingested", 0),
            "chunks_created": result.get("chunks_created", 0),
            "vectors_stored": len(result.get("document_ids", [])),
            "trace_id": result.get("trace_id", ""),
        }
        print(json.dumps(export_data, indent=2, default=str))
    else:
        if result.get("success"):
            print(f"\n  ✓ Ingestion complete!")
            print(f"    Documents:  {result.get('documents_ingested', 0)}")
            print(f"    Chunks:     {result.get('chunks_created', 0)}")
            print(f"    Vectors:    {len(result.get('document_ids', []))}")
            trace_id = result.get("trace_id", "")
            if trace_id:
                print(f"    Trace ID:   {trace_id[:12]}…")
            print()
        else:
            print(f"\n  ✗ Ingestion failed: {result.get('error', 'unknown error')}")
            sys.exit(1)


# ── health ────────────────────────────────────────────────────────


def _run_health(config_path: Path, output_json: bool) -> None:
    """Run a health check on the engine and its components."""
    import asyncio
    import json

    engine, config = _init_engine(config_path)

    async def _do_health():
        try:
            from glassbox_rag import __version__

            components: dict[str, str] = {}

            # Encoder
            components["encoder"] = (
                "operational" if engine.encoding_layer.encoders else "no_encoders"
            )
            components["retriever"] = "operational"
            components["writeback"] = "operational"
            components["trace"] = "operational"
            components["reranker"] = (
                "operational" if engine.reranker else "disabled"
            )

            # Vector store
            if engine.vector_store:
                try:
                    ok = await engine.vector_store.health_check()
                    components["vector_store"] = "operational" if ok else "unhealthy"
                except Exception:
                    components["vector_store"] = "unhealthy"
            else:
                components["vector_store"] = "not_configured"

            # Database
            if engine.database:
                try:
                    ok = await engine.database.health_check()
                    components["database"] = "operational" if ok else "unhealthy"
                except Exception:
                    components["database"] = "unhealthy"
            else:
                components["database"] = "not_configured"

            # Telemetry
            components["telemetry_otel"] = (
                "enabled" if config.telemetry.otel_enabled else "disabled"
            )
            components["telemetry_prometheus"] = (
                "enabled" if config.telemetry.prometheus_enabled else "disabled"
            )

            all_ok = all(
                v in ("operational", "not_configured", "enabled", "disabled")
                for v in components.values()
            )
            status = "healthy" if all_ok else "degraded"

            return {
                "status": status,
                "version": __version__,
                "components": components,
            }
        finally:
            await engine.shutdown()

    try:
        result = asyncio.get_event_loop().run_until_complete(_do_health())
    except RuntimeError:
        result = asyncio.run(_do_health())

    if output_json:
        print(json.dumps(result, indent=2))
    else:
        status = result["status"]
        icon = "✓" if status == "healthy" else "⚠"
        print(f"\nGlassBox RAG -- Health Check\n")
        print(f"  Status:  {icon} {status}")
        print(f"  Version: {result['version']}\n")
        print("  Components:")
        for name, state in result["components"].items():
            if state == "operational":
                dot = "🟢"
            elif state in ("enabled",):
                dot = "🟢"
            elif state in ("disabled", "not_configured"):
                dot = "⚪"
            elif state == "unhealthy":
                dot = "🔴"
            else:
                dot = "🟡"
            print(f"    {dot}  {name:<24} {state}")
        print()


# ── export ────────────────────────────────────────────────────────


def _run_export(
    config_path: Path,
    export_type: str,
    output_path: Path | None,
    output_format: str,
) -> None:
    """Export engine metrics, traces, or cache stats."""
    import asyncio
    import json

    engine, config = _init_engine(config_path)

    async def _collect():
        try:
            data: dict = {}

            if export_type in ("metrics", "all"):
                data["metrics"] = engine.get_metrics_summary()

            if export_type in ("traces", "all"):
                raw_traces = engine.list_traces(limit=100)
                data["traces"] = {
                    "count": len(raw_traces),
                    "traces": raw_traces,
                }

            if export_type in ("cache", "all"):
                data["cache"] = engine.get_cache_stats()

            if export_type == "all":
                data["telemetry"] = engine.get_telemetry_status()
                data["hooks"] = engine.get_hook_info()
                data["chunks"] = engine.get_chunk_report()

            return data
        finally:
            await engine.shutdown()

    try:
        data = asyncio.get_event_loop().run_until_complete(_collect())
    except RuntimeError:
        data = asyncio.run(_collect())

    # Format output
    if output_format == "json":
        output_text = json.dumps(data, indent=2, default=str)
    else:
        # Human-readable text format
        lines: list[str] = ["GlassBox RAG -- Export\n"]

        if "metrics" in data:
            m = data["metrics"]
            lines.append("  ── Metrics ──")
            lines.append(f"    Total Requests:  {m.get('total_requests', 0)}")
            lines.append(f"    Total Tokens:    {m.get('total_tokens_used', 0)}")
            lines.append(f"    Total Cost:      ${m.get('total_cost_usd', 0):.6f}")
            hists = m.get("histograms", {})
            if hists:
                lines.append("    Per-Operation:")
                for op, h in hists.items():
                    mean = h.get("mean_ms", 0)
                    count = h.get("count", 0)
                    lines.append(f"      {op:<20} mean={mean:.1f}ms  count={count}")
            lines.append("")

        if "traces" in data:
            t = data["traces"]
            lines.append("  ── Traces ──")
            lines.append(f"    Total Traces:  {t.get('count', 0)}")
            for trace in t.get("traces", [])[:10]:
                tid = trace.get("trace_id", trace.get("id", ""))[:12]
                dur = trace.get("duration_ms", 0)
                lines.append(f"      {tid}…  {dur:.1f}ms")
            if t.get("count", 0) > 10:
                lines.append(f"      … and {t['count'] - 10} more")
            lines.append("")

        if "cache" in data:
            c = data["cache"]
            lines.append("  ── Cache ──")
            if c:
                lines.append(f"    Hits:       {c.get('hits', 0)}")
                lines.append(f"    Misses:     {c.get('misses', 0)}")
                lines.append(f"    Size:       {c.get('size', c.get('current_size', 0))}")
                lines.append(f"    Evictions:  {c.get('evictions', 0)}")
            else:
                lines.append("    No cache data available")
            lines.append("")

        output_text = "\n".join(lines)

    # Write output
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"✓ Exported to {output_path}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()


# ── eval ──────────────────────────────────────────────────────────


def _handle_eval(args) -> None:
    """Dispatch evaluation subcommands."""
    import asyncio
    import json

    if args.eval_command == "run":
        from glassbox_rag.evaluation.datasets import GoldenDataset
        from glassbox_rag.evaluation.runner import EvaluationRunner

        engine, config = _init_engine(args.config)
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

        dataset_path = args.dataset
        if not dataset_path.exists():
            print(f"✗ Dataset file not found: {dataset_path}")
            sys.exit(1)

        dataset = GoldenDataset.from_yaml(str(dataset_path))
        runner = EvaluationRunner(
            pipeline=engine,
            eval_config=config.evaluation,
        )

        async def _do_eval():
            try:
                return await runner.run(dataset, metrics=metrics)
            finally:
                await engine.shutdown()

        try:
            report = asyncio.get_event_loop().run_until_complete(_do_eval())
        except RuntimeError:
            report = asyncio.run(_do_eval())

        if args.output_json:
            print(json.dumps(report.to_dict(), indent=2, default=str))
        else:
            print(f"\n{report.summary()}\n")

    elif args.eval_command == "list":
        from pathlib import Path as _P

        eval_dir = _P(".glassbox/evaluations")
        if not eval_dir.exists():
            print("📭  No evaluation reports found.")
            return

        files = sorted(eval_dir.glob("eval_*.json"), reverse=True)[:args.limit]
        if not files:
            print("📭  No evaluation reports found.")
            return

        if args.output_json:
            reports = []
            for f in files:
                try:
                    reports.append(json.loads(f.read_text(encoding="utf-8")))
                except Exception:
                    pass
            print(json.dumps(reports, indent=2, default=str))
        else:
            print(f"\nEvaluation Reports ({len(files)} found)\n")
            for f in files:
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    ts = data.get("timestamp", "")[:19]
                    pv = data.get("pipeline_version", 0)
                    n_q = len(data.get("per_query_scores", []))
                    agg = data.get("aggregate_scores", {})
                    if agg:
                        avg = sum(agg.values()) / len(agg)
                        print(f"  {ts}  v{pv}  {n_q} queries  avg={avg:.4f}")
                    else:
                        print(f"  {ts}  v{pv}  {n_q} queries")
                except Exception:
                    print(f"  {f.name}  (parse error)")
            print()

    elif args.eval_command == "compare":
        report_a_path = Path(args.report_a)
        report_b_path = Path(args.report_b)
        if not report_a_path.exists() or not report_b_path.exists():
            print("✗ Both report files must exist.")
            sys.exit(1)

        from glassbox_rag.evaluation.runner import EvaluationReport

        data_a = json.loads(report_a_path.read_text(encoding="utf-8"))
        data_b = json.loads(report_b_path.read_text(encoding="utf-8"))

        report_a = EvaluationReport(
            aggregate_scores=data_a.get("aggregate_scores", {}),
        )
        report_b = EvaluationReport(
            aggregate_scores=data_b.get("aggregate_scores", {}),
        )

        regression = report_a.compare(report_b, threshold=args.threshold)
        print(f"\n{regression.summary()}\n")

    else:
        print("Usage: glassbox-rag eval {run|list|compare}")
        sys.exit(1)


# ── experiment ────────────────────────────────────────────────────


def _handle_experiment(args) -> None:
    """Dispatch experiment subcommands."""
    import json
    from glassbox_rag.experiments import ExperimentTracker

    tracker = ExperimentTracker.from_workspace()

    if args.exp_command == "list":
        experiments = tracker.list(limit=args.limit)
        if not experiments:
            print("📭  No experiments recorded.")
            return

        if args.output_json:
            print(json.dumps(
                [e.to_dict() for e in experiments],
                indent=2, default=str,
            ))
        else:
            print(f"\nExperiments ({len(experiments)} found)\n")
            for exp in experiments:
                ts = exp.timestamp[:19]
                git = exp.git_commit or "—"
                n_metrics = len(exp.evaluation_results)
                print(
                    f"  {exp.experiment_id}  v{exp.pipeline_version}  "
                    f"git:{git}  {n_metrics} metrics  {ts}"
                )
            print()

    elif args.exp_command == "record":
        pipeline_version = args.version
        if pipeline_version == 0:
            # Auto-detect from .glassbox/versions/current.json
            import json as _json
            current_file = Path(".glassbox/versions/current.json")
            if current_file.exists():
                try:
                    data = _json.loads(current_file.read_text(encoding="utf-8"))
                    pipeline_version = data.get("version", 0)
                except Exception:
                    pass

        record = tracker.record(
            config_version=pipeline_version,
            dataset_version=args.dataset,
            trace_id=args.trace_id,
        )

        if args.output_json:
            print(json.dumps(record.to_dict(), indent=2, default=str))
        else:
            print(f"\n✓ Experiment recorded: {record.experiment_id}")
            print(f"  Pipeline: v{record.pipeline_version}")
            print(f"  Git:      {record.git_commit or '—'}")
            print(f"  Time:     {record.timestamp[:19]}\n")

    elif args.exp_command == "compare":
        try:
            report = tracker.compare(args.exp_a, args.exp_b)
        except ValueError as e:
            print(f"✗ {e}")
            sys.exit(1)

        if args.output_json:
            print(json.dumps(report.to_dict(), indent=2, default=str))
        else:
            print(f"\n{report.summary}\n")
            if report.metric_changes:
                print("  Metric Changes:")
                for metric, info in report.metric_changes.items():
                    icon = "📈" if info.get("improved") else "📉"
                    print(
                        f"    {icon} {metric}: "
                        f"{info['before']:.4f} → {info['after']:.4f} "
                        f"(Δ{info['delta']:+.4f})"
                    )
            print()

    else:
        print("Usage: glassbox-rag experiment {list|record|compare}")
        sys.exit(1)
