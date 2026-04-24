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
    parser = argparse.ArgumentParser(
        prog="glassbox-rag",
        description="GlassBox RAG — A transparent modular RAG framework",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # ── serve ─────────────────────────────────────────────────
    serve_parser = subparsers.add_parser("serve", help="Start the RAG server")
    serve_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to configuration file",
    )
    serve_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")
    serve_parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload"
    )
    serve_parser.add_argument(
        "--workers", type=int, default=4, help="Number of workers"
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

    args = parser.parse_args()

    if args.command == "serve":
        # Lazy import to avoid module-level side effects
        from glassbox_rag.server import run_server

        run_server(
            config_path=args.config,
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
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
    print(f"  2. Run: glassbox-rag serve --config config/default.yaml")


def _generate_config(encoder: str, vector_store: str, database: str) -> str:
    """Generate a project-specific YAML configuration."""
    config = f"""# GlassBox RAG Configuration
# Generated by `glassbox-rag init`

server:
  host: "0.0.0.0"
  port: 8000
  debug: false
  workers: 4

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
        print(f"  Server:       {config.server.host}:{config.server.port}")
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


if __name__ == "__main__":
    main()
