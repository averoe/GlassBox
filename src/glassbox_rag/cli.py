"""CLI entry point for GlassBox RAG.

No module-level side effects — imports server lazily.
"""

import argparse
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

    elif args.command == "version":
        print("glassbox-rag 0.2.0")

    elif args.command == "check":
        _check_config(args.config)

    else:
        parser.print_help()
        sys.exit(1)


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
