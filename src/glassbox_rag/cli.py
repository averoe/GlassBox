"""CLI entry point for GlassBox RAG."""

import argparse
import sys
from pathlib import Path

from glassbox_rag.server import app, run_server


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="glassbox-rag",
        description="GlassBox RAG - A transparent modular RAG framework",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the RAG server")
    serve_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to configuration file",
    )
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on file changes",
    )
    serve_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes",
    )
    
    args = parser.parse_args()
    
    if args.command == "serve":
        run_server(
            config_path=args.config,
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
