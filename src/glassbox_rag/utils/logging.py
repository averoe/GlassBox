"""Structured logging setup for GlassBox RAG.

Provides JSON and text formatters, and a consistent logger factory
so every module uses structured logging instead of print().
"""

from __future__ import annotations

import logging
import json
import sys
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include extra fields if present
        for key in ("trace_id", "request_id", "step_name", "component"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable colored text formatter."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        return (
            f"{color}{timestamp} [{record.levelname:<8}]{self.RESET} "
            f"{record.name}: {record.getMessage()}"
        )


_configured = False


def setup_logging(
    level: str = "INFO",
    log_format: str = "text",
    stream: object | None = None,
) -> None:
    """
    Configure the root glassbox_rag logger.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_format: 'json' or 'text'.
        stream: Output stream (defaults to sys.stderr).
    """
    global _configured
    if _configured:
        return

    root_logger = logging.getLogger("glassbox_rag")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(stream or sys.stderr)

    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root_logger.addHandler(handler)
    root_logger.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.

    Usage:
        logger = get_logger(__name__)
        logger.info("Something happened", extra={"trace_id": "abc"})
    """
    # Ensure logging is configured with defaults
    setup_logging()
    return logging.getLogger(name)
