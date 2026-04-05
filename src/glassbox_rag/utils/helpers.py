"""General helper utilities."""

import json
from typing import Any, Dict


def safe_json_encode(obj: Any) -> str:
    """
    Safely encode object to JSON string.

    Handles non-serializable objects by converting to string.
    Handles numpy arrays, datetime, bytes, etc.
    """
    try:
        return json.dumps(obj, default=_json_default)
    except (TypeError, ValueError):
        return json.dumps(str(obj))


def _json_default(obj: Any) -> Any:
    """Default handler for JSON serialization of non-standard types."""
    import datetime

    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge two dictionaries. Values from override take precedence.

    Nested dicts are merged recursively; lists are replaced (not merged).
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def truncate(text: str, max_length: int = 200, suffix: str = "…") -> str:
    """Truncate text to max_length, appending suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def estimate_tokens(text: str) -> int:
    """
    Rough token count estimate (for cost tracking before actual encoding).

    Uses the ~4 chars per token heuristic.
    """
    return max(1, len(text) // 4)
