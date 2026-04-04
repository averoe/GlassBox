"""General helper utilities."""

import json
from typing import Any, Dict


def safe_json_encode(obj: Any) -> str:
    """
    Safely encode object to JSON string.

    Handles non-serializable objects by converting to string.
    """
    try:
        return json.dumps(obj)
    except TypeError:
        return json.dumps(str(obj))


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries (recursive).

    Values from override dict take precedence.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
