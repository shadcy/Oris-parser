"""
exporters.json_exporter
=======================
Serialises DocumentResult objects to JSON files (or returns dict/str).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.schema import DocumentResult

logger = logging.getLogger(__name__)


def _default_serialiser(obj: Any) -> Any:
    """JSON fallback for Pydantic / enum objects."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "value"):          # Enum
        return obj.value
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def to_dict(result: DocumentResult) -> dict[str, Any]:
    """Convert DocumentResult to a plain nested dict."""
    return result.to_dict()


def to_json_string(result: DocumentResult, *, indent: int = 2) -> str:
    """Serialise DocumentResult to a JSON string."""
    data = to_dict(result)
    return json.dumps(data, indent=indent, ensure_ascii=False, default=_default_serialiser)


def export_json(
    result: DocumentResult,
    output_path: str,
    *,
    indent: int = 2,
) -> Path:
    """
    Write DocumentResult to a JSON file.

    Args:
        result:      The DocumentResult to export.
        output_path: Path to the output .json file.
        indent:      JSON indentation (default 2).

    Returns:
        Resolved Path of the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    json_str = to_json_string(result, indent=indent)
    out.write_text(json_str, encoding="utf-8")

    logger.info("JSON exported -> %s", out)
    return out
