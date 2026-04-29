"""
core.utils
==========
Shared helpers used across all extraction pipelines:
  - Text normalisation & whitespace cleanup
  - Bounding-box helpers & coordinate normalisation
  - Reading-order sorting (top-to-bottom, left-to-right)
  - Multi-column detection
  - Logging configuration
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Sequence

import numpy as np

from .schema import BoundingBox

# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


# ── Text normalisation ────────────────────────────────────────────────────────

_MULTI_SPACE_RE  = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_CONTROL_CHAR_RE  = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalise_text(text: str, *, keep_newlines: bool = True) -> str:
    """
    Clean raw extracted text:
      1. Unicode NFC normalisation.
      2. Strip invisible / control characters.
      3. Collapse horizontal whitespace (spaces / tabs) to a single space.
      4. Collapse excessive blank lines to at most two consecutive newlines.
      5. Strip leading / trailing whitespace.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_CHAR_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    if keep_newlines:
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    else:
        text = text.replace("\n", " ")
    return text.strip()


def is_uppercase_title(text: str) -> bool:
    """Heuristic: a string is likely a title/heading if it is ALL CAPS."""
    stripped = text.strip()
    if not stripped:
        return False
    letters = [c for c in stripped if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def looks_like_bullet(text: str) -> tuple[bool, int, str]:
    """
    Detect common bullet patterns.

    Returns:
        (is_bullet, indent_level, cleaned_text)

    Patterns recognised:
      •  ·  -  *  ►  ▪  ▸  (unicode bullets)
      1. / a) / i. / (1) / [1] (ordered lists)
    """
    text = text.lstrip()
    # Ordered list: leading digits/letters followed by . or )
    m = re.match(r"^(\s*)(?:\(?\d+[\.\)]\s*|\(?[a-zA-Z][\.\)]\s*)", text)
    if m:
        indent = len(m.group(1)) // 2
        cleaned = text[m.end():].strip()
        return True, indent, cleaned

    # Unordered / unicode bullets
    m = re.match(r"^(\s*)[•·\-\*►▪▸‣⁃◦]\s+", text)
    if m:
        indent = len(m.group(1)) // 2
        cleaned = text[m.end():].strip()
        return True, indent, cleaned

    return False, 0, text


# ── Bounding-box helpers ──────────────────────────────────────────────────────

def normalise_bbox(
    x0: float, y0: float, x1: float, y1: float,
    page_width: float, page_height: float,
) -> BoundingBox:
    """Convert absolute pixel / point coords to normalised [0, 1] space."""
    pw = page_width  or 1.0
    ph = page_height or 1.0
    return BoundingBox(
        x0=max(0.0, x0 / pw),
        y0=max(0.0, y0 / ph),
        x1=min(1.0, x1 / pw),
        y1=min(1.0, y1 / ph),
    )


def bbox_area(bbox: BoundingBox) -> float:
    return max(0.0, bbox.x1 - bbox.x0) * max(0.0, bbox.y1 - bbox.y0)


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-Union for two bounding boxes."""
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


# ── Reading-order sorting ─────────────────────────────────────────────────────

def reading_order_key(block: dict[str, Any]) -> tuple[float, float]:
    """
    Produce a (row_band, x0) sort key so that text blocks are ordered
    top-to-bottom then left-to-right.

    ``block`` must contain keys ``x0``, ``y0``, ``x1``, ``y1``
    (absolute coordinates).

    A 'row band' groups elements whose vertical centres differ by less than
    ``band_height`` so that columns on the same visual row sort together.
    """
    cy = (block["y0"] + block["y1"]) / 2.0
    return (round(cy, -1), block["x0"])   # round to nearest 10 pts


def sort_blocks_reading_order(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort a list of bounding-box dicts in reading order."""
    return sorted(blocks, key=reading_order_key)


# ── Multi-column detection ────────────────────────────────────────────────────

def detect_columns(
    blocks: list[dict[str, Any]],
    page_width: float,
    gap_threshold: float = 0.05,   # fraction of page width
) -> int:
    """
    Estimate the number of text columns on a page using x-coordinate
    projection analysis.

    Args:
        blocks: List of dicts with at least ``x0`` and ``x1`` keys.
        page_width: Page width in the same unit as the block coords.
        gap_threshold: Minimum relative gap size to count as a column break.

    Returns:
        Estimated number of columns (1, 2, or 3).
    """
    if not blocks:
        return 1

    # Build an x-coverage array at 1-point resolution
    resolution = int(page_width) or 800
    coverage = np.zeros(resolution, dtype=np.float32)
    for b in blocks:
        l = max(0, int(b.get("x0", 0)))
        r = min(resolution - 1, int(b.get("x1", resolution)))
        if l < r:
            coverage[l:r] += 1.0

    # Smooth with a small kernel to remove tiny gaps
    kernel_size = max(1, int(page_width * 0.02))
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(coverage, kernel, mode="same")

    # Find significant gaps (coverage ≈ 0 in middle 80% of page)
    margin = int(page_width * 0.05)
    mid = smoothed[margin: resolution - margin]
    gap_val = smoothed.max() * 0.1
    gaps = (mid < gap_val).astype(int)

    # Count transitions from non-gap to gap
    transitions = int(np.sum(np.diff(gaps) == 1))
    columns = min(3, transitions + 1)
    return max(1, columns)


def assign_columns(
    blocks: list[dict[str, Any]],
    num_columns: int,
    page_width: float,
) -> list[dict[str, Any]]:
    """
    Tag each block with a ``column`` index (0-based) based on its x-centre.
    """
    if num_columns <= 1:
        for b in blocks:
            b["column"] = 0
        return blocks

    col_width = page_width / num_columns
    for b in blocks:
        cx = (b.get("x0", 0) + b.get("x1", 0)) / 2.0
        b["column"] = min(num_columns - 1, int(cx / col_width))
    return blocks


# ── Misc helpers ──────────────────────────────────────────────────────────────

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def file_extension(path: str) -> str:
    """Return lower-case extension without the leading dot."""
    import os
    return os.path.splitext(path)[1].lstrip(".").lower()
