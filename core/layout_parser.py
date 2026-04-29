"""
core.layout_parser
==================
Layout-aware grouping of raw text/OCR blocks into structured sections,
supporting:
  • Multi-column layout detection and per-column ordering
  • Heading / subtitle identification via font-size and UPPERCASE heuristics
  • Bullet detection and nested list reconstruction
  • Table boundary detection (basic grid approach for native PDF tables)
  • Whitespace normalisation with semantic-break preservation

Input:  A list of raw "block dicts" produced by the PDF/OCR extractors.
Output: A populated list of ``Section`` objects, plus lists of
        ``BulletItem``s and ``Table``s for the top-level page record.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from .schema import (
    BoundingBox,
    BulletItem,
    Section,
    Table,
    TableCell,
    TextSpan,
)
from .utils import (
    assign_columns,
    detect_columns,
    is_uppercase_title,
    looks_like_bullet,
    normalise_text,
    sort_blocks_reading_order,
)

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

TITLE_FONT_SIZE_FACTOR   = 1.4   # block font_size > median * factor → title
SUBTITLE_FONT_SIZE_FACTOR = 1.15
HEADING_FONT_SIZE_FACTOR  = 1.1
MIN_TABLE_COLS = 2
MIN_TABLE_ROWS = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _median_font_size(blocks: list[dict[str, Any]]) -> float:
    sizes = [b.get("size", 0) for b in blocks if b.get("size", 0) > 0]
    if not sizes:
        return 12.0
    sizes.sort()
    mid = len(sizes) // 2
    return float(sizes[mid])


def _block_to_span(block: dict[str, Any]) -> TextSpan:
    text = normalise_text(block.get("text", ""), keep_newlines=False)
    flags = block.get("flags", 0)           # PyMuPDF font flags
    bold   = bool(flags & 16) or block.get("bold", False)
    italic = bool(flags & 2)  or block.get("italic", False)
    return TextSpan(
        text=text,
        bold=bold or is_uppercase_title(text),
        italic=italic,
        font_size=block.get("size"),
        font_name=block.get("font"),
    )


def _classify_block(
    block: dict[str, Any],
    median_size: float,
) -> str:
    """Return 'title' | 'subtitle' | 'heading' | 'bullet' | 'table' | 'body'."""
    text = block.get("text", "").strip()
    size = block.get("size", 0)

    # Table blocks are pre-tagged by the extractor
    if block.get("type") == "table":
        return "table"

    # Font-size based heading detection
    if size > 0:
        if size >= median_size * TITLE_FONT_SIZE_FACTOR:
            return "title"
        if size >= median_size * SUBTITLE_FONT_SIZE_FACTOR:
            return "subtitle"
        if size >= median_size * HEADING_FONT_SIZE_FACTOR:
            return "heading"

    # Uppercase heuristic
    if is_uppercase_title(text) and len(text.split()) <= 12:
        return "heading"

    # Bullet heuristic
    is_b, _, _ = looks_like_bullet(text)
    if is_b:
        return "bullet"

    return "body"


def _build_bullet_item(block: dict[str, Any]) -> BulletItem:
    is_b, level, cleaned = looks_like_bullet(block.get("text", ""))
    span = _block_to_span({**block, "text": cleaned})
    return BulletItem(level=level, text=cleaned, spans=[span])


def _nest_bullets(flat: list[BulletItem]) -> list[BulletItem]:
    """
    Convert a flat list of BulletItems (with level attributes) into a tree
    where children are nested under their parents.
    """
    root: list[BulletItem] = []
    stack: list[BulletItem] = []

    for item in flat:
        # Pop stack until the parent level is < item level
        while stack and stack[-1].level >= item.level:
            stack.pop()
        if stack:
            stack[-1].children.append(item)
        else:
            root.append(item)
        stack.append(item)
    return root


def _extract_table_from_block(block: dict[str, Any]) -> Optional[Table]:
    """
    Convert a pre-extracted table block (dict with 'rows' key) into a
    Table schema object.  ``block["rows"]`` should be a list[list[str]].
    """
    raw_rows: list[list[str]] = block.get("rows", [])
    if not raw_rows:
        return None

    has_header = block.get("has_header", True)
    headers: list[str] = []
    data_rows: list[list[TableCell]] = []

    if has_header and raw_rows:
        headers = [str(c).strip() for c in raw_rows[0]]
        raw_rows = raw_rows[1:]

    for raw_row in raw_rows:
        row = [TableCell(text=str(c).strip()) for c in raw_row]
        data_rows.append(row)

    caption = block.get("caption")
    bbox_raw = block.get("bbox")
    bbox = None
    if bbox_raw:
        x0, y0, x1, y1 = bbox_raw
        bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

    return Table(caption=caption, headers=headers, rows=data_rows, bbox=bbox)


# ── Section builder ───────────────────────────────────────────────────────────

def _build_sections(
    column_blocks: list[dict[str, Any]],
    median_size: float,
    column_idx: int,
) -> tuple[list[Section], list[BulletItem], list[Table]]:
    """
    Process all blocks for a single column and assemble Section objects.

    Returns: (sections, top_level_bullets, top_level_tables)
    """
    sections: list[Section]     = []
    top_bullets: list[BulletItem] = []
    top_tables:  list[Table]      = []

    current_section: Optional[Section] = None
    pending_bullets: list[BulletItem]  = []

    def _flush_section():
        nonlocal current_section, pending_bullets
        if current_section is not None:
            current_section.bullets = _nest_bullets(pending_bullets)
            sections.append(current_section)
        pending_bullets = []

    def _new_section(heading: Optional[str], bbox_raw=None) -> Section:
        bbox = None
        if bbox_raw:
            x0, y0, x1, y1 = bbox_raw
            bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
        return Section(heading=heading, column=column_idx, bbox=bbox)

    for block in column_blocks:
        text  = normalise_text(block.get("text", ""))
        btype = _classify_block(block, median_size)

        if not text and btype != "table":
            continue

        if btype == "table":
            table = _extract_table_from_block(block)
            if table:
                if current_section is not None:
                    current_section.tables.append(table)
                else:
                    top_tables.append(table)
            continue

        if btype in ("title", "subtitle", "heading"):
            _flush_section()
            current_section = _new_section(
                heading=text,
                bbox_raw=block.get("bbox"),
            )
            span = _block_to_span(block)
            current_section.spans.append(span)

        elif btype == "bullet":
            item = _build_bullet_item(block)
            if current_section is not None:
                pending_bullets.append(item)
            else:
                top_bullets.append(item)

        else:  # body
            if current_section is None:
                current_section = _new_section(None)
            span = _block_to_span(block)
            current_section.spans.append(span)
            body_addition = text
            if current_section.body:
                current_section.body += "\n" + body_addition
            else:
                current_section.body = body_addition

    _flush_section()

    # Nest top-level bullets
    top_bullets = _nest_bullets(top_bullets)

    return sections, top_bullets, top_tables


# ── Public API ────────────────────────────────────────────────────────────────

def parse_layout(
    blocks: list[dict[str, Any]],
    page_width: float,
    page_height: float,
) -> dict[str, Any]:
    """
    Full layout-parsing pass for one page.

    Args:
        blocks:      Raw text blocks, each a dict with keys:
                       text, x0, y0, x1, y1,
                       optionally: size, font, flags, bold, italic,
                                   type (='table'), rows, has_header, caption
        page_width:  Page width in points/pixels.
        page_height: Page height in points/pixels.

    Returns:
        dict with keys:
          title, subtitle, sections, bullets, tables,
          columns_detected, raw_text
    """
    if not blocks:
        return {
            "title": None, "subtitle": None,
            "sections": [], "bullets": [], "tables": [],
            "columns_detected": 1, "raw_text": "",
        }

    # 1. Sort in reading order
    blocks = sort_blocks_reading_order(blocks)

    # 2. Detect columns
    num_cols = detect_columns(blocks, page_width)

    # 3. Assign column tags
    blocks = assign_columns(blocks, num_cols, page_width)

    # 4. Compute median font size for heading classification
    median_size = _median_font_size(blocks)

    # 5. Group blocks by column and sort within each column
    col_groups: dict[int, list[dict[str, Any]]] = {}
    for b in blocks:
        col = b.get("column", 0)
        col_groups.setdefault(col, []).append(b)

    # Within each column, re-sort by reading order (y then x)
    for col in col_groups:
        col_groups[col] = sort_blocks_reading_order(col_groups[col])

    # 6. Build sections per column
    all_sections:  list[Section]     = []
    all_bullets:   list[BulletItem]  = []
    all_tables:    list[Table]        = []

    for col_idx in sorted(col_groups.keys()):
        secs, buls, tabs = _build_sections(
            col_groups[col_idx], median_size, col_idx
        )
        all_sections.extend(secs)
        all_bullets.extend(buls)
        all_tables.extend(tabs)

    # 7. Identify page title and subtitle from first two heading sections
    page_title:    Optional[str] = None
    page_subtitle: Optional[str] = None
    text_only_sections: list[Section] = []

    for sec in all_sections:
        if sec.heading:
            if page_title is None:
                page_title = sec.heading
            elif page_subtitle is None:
                page_subtitle = sec.heading
                text_only_sections.append(sec)
            else:
                text_only_sections.append(sec)
        else:
            text_only_sections.append(sec)

    # 8. Build raw_text from all blocks in reading order
    raw_text = "\n".join(
        normalise_text(b.get("text", ""))
        for b in blocks
        if b.get("text", "").strip()
    )

    return {
        "title":             page_title,
        "subtitle":          page_subtitle,
        "sections":          all_sections,
        "bullets":           all_bullets,
        "tables":            all_tables,
        "columns_detected":  num_cols,
        "raw_text":          normalise_text(raw_text),
    }
