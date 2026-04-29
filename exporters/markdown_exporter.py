"""
exporters.markdown_exporter
============================
Renders DocumentResult objects to a human-readable Markdown file,
preserving the logical hierarchy of the document:

  # Document: <filename>
  ---
  ## Page N  [type | extraction_source]
  ### Title
  #### Subtitle
  ##### Section heading
  body text …
  - bullet
    - nested bullet
  | table | headers |
  | ----- | ------- |
  | cell  | cell    |
  > 💬 Speaker notes …

  ---
  ## Appendix: Errors
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.schema import BulletItem, DocumentResult, PageRecord, Section, Table

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _escape_md(text: str) -> str:
    """Minimal markdown escaping for pipe characters inside table cells."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _render_bullet_item(item: BulletItem, indent: int = 0) -> list[str]:
    prefix = "  " * indent + "- "
    lines = [prefix + item.text]
    for child in item.children:
        lines.extend(_render_bullet_item(child, indent + 1))
    return lines


def _render_table(table: Table) -> list[str]:
    lines: list[str] = []

    if table.caption:
        lines.append(f"*Table: {table.caption}*")
        lines.append("")

    if table.headers:
        header_row = "| " + " | ".join(_escape_md(h) for h in table.headers) + " |"
        sep_row    = "| " + " | ".join("---" for _ in table.headers) + " |"
        lines.append(header_row)
        lines.append(sep_row)

    for row in table.rows:
        # rows may be list[TableCell] or list[dict]
        cells: list[str] = []
        for cell in row:
            if hasattr(cell, "text"):
                cells.append(_escape_md(cell.text))
            elif isinstance(cell, dict):
                cells.append(_escape_md(cell.get("text", "")))
            else:
                cells.append(_escape_md(str(cell)))
        lines.append("| " + " | ".join(cells) + " |")

    return lines


def _render_section(section: Section, depth: int = 4) -> list[str]:
    """Recursively render a Section to Markdown lines."""
    lines: list[str] = []
    hashes = "#" * min(depth, 6)

    if section.heading:
        lines.append(f"{hashes} {section.heading}")
        lines.append("")

    if section.body.strip():
        lines.append(section.body.strip())
        lines.append("")

    # Bullets
    for item in section.bullets:
        lines.extend(_render_bullet_item(item))
    if section.bullets:
        lines.append("")

    # Tables
    for table in section.tables:
        lines.extend(_render_table(table))
        lines.append("")

    return lines


def _render_page(page: PageRecord, page_label: str = "Page") -> list[str]:
    lines: list[str] = []

    # ── Page header ───────────────────────────────────────────────────────────
    type_badge = f"`{page.page_type.value}`" if page.page_type else ""
    src_badge  = f"`{page.extraction_source.value}`" if page.extraction_source else ""
    lines.append(
        f"## {page_label} {page.page_number}  "
        f"{type_badge} · {src_badge}"
    )
    lines.append("")

    if page.is_corrupted:
        lines.append(f"> ⚠️ **Corrupted / extraction failed**")
        if page.error_message:
            lines.append(f"> {page.error_message}")
        lines.append("")
        return lines

    # ── Title / subtitle ──────────────────────────────────────────────────────
    if page.title:
        lines.append(f"### {page.title}")
        lines.append("")
    if page.subtitle:
        lines.append(f"#### {page.subtitle}")
        lines.append("")

    # ── Top-level bullets ─────────────────────────────────────────────────────
    for item in page.bullets:
        lines.extend(_render_bullet_item(item))
    if page.bullets:
        lines.append("")

    # ── Sections ──────────────────────────────────────────────────────────────
    for section in page.sections:
        lines.extend(_render_section(section, depth=4))

    # ── Top-level tables ──────────────────────────────────────────────────────
    for table in page.tables:
        lines.extend(_render_table(table))
        lines.append("")

    # ── Images ────────────────────────────────────────────────────────────────
    for img_desc in page.images:
        lines.append(f"*{img_desc}*")
    if page.images:
        lines.append("")

    # ── OCR metadata ──────────────────────────────────────────────────────────
    if page.ocr_metadata:
        om = page.ocr_metadata
        lines.append(
            f"> 🔍 **OCR** [{om.engine.value}]  "
            f"confidence: {om.mean_confidence:.1f}%  "
            f"skew: {om.skew_angle_deg:.1f}°"
        )
        if om.low_conf_blocks:
            lines.append(f"> ⚠️ {om.low_conf_blocks} low-confidence block(s)")
        lines.append("")

    # ── Columns hint ─────────────────────────────────────────────────────────
    if page.columns_detected and page.columns_detected > 1:
        lines.append(f"*Layout: {page.columns_detected}-column*")
        lines.append("")

    return lines


# ── Public API ────────────────────────────────────────────────────────────────

def to_markdown_string(result: DocumentResult) -> str:
    """Render DocumentResult to a Markdown string."""
    lines: list[str] = []
    file_name = Path(result.source_file).name

    # Document header
    lines.append(f"# Document: {file_name}")
    lines.append("")
    lines.append(
        f"**Type:** `{result.file_type.upper()}` · "
        f"**Pages/Slides:** {result.total_pages}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Page label
    page_label = "Slide" if result.file_type in ("pptx", "ppt") else "Page"

    # Pages / slides
    for page in result.pages:
        lines.extend(_render_page(page, page_label=page_label))
        lines.append("---")
        lines.append("")

    # Errors appendix
    if result.errors:
        lines.append("## ⚠️ Processing Errors")
        lines.append("")
        for err in result.errors:
            lines.append(f"- {err}")
        lines.append("")

    return "\n".join(lines)


def export_markdown(
    result: DocumentResult,
    output_path: str,
) -> Path:
    """
    Write DocumentResult to a Markdown file.

    Args:
        result:      The DocumentResult to export.
        output_path: Path to the output .md file.

    Returns:
        Resolved Path of the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    md_str = to_markdown_string(result)
    out.write_text(md_str, encoding="utf-8")
    logger.info("Markdown exported -> %s", out)
    return out
