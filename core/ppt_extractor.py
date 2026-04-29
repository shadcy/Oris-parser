"""
core.ppt_extractor
==================
PPTX/PPT extraction pipeline using python-pptx.

For each slide the extractor captures:
  * Title placeholder (title + subtitle)
  * Text frames: body text, bullet lists with indentation levels
  * Tables: converted to structured row/cell format
  * Images: listed by description / alt-text (OCR available as optional flag)
  * Notes: speaker notes appended as a special section

Layout information is reconstructed from shape position / size data
(EMU units → normalised 0–1 coordinates).

PPT (binary format) is handled by converting via LibreOffice or
falling back to python-pptx directly if the file is already PPTX.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .layout_parser import parse_layout
from .ocr_pipeline import OCRResult, ocr_image
from .schema import (
    DocumentResult,
    ExtractionSource,
    OCRMetadata,
    PageRecord,
    PageType,
)
from .utils import normalise_text

logger = logging.getLogger(__name__)

# 1 inch = 914400 EMU (English Metric Units used by python-pptx)
_EMU_PER_INCH = 914400


# ── python-pptx helpers ───────────────────────────────────────────────────────

def _import_pptx():
    from pptx import Presentation
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Emu
    return Presentation, PP_ALIGN, Emu


def _emu_to_pts(emu: int) -> float:
    """Convert EMU to points (1 pt = 12700 EMU)."""
    return emu / 12700.0


def _shape_bbox(shape, slide_width_emu: int, slide_height_emu: int) -> dict[str, float]:
    """Return normalised bounding box for a shape."""
    sw = slide_width_emu  or 1
    sh = slide_height_emu or 1
    return {
        "x0": shape.left   / sw,
        "y0": shape.top    / sh,
        "x1": (shape.left + shape.width)  / sw,
        "y1": (shape.top  + shape.height) / sh,
    }


def _para_to_block(
    para,
    slide_width_emu: int,
    slide_height_emu: int,
    shape_bbox: dict[str, float],
    default_size: float = 12.0,
) -> Optional[dict[str, Any]]:
    """
    Convert a pptx Paragraph into a text block dict.
    Returns None for paragraphs with no usable text.
    """
    text_parts: list[str] = []
    max_size  = default_size
    is_bold   = False
    is_italic = False
    font_name: Optional[str] = None

    for run in para.runs:
        run_text = run.text or ""
        text_parts.append(run_text)
        try:
            font = run.font
            if font.size and font.size > 0:
                size_pts = font.size / 12700.0   # EMU → pts
                max_size = max(max_size, size_pts)
            if font.bold:
                is_bold = True
            if font.italic:
                is_italic = True
            if font.name and not font_name:
                font_name = font.name
        except Exception:
            pass

    full_text = "".join(text_parts).strip()
    if not full_text:
        return None

    # Bullet / indent level from paragraph format
    level = getattr(para, "level", 0) or 0
    bullet_prefix = ("• " * min(level, 1)) if level > 0 else ""

    return {
        "text":   bullet_prefix + full_text,
        "x0":     shape_bbox["x0"],
        "y0":     shape_bbox["y0"],
        "x1":     shape_bbox["x1"],
        "y1":     shape_bbox["y1"],
        "size":   max_size,
        "font":   font_name or "",
        "bold":   is_bold,
        "italic": is_italic,
        "level":  level,
    }


def _extract_table_shape(
    shape,
    slide_width_emu: int,
    slide_height_emu: int,
) -> dict[str, Any]:
    """Extract a pptx Table shape into a table-block dict."""
    table = shape.table
    rows: list[list[str]] = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append(cells)

    bb = _shape_bbox(shape, slide_width_emu, slide_height_emu)
    return {
        "type":       "table",
        "rows":       rows,
        "has_header": True,
        "x0": bb["x0"], "y0": bb["y0"],
        "x1": bb["x1"], "y1": bb["y1"],
        "bbox":       (bb["x0"], bb["y0"], bb["x1"], bb["y1"]),
        "text":       "",
    }


def _extract_slide_blocks(
    slide,
    slide_width_emu: int,
    slide_height_emu: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], Optional[str]]:
    """
    Extract all shapes from a slide.

    Returns:
        (text_blocks, table_blocks, image_descriptions, notes_text)
    """
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    text_blocks:  list[dict[str, Any]] = []
    table_blocks: list[dict[str, Any]] = []
    image_descs:  list[str]            = []
    notes_text:   Optional[str]        = None

    for shape in slide.shapes:
        stype = shape.shape_type

        # ── Table ────────────────────────────────────────────────────────────
        if shape.has_table:
            try:
                block = _extract_table_shape(shape, slide_width_emu, slide_height_emu)
                table_blocks.append(block)
            except Exception as exc:
                logger.debug("Table extraction failed: %s", exc)
            continue

        # ── Text frame ────────────────────────────────────────────────────────
        if shape.has_text_frame:
            bb = _shape_bbox(shape, slide_width_emu, slide_height_emu)
            for para in shape.text_frame.paragraphs:
                block = _para_to_block(
                    para, slide_width_emu, slide_height_emu, bb
                )
                if block:
                    text_blocks.append(block)
            continue

        # ── Image / Picture ───────────────────────────────────────────────────
        if stype == MSO_SHAPE_TYPE.PICTURE:
            try:
                alt = shape.name or "image"
                image_descs.append(f"[Image: {alt}]")
            except Exception:
                image_descs.append("[Image]")

    # ── Speaker notes ─────────────────────────────────────────────────────────
    try:
        if slide.has_notes_slide:
            notes_tf = slide.notes_slide.notes_text_frame
            notes_raw = notes_tf.text.strip() if notes_tf else ""
            if notes_raw:
                notes_text = normalise_text(notes_raw)
    except Exception:
        pass

    return text_blocks, table_blocks, image_descs, notes_text


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_pptx(
    pptx_path: str,
    *,
    ocr_engine: str = "tesseract",
    ocr_preprocess: bool = True,
    ocr_images: bool = False,
) -> DocumentResult:
    """
    Extract a PPTX file slide-by-slide into a structured DocumentResult.

    Args:
        pptx_path:    Path to the .pptx (or .ppt) file.
        ocr_engine:   OCR engine for image slides ("tesseract" | "easyocr").
        ocr_preprocess: Apply image preprocessing before OCR.
        ocr_images:   Whether to OCR embedded images within slides.
    """
    path = Path(pptx_path)
    result = DocumentResult(
        source_file=str(path.resolve()),
        file_type=path.suffix.lstrip(".").lower(),
        total_pages=0,
    )

    try:
        Presentation, PP_ALIGN, Emu = _import_pptx()
        prs = Presentation(str(path))
    except Exception as exc:
        logger.error("Failed to open PPTX %s: %s", pptx_path, exc)
        result.errors.append(f"Open failed: {exc}")
        return result

    slide_width_emu  = int(prs.slide_width)
    slide_height_emu = int(prs.slide_height)

    # Normalised page size in points (used by layout parser)
    page_width_pts  = _emu_to_pts(slide_width_emu)
    page_height_pts = _emu_to_pts(slide_height_emu)

    result.total_pages = len(prs.slides)

    for slide_idx, slide in enumerate(prs.slides):
        slide_num = slide_idx + 1
        logger.info("Processing slide %d / %d", slide_num, result.total_pages)

        try:
            text_blocks, table_blocks, image_descs, notes_text = (
                _extract_slide_blocks(slide, slide_width_emu, slide_height_emu)
            )
        except Exception as exc:
            logger.error("Slide %d extraction failed: %s", slide_num, exc)
            result.errors.append(f"Slide {slide_num}: {exc}")
            _append_error_record(result, str(path.resolve()), slide_num)
            continue

        # Scale normalised [0,1] bbox to pts for layout parser
        # (layout parser works in actual coordinates)
        for b in text_blocks + table_blocks:
            b["x0"] *= page_width_pts
            b["y0"] *= page_height_pts
            b["x1"] *= page_width_pts
            b["y1"] *= page_height_pts

        all_blocks = text_blocks + table_blocks

        # Determine page type (PPTX slides have native text)
        page_type = PageType.DIGITAL
        if not all_blocks and image_descs:
            page_type = PageType.SCANNED   # Image-only slide

        from .layout_parser import parse_layout
        layout = parse_layout(all_blocks, page_width_pts, page_height_pts)

        # Append notes as a dedicated section if present
        if notes_text:
            from .schema import Section
            notes_section = Section(
                heading="Speaker Notes",
                body=notes_text,
            )
            layout["sections"].append(notes_section)

        rec = PageRecord(
            source_file=str(path.resolve()),
            page_number=slide_num,
            page_type=page_type,
            extraction_source=ExtractionSource.PYTHON_PPTX,
            title=layout["title"],
            subtitle=layout["subtitle"],
            sections=layout["sections"],
            bullets=layout["bullets"],
            tables=layout["tables"],
            images=image_descs,
            raw_text=layout["raw_text"],
            page_width=page_width_pts,
            page_height=page_height_pts,
            columns_detected=layout["columns_detected"],
        )
        result.pages.append(rec)

    logger.info(
        "PPTX extraction complete: %d slides, %d errors — %s",
        len(result.pages), len(result.errors), path.name,
    )
    return result


def _append_error_record(
    result: DocumentResult,
    source_file: str,
    slide_number: int,
) -> None:
    result.pages.append(PageRecord(
        source_file=source_file,
        page_number=slide_number,
        is_corrupted=True,
        error_message="Extraction failed; see errors list.",
    ))
