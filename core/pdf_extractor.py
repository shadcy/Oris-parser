"""
core.pdf_extractor
==================
PDF extraction pipeline:
  1. Native extraction (pdfplumber → PyMuPDF fallback)
  2. Automatic scanned-page detection
  3. OCR fallback (Tesseract / EasyOCR) for scanned pages
  4. Layout parsing via core.layout_parser
  5. Produces one PageRecord per page

Design philosophy
-----------------
* pdfplumber   – primary engine; exposes character-level bbox data and
                 native table extraction.
* PyMuPDF (fitz) – secondary engine; richer font flags, better image
                   rendering for OCR, and faster on large files.
* Both engines are tried in order; the one that yields more text wins.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .layout_parser import parse_layout
from .ocr_pipeline import OCRResult, is_scanned_page, ocr_image
from .schema import (
    DocumentResult,
    ExtractionSource,
    OCRMetadata,
    PageRecord,
    PageType,
)
from .utils import normalise_text

logger = logging.getLogger(__name__)

# ── Lazy imports (avoid paying import cost unless the engine is used) ──────────

def _import_pdfplumber():
    import pdfplumber
    return pdfplumber


def _import_fitz():
    import fitz   # PyMuPDF
    return fitz


# ── pdfplumber extraction ─────────────────────────────────────────────────────

def _extract_page_pdfplumber(page) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Extract text blocks and tables from a single pdfplumber page.

    Returns:
        (text_blocks, table_blocks)
        Each text block: {text, x0, y0, x1, y1, size, font, bold}
        Each table block: {type='table', rows, bbox, has_header, x0,y0,x1,y1}
    """
    text_blocks: list[dict[str, Any]] = []
    table_blocks: list[dict[str, Any]] = []

    # ── Tables first (remove their area from text extraction) ─────────────────
    try:
        tables = page.extract_tables(
            table_settings={
                "vertical_strategy":   "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance":      3,
                "join_tolerance":      3,
                "edge_min_length":     3,
            }
        )
        for tbl_obj in (page.find_tables() or []):
            rows = tbl_obj.extract()
            if not rows:
                continue
            bb = tbl_obj.bbox   # (x0, y0, x1, y1) in points
            table_blocks.append({
                "type":       "table",
                "rows":       rows,
                "has_header": True,
                "x0": bb[0], "y0": bb[1], "x1": bb[2], "y1": bb[3],
                "bbox":       (bb[0], bb[1], bb[2], bb[3]),
                "text":       "",
            })
    except Exception as exc:
        logger.debug("pdfplumber table extraction failed: %s", exc)

    # ── Text characters grouped into word-level blocks ─────────────────────────
    try:
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
            extra_attrs=["fontname", "size"],
        )
        for w in words:
            text_blocks.append({
                "text": w.get("text", ""),
                "x0":   w.get("x0", 0),
                "y0":   w.get("top", 0),
                "x1":   w.get("x1", 0),
                "y1":   w.get("bottom", 0),
                "size": w.get("size", 0),
                "font": w.get("fontname", ""),
            })
    except Exception as exc:
        logger.debug("pdfplumber word extraction failed: %s", exc)

    return text_blocks, table_blocks


def _extract_with_pdfplumber(
    pdf_path: str,
) -> Optional[list[tuple[list[dict], list[dict], float, float]]]:
    """
    Extract all pages using pdfplumber.
    Returns list of (text_blocks, table_blocks, width, height) per page,
    or None if the library is unavailable or the file fails to open.
    """
    try:
        pdfplumber = _import_pdfplumber()
    except ImportError:
        logger.warning("pdfplumber not installed.")
        return None

    try:
        results = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tb, tbl = _extract_page_pdfplumber(page)
                results.append((tb, tbl, float(page.width), float(page.height)))
        return results
    except Exception as exc:
        logger.warning("pdfplumber failed on %s: %s", pdf_path, exc)
        return None


# ── PyMuPDF extraction ────────────────────────────────────────────────────────

def _extract_page_pymupdf(page) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Extract text spans and tables from a single PyMuPDF page.

    Uses page.get_text("dict") for rich per-span font metadata.
    """
    text_blocks: list[dict[str, Any]] = []
    table_blocks: list[dict[str, Any]] = []

    try:
        data = page.get_text("dict", flags=7)   # flags: preserve whitespace
        for block in data.get("blocks", []):
            btype = block.get("type", -1)
            if btype == 1:
                # Image block — skip (will be handled by OCR if needed)
                continue
            if btype != 0:
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if not span_text:
                        continue
                    bbox = span.get("bbox", (0, 0, 0, 0))
                    text_blocks.append({
                        "text":  span_text,
                        "x0":    bbox[0], "y0": bbox[1],
                        "x1":    bbox[2], "y1": bbox[3],
                        "size":  span.get("size", 0),
                        "font":  span.get("font", ""),
                        "flags": span.get("flags", 0),
                    })
    except Exception as exc:
        logger.debug("PyMuPDF span extraction failed: %s", exc)

    # Table extraction via PyMuPDF's built-in finder (v1.23+)
    try:
        tabs = page.find_tables()
        for tbl in tabs:
            rows = tbl.extract()
            if not rows:
                continue
            bb = tbl.bbox
            table_blocks.append({
                "type":       "table",
                "rows":       rows,
                "has_header": True,
                "x0": bb[0], "y0": bb[1], "x1": bb[2], "y1": bb[3],
                "bbox":       (bb[0], bb[1], bb[2], bb[3]),
                "text":       "",
            })
    except Exception:
        pass  # find_tables not available in older PyMuPDF

    return text_blocks, table_blocks


def _render_page_to_pil(page, dpi: int = 200):
    """Render a PyMuPDF page to a PIL Image for OCR."""
    fitz = _import_fitz()
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    from PIL import Image
    import io
    return Image.open(io.BytesIO(pix.tobytes("png")))


def _extract_with_pymupdf(
    pdf_path: str,
) -> Optional[list[tuple[list[dict], list[dict], float, float, object]]]:
    """
    Extract all pages using PyMuPDF.
    Returns list of (text_blocks, table_blocks, width, height, fitz_page) per page.
    """
    try:
        fitz = _import_fitz()
    except ImportError:
        logger.warning("PyMuPDF (fitz) not installed.")
        return None

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.warning("PyMuPDF failed to open %s: %s", pdf_path, exc)
        return None

    results = []
    for page in doc:
        tb, tbl = _extract_page_pymupdf(page)
        rect = page.rect
        results.append((tb, tbl, rect.width, rect.height, page))
    return results


# ── OCR helpers ───────────────────────────────────────────────────────────────

def _ocr_blocks_from_result(result: OCRResult) -> list[dict[str, Any]]:
    """Convert OCRResult blocks into the standard block-dict format."""
    return [
        {
            "text": b.text,
            "x0":   b.x0, "y0": b.y0,
            "x1":   b.x1, "y1": b.y1,
            "size": 0, "font": "",
        }
        for b in result.blocks
        if b.text.strip()
    ]


# ── Page record builder ───────────────────────────────────────────────────────

def _build_page_record(
    source_file: str,
    page_number: int,
    text_blocks: list[dict[str, Any]],
    table_blocks: list[dict[str, Any]],
    page_width: float,
    page_height: float,
    extraction_source: ExtractionSource,
    ocr_meta: Optional[OCRMetadata] = None,
    page_type: PageType = PageType.DIGITAL,
) -> PageRecord:
    """Assemble a PageRecord from raw blocks using the layout parser."""
    all_blocks = text_blocks + table_blocks
    layout = parse_layout(all_blocks, page_width, page_height)

    return PageRecord(
        source_file=source_file,
        page_number=page_number,
        page_type=page_type,
        extraction_source=extraction_source,
        title=layout["title"],
        subtitle=layout["subtitle"],
        sections=layout["sections"],
        bullets=layout["bullets"],
        tables=layout["tables"],
        raw_text=layout["raw_text"],
        page_width=page_width,
        page_height=page_height,
        columns_detected=layout["columns_detected"],
        ocr_metadata=ocr_meta,
    )


# ── Scanned-page detection ────────────────────────────────────────────────────

_NATIVE_TEXT_MIN_CHARS = 30   # Fewer chars than this → treat as scanned


def _is_low_text(blocks: list[dict[str, Any]]) -> bool:
    total_chars = sum(len(b.get("text", "")) for b in blocks)
    return total_chars < _NATIVE_TEXT_MIN_CHARS


# ── Public API ────────────────────────────────────────────────────────────────

def extract_pdf(
    pdf_path: str,
    *,
    ocr_engine: str = "tesseract",
    ocr_preprocess: bool = True,
    ocr_dpi: int = 200,
    force_ocr: bool = False,
) -> DocumentResult:
    """
    Extract a PDF file page-by-page into a structured DocumentResult.

    Strategy
    --------
    1. Try pdfplumber (primary native engine).
    2. If pdfplumber fails or yields very little text, try PyMuPDF.
    3. For each page individually:
       a. If native text is sufficient → DIGITAL page.
       b. Otherwise render the page to an image and run OCR → SCANNED / MIXED.

    Args:
        pdf_path:       Absolute or relative path to the PDF file.
        ocr_engine:     "tesseract" | "easyocr"
        ocr_preprocess: Apply image preprocessing before OCR.
        ocr_dpi:        DPI for page rendering (higher = more accurate OCR).
        force_ocr:      Force OCR on every page, even if native text exists.
    """
    path = Path(pdf_path)
    result = DocumentResult(
        source_file=str(path.resolve()),
        file_type="pdf",
        total_pages=0,
    )

    # ── Try pdfplumber ────────────────────────────────────────────────────────
    plumber_pages = _extract_with_pdfplumber(pdf_path)

    # ── Always load PyMuPDF for page rendering and fallback ───────────────────
    fitz_pages = _extract_with_pymupdf(pdf_path)

    if plumber_pages is None and fitz_pages is None:
        result.errors.append(f"All native engines failed on {pdf_path}")
        return result

    # Determine page count from whichever engine worked
    num_pages = len(plumber_pages) if plumber_pages else len(fitz_pages)
    result.total_pages = num_pages

    for page_idx in range(num_pages):
        page_num = page_idx + 1
        logger.info("Processing PDF page %d / %d", page_num, num_pages)

        # ── Gather native blocks ──────────────────────────────────────────────
        native_text_blocks: list[dict] = []
        native_table_blocks: list[dict] = []
        page_width  = 595.0   # A4 default
        page_height = 842.0

        if plumber_pages and page_idx < len(plumber_pages):
            pb_tb, pb_tbl, pw, ph = plumber_pages[page_idx]
            native_text_blocks  = pb_tb
            native_table_blocks = pb_tbl
            page_width, page_height = pw, ph

        # Supplement or replace with PyMuPDF if pdfplumber gave less text
        if fitz_pages and page_idx < len(fitz_pages):
            fz_tb, fz_tbl, fw, fh, fz_page = fitz_pages[page_idx]
            page_width, page_height = fw, fh
            fz_chars = sum(len(b.get("text", "")) for b in fz_tb)
            pb_chars = sum(len(b.get("text", "")) for b in native_text_blocks)
            if fz_chars > pb_chars:
                native_text_blocks  = fz_tb
                native_table_blocks = fz_tbl

        # ── Decide extraction path ────────────────────────────────────────────
        use_ocr = force_ocr or _is_low_text(native_text_blocks)

        if not use_ocr:
            # Pure digital page
            try:
                rec = _build_page_record(
                    source_file=str(path.resolve()),
                    page_number=page_num,
                    text_blocks=native_text_blocks,
                    table_blocks=native_table_blocks,
                    page_width=page_width,
                    page_height=page_height,
                    extraction_source=(
                        ExtractionSource.PDFPLUMBER
                        if plumber_pages
                        else ExtractionSource.PYMUPDF
                    ),
                    page_type=PageType.DIGITAL,
                )
                result.pages.append(rec)
            except Exception as exc:
                logger.error("Layout parsing failed page %d: %s", page_num, exc)
                result.errors.append(f"Page {page_num}: {exc}")
                _append_error_record(result, str(path.resolve()), page_num)
            continue

        # ── OCR path ──────────────────────────────────────────────────────────
        pil_image = None
        if fitz_pages and page_idx < len(fitz_pages):
            _, _, _, _, fz_page = fitz_pages[page_idx]
            try:
                pil_image = _render_page_to_pil(fz_page, dpi=ocr_dpi)
            except Exception as exc:
                logger.warning("Page render failed (page %d): %s", page_num, exc)

        if pil_image is None:
            # Can't OCR without an image
            logger.warning("No image available for OCR on page %d", page_num)
            _append_error_record(result, str(path.resolve()), page_num)
            continue

        try:
            ocr_result = ocr_image(
                pil_image,
                engine=ocr_engine,
                preprocess=ocr_preprocess,
            )
        except Exception as exc:
            logger.error("OCR failed page %d: %s", page_num, exc)
            result.errors.append(f"Page {page_num} OCR: {exc}")
            _append_error_record(result, str(path.resolve()), page_num)
            continue

        ocr_blocks = _ocr_blocks_from_result(ocr_result)

        # Determine if mixed (some native text + OCR) or fully scanned
        if native_text_blocks:
            page_type = PageType.MIXED
            all_text_blocks = native_text_blocks + ocr_blocks
        else:
            page_type = PageType.SCANNED
            all_text_blocks = ocr_blocks

        ocr_meta = OCRMetadata(
            engine=ExtractionSource.TESSERACT
                   if ocr_engine == "tesseract"
                   else ExtractionSource.EASYOCR,
            mean_confidence=ocr_result.mean_confidence,
            low_conf_blocks=ocr_result.low_conf_blocks,
            skew_angle_deg=ocr_result.skew_angle_deg,
            preprocessed=ocr_result.preprocessed,
        )

        try:
            rec = _build_page_record(
                source_file=str(path.resolve()),
                page_number=page_num,
                text_blocks=all_text_blocks,
                table_blocks=native_table_blocks,
                page_width=page_width,
                page_height=page_height,
                extraction_source=(
                    ExtractionSource.TESSERACT
                    if ocr_engine == "tesseract"
                    else ExtractionSource.EASYOCR
                ),
                page_type=page_type,
                ocr_meta=ocr_meta,
            )
            result.pages.append(rec)
        except Exception as exc:
            logger.error("Page record build failed (page %d): %s", page_num, exc)
            result.errors.append(f"Page {page_num}: {exc}")
            _append_error_record(result, str(path.resolve()), page_num)

    logger.info(
        "PDF extraction complete: %d pages, %d errors — %s",
        len(result.pages), len(result.errors), path.name,
    )
    return result


def _append_error_record(
    result: DocumentResult,
    source_file: str,
    page_number: int,
) -> None:
    """Append a minimal error-flagged PageRecord to the result."""
    result.pages.append(PageRecord(
        source_file=source_file,
        page_number=page_number,
        is_corrupted=True,
        error_message="Extraction failed; see errors list.",
    ))
