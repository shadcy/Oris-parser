"""
core.schema
===========
Canonical output schema (Pydantic v2) for every page/slide extracted by the
document parsing pipeline.  All downstream exporters consume these models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field, model_validator


# ── Enumerations ─────────────────────────────────────────────────────────────

class PageType(str, Enum):
    """Broad classification of the page/slide content origin."""
    DIGITAL   = "digital"    # Native text layer present
    SCANNED   = "scanned"    # No native text; OCR was used
    MIXED     = "mixed"      # Partial native text + OCR fills
    EMPTY     = "empty"      # Blank page / slide


class ExtractionSource(str, Enum):
    """Which library/method produced the text for this unit."""
    PDFPLUMBER = "pdfplumber"
    PYMUPDF    = "pymupdf"
    PYTHON_PPTX = "python_pptx"
    TESSERACT  = "tesseract"
    EASYOCR    = "easyocr"
    NONE       = "none"


# ── Sub-models ────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """Normalised bounding box (0.0 – 1.0 relative to page dimensions)."""
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _validate_coords(self) -> "BoundingBox":
        if self.x0 > self.x1 or self.y0 > self.y1:
            raise ValueError("BoundingBox coordinates are inverted.")
        return self


class TextSpan(BaseModel):
    """Atomic run of text with optional inline formatting hints."""
    text: str
    bold: bool = False
    italic: bool = False
    font_size: Optional[float] = None
    font_name: Optional[str]   = None


class BulletItem(BaseModel):
    """A single bullet / list item, potentially nested."""
    level: int = 0                        # Indentation depth (0 = top level)
    text: str
    spans: List[TextSpan] = Field(default_factory=list)
    children: List["BulletItem"] = Field(default_factory=list)


class TableCell(BaseModel):
    text: str
    row_span: int = 1
    col_span: int = 1


class Table(BaseModel):
    """A table extracted from the page, stored as a list of rows."""
    caption: Optional[str] = None
    headers:  List[str]              = Field(default_factory=list)
    rows:     List[List[TableCell]]  = Field(default_factory=list)
    bbox:     Optional[BoundingBox]  = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "caption": self.caption,
            "headers": self.headers,
            "rows": [
                [{"text": c.text, "row_span": c.row_span, "col_span": c.col_span}
                 for c in row]
                for row in self.rows
            ],
        }


class Section(BaseModel):
    """A logical block / section within a page."""
    heading:  Optional[str]       = None
    body:     str                 = ""
    spans:    List[TextSpan]      = Field(default_factory=list)
    bullets:  List[BulletItem]    = Field(default_factory=list)
    tables:   List[Table]         = Field(default_factory=list)
    bbox:     Optional[BoundingBox] = None
    column:   int = 0             # Column index in multi-column layouts


class OCRMetadata(BaseModel):
    """Diagnostics produced by the OCR fallback pipeline."""
    engine:         ExtractionSource
    mean_confidence: float = 0.0     # 0–100
    low_conf_blocks: int   = 0       # Blocks below confidence threshold
    skew_angle_deg:  float = 0.0
    preprocessed:    bool  = False


# ── Primary record ────────────────────────────────────────────────────────────

class PageRecord(BaseModel):
    """
    Canonical structured representation for one page (PDF) or slide (PPTX).
    """
    # Identity
    source_file:   str
    page_number:   int                    # 1-based
    page_type:     PageType               = PageType.DIGITAL
    extraction_source: ExtractionSource  = ExtractionSource.NONE

    # High-level structure
    title:         Optional[str]          = None
    subtitle:      Optional[str]          = None
    sections:      List[Section]          = Field(default_factory=list)
    bullets:       List[BulletItem]       = Field(default_factory=list)  # top-level bullets
    tables:        List[Table]            = Field(default_factory=list)  # top-level tables
    images:        List[str]              = Field(default_factory=list)  # image descriptions / paths

    # Raw fallback
    raw_text:      str                    = ""

    # Layout hints
    page_width:    Optional[float]        = None
    page_height:   Optional[float]        = None
    columns_detected: int                 = 1

    # OCR diagnostics (only present when OCR was used)
    ocr_metadata:  Optional[OCRMetadata]  = None

    # Processing flags
    is_corrupted:  bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict suitable for JSON export."""
        d = self.model_dump(exclude_none=False)
        # Replace Table objects with their dict form inside sections
        d["tables"] = [t.to_dict() for t in self.tables]
        for i, sec in enumerate(self.sections):
            d["sections"][i]["tables"] = [t.to_dict() for t in sec.tables]
        return d


# ── Document-level wrapper ────────────────────────────────────────────────────

class DocumentResult(BaseModel):
    """Aggregated output for an entire file."""
    source_file:   str
    file_type:     str                       # "pdf" | "pptx" | "ppt"
    total_pages:   int
    pages:         List[PageRecord]          = Field(default_factory=list)
    errors:        List[str]                 = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file":  self.source_file,
            "file_type":    self.file_type,
            "total_pages":  self.total_pages,
            "pages":        [p.to_dict() for p in self.pages],
            "errors":       self.errors,
        }
