"""
core.ocr_pipeline
=================
OCR fallback pipeline with full image pre-processing:
  1. Grayscale conversion
  2. Denoising (Non-Local Means via OpenCV)
  3. Adaptive / Otsu thresholding
  4. Skew detection & correction (Hough-transform based)
  5. Tesseract / EasyOCR extraction with per-word confidence scoring
  6. Post-processing: filtering low-confidence blocks, merging lines

The module exposes a single public function ``ocr_image`` that accepts a
PIL Image and returns a list of text-block dicts plus OCR metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Auto-configure Tesseract from the project-local portable binary (no PATH needed)
try:
    import sys as _sys
    import os as _os
    _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from tesseract_config import configure_tesseract as _configure_tesseract
    _configure_tesseract()
except Exception:
    pass   # If config module is missing, fall back to PATH-based Tesseract

# Lazy imports – only loaded when actually needed
_cv2: Optional[object] = None
_pytesseract: Optional[object] = None
_easyocr_reader: Optional[object] = None  # singleton

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

OCR_MIN_CONFIDENCE = 40      # Tesseract word confidence threshold (0–100)
EASYOCR_MIN_CONFIDENCE = 0.4 # EasyOCR paragraph confidence (0–1)
SKEW_ANGLE_LIMIT = 45.0      # Max skew angle to attempt correction (degrees)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class OCRBlock:
    """A single recognised text block from the OCR engine."""
    text:       str
    confidence: float          # 0–100
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    level: int = 0             # hierarchy level (Tesseract tsv level)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "x0": self.x0, "y0": self.y0,
            "x1": self.x1, "y1": self.y1,
        }


@dataclass
class OCRResult:
    blocks: list[OCRBlock] = field(default_factory=list)
    mean_confidence: float = 0.0
    low_conf_blocks: int   = 0
    skew_angle_deg:  float = 0.0
    engine_used:     str   = "none"
    preprocessed:    bool  = False


# ── Private helpers ───────────────────────────────────────────────────────────

def _import_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as _cv2_mod
        _cv2 = _cv2_mod
    return _cv2


def _import_tesseract():
    global _pytesseract
    if _pytesseract is None:
        import pytesseract as _tess
        _pytesseract = _tess
    return _pytesseract


def _pil_to_numpy(pil_image) -> np.ndarray:
    """Convert a PIL RGBA/RGB/L image to a BGR uint8 numpy array."""
    import numpy as np
    from PIL import Image
    img = pil_image.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    cv2 = _import_cv2()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _to_grayscale(bgr: np.ndarray) -> np.ndarray:
    cv2 = _import_cv2()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _denoise(gray: np.ndarray) -> np.ndarray:
    """Fast non-local-means denoising suitable for document scans."""
    cv2 = _import_cv2()
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def _threshold(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian thresholding is preferred for documents with uneven
    lighting; fall back to Otsu for uniformly-lit scans.
    """
    cv2 = _import_cv2()
    # Try Otsu first (fast)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Adaptive threshold retains detail in shadow areas
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    # Choose the one with higher mean (more white = cleaner background)
    if np.mean(otsu) >= np.mean(adaptive):
        return otsu
    return adaptive


def _detect_skew(binary: np.ndarray) -> float:
    """
    Estimate document skew angle using the Hough line transform on edge image.
    Returns angle in degrees; positive = clockwise tilt.
    """
    cv2 = _import_cv2()
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return 0.0

    angles: list[float] = []
    for line in lines[:50]:   # Only inspect the strongest 50 lines
        rho, theta = line[0]
        # Convert Hough theta to skew angle
        angle = float(np.degrees(theta) - 90.0)
        if abs(angle) < SKEW_ANGLE_LIMIT:
            angles.append(angle)

    return float(np.median(angles)) if angles else 0.0


def _correct_skew(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate the image to correct a detected skew angle."""
    if abs(angle_deg) < 0.3:
        return bgr
    cv2 = _import_cv2()
    h, w = bgr.shape[:2]
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def _preprocess(pil_image) -> tuple[np.ndarray, float]:
    """
    Full pre-processing chain.
    Returns: (processed_bgr, skew_angle_deg)
    """
    bgr   = _pil_to_numpy(pil_image)
    gray  = _to_grayscale(bgr)
    gray  = _denoise(gray)
    binary = _threshold(gray)
    angle  = _detect_skew(binary)
    if abs(angle) >= 0.3:
        bgr = _correct_skew(bgr, angle)
    return bgr, angle


def _numpy_to_pil(bgr: np.ndarray):
    """Convert BGR numpy array back to PIL RGB Image."""
    from PIL import Image
    cv2 = _import_cv2()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ── Tesseract engine ──────────────────────────────────────────────────────────

def _run_tesseract(pil_image) -> OCRResult:
    """
    Run pytesseract on a PIL image and return structured OCRResult.
    Uses TSV output for per-word confidence scores.
    """
    tess = _import_tesseract()
    import pandas as pd

    config = "--oem 3 --psm 6"   # LSTM engine, uniform block of text
    try:
        df = tess.image_to_data(
            pil_image,
            config=config,
            output_type=tess.Output.DATAFRAME,
        )
    except Exception as exc:
        logger.warning("Tesseract failed: %s", exc)
        return OCRResult(engine_used="tesseract")

    # Filter to word-level rows with valid text
    df = df[df["level"] == 5].copy()               # level 5 = word
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(0)

    blocks: list[OCRBlock] = []
    for _, row in df.iterrows():
        conf = float(row["conf"])
        text = str(row["text"]).strip()
        if not text:
            continue
        blocks.append(OCRBlock(
            text=text,
            confidence=conf,
            x0=float(row.get("left", 0)),
            y0=float(row.get("top", 0)),
            x1=float(row.get("left", 0)) + float(row.get("width", 0)),
            y1=float(row.get("top", 0)) + float(row.get("height", 0)),
            level=int(row.get("level", 5)),
        ))

    confs = [b.confidence for b in blocks if b.confidence >= 0]
    mean_conf = float(np.mean(confs)) if confs else 0.0
    low_conf  = sum(1 for c in confs if c < OCR_MIN_CONFIDENCE)

    # Merge word-level blocks into line-level blocks for downstream consumers
    merged = _merge_words_to_lines(blocks)

    return OCRResult(
        blocks=merged,
        mean_confidence=mean_conf,
        low_conf_blocks=low_conf,
        engine_used="tesseract",
    )


def _merge_words_to_lines(
    blocks: list[OCRBlock],
    y_tolerance: float = 8.0,
) -> list[OCRBlock]:
    """
    Group word-level OCRBlocks that share approximately the same y-range
    into line-level blocks.  Words on the same line are concatenated with a
    space and the bounding box is expanded to cover all words.
    """
    if not blocks:
        return []

    # Sort by (y0, x0)
    sorted_blocks = sorted(blocks, key=lambda b: (b.y0, b.x0))
    lines: list[list[OCRBlock]] = []
    current_line: list[OCRBlock] = [sorted_blocks[0]]

    for block in sorted_blocks[1:]:
        last = current_line[-1]
        # Same line if centres are within tolerance
        cy_last  = (last.y0  + last.y1)  / 2.0
        cy_block = (block.y0 + block.y1) / 2.0
        if abs(cy_block - cy_last) <= y_tolerance:
            current_line.append(block)
        else:
            lines.append(current_line)
            current_line = [block]
    lines.append(current_line)

    merged: list[OCRBlock] = []
    for line in lines:
        line_sorted = sorted(line, key=lambda b: b.x0)
        text   = " ".join(b.text for b in line_sorted)
        x0     = min(b.x0 for b in line_sorted)
        y0     = min(b.y0 for b in line_sorted)
        x1     = max(b.x1 for b in line_sorted)
        y1     = max(b.y1 for b in line_sorted)
        confs  = [b.confidence for b in line_sorted]
        mean_c = float(np.mean(confs)) if confs else 0.0
        merged.append(OCRBlock(text=text, confidence=mean_c,
                               x0=x0, y0=y0, x1=x1, y1=y1))
    return merged


# ── EasyOCR engine ────────────────────────────────────────────────────────────

def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


def _run_easyocr(pil_image) -> OCRResult:
    """
    Run EasyOCR on a PIL image and return structured OCRResult.
    """
    try:
        reader = _get_easyocr_reader()
    except Exception as exc:
        logger.warning("EasyOCR init failed: %s", exc)
        return OCRResult(engine_used="easyocr")

    import numpy as np
    img_arr = np.array(pil_image.convert("RGB"))
    try:
        results = reader.readtext(img_arr, detail=1, paragraph=False)
    except Exception as exc:
        logger.warning("EasyOCR readtext failed: %s", exc)
        return OCRResult(engine_used="easyocr")

    blocks: list[OCRBlock] = []
    for bbox_pts, text, conf in results:
        conf_pct = float(conf) * 100.0
        # bbox_pts is [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        blocks.append(OCRBlock(
            text=str(text).strip(),
            confidence=conf_pct,
            x0=float(min(xs)), y0=float(min(ys)),
            x1=float(max(xs)), y1=float(max(ys)),
        ))

    confs = [b.confidence for b in blocks]
    mean_conf = float(np.mean(confs)) if confs else 0.0
    low_conf  = sum(1 for c in confs if c < OCR_MIN_CONFIDENCE)

    return OCRResult(
        blocks=blocks,
        mean_confidence=mean_conf,
        low_conf_blocks=low_conf,
        engine_used="easyocr",
    )


# ── Public API ────────────────────────────────────────────────────────────────

def ocr_image(
    pil_image,
    *,
    engine: str = "tesseract",   # "tesseract" | "easyocr"
    preprocess: bool = True,
) -> OCRResult:
    """
    Run the full OCR pipeline on a PIL Image.

    Args:
        pil_image:   PIL.Image.Image object (any mode; will be converted).
        engine:      Which OCR engine to use ("tesseract" or "easyocr").
        preprocess:  Whether to apply the full image-preprocessing chain
                     (denoising, thresholding, skew correction).

    Returns:
        OCRResult with blocks, confidence stats, and diagnostics.
    """
    skew_angle = 0.0
    preprocessed = False

    if preprocess:
        try:
            bgr, skew_angle = _preprocess(pil_image)
            pil_image = _numpy_to_pil(bgr)
            preprocessed = True
        except Exception as exc:
            logger.warning("Preprocessing failed, using raw image: %s", exc)

    if engine == "easyocr":
        result = _run_easyocr(pil_image)
    else:
        result = _run_tesseract(pil_image)

    result.skew_angle_deg = skew_angle
    result.preprocessed   = preprocessed

    # Filter blocks below confidence threshold
    result.blocks = [
        b for b in result.blocks
        if b.confidence >= OCR_MIN_CONFIDENCE and b.text.strip()
    ]

    logger.debug(
        "OCR[%s]: %d blocks, mean_conf=%.1f, skew=%.1f°",
        result.engine_used, len(result.blocks),
        result.mean_confidence, result.skew_angle_deg,
    )
    return result


def is_scanned_page(pil_image, *, text_char_threshold: int = 20) -> bool:
    """
    Heuristic to decide if a page is scanned (no digital text layer) by
    attempting a quick Tesseract pass and checking the character yield.

    Returns True when the page should be treated as scanned.
    """
    try:
        tess = _import_tesseract()
        raw = tess.image_to_string(pil_image, config="--psm 6")
        chars = len(raw.strip())
        return chars < text_char_threshold
    except Exception:
        return True  # Assume scanned if Tesseract is unavailable
