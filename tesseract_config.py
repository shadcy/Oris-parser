"""
tesseract_config.py
====================
Zero-PATH Tesseract configuration.

Drop the Tesseract portable folder inside the project as:

    Oris/
    └── tesseract/
        ├── tesseract.exe          ← the binary
        ├── tessdata/
        │   └── eng.traineddata    ← language model(s)
        └── ... (DLLs etc.)

This module auto-detects the binary in that location and points
pytesseract at it so you NEVER need to touch system PATH or environment
variables.

If the local bundle is not found, it falls back gracefully to:
  1. Common Windows install paths (Program Files/Tesseract-OCR/)
  2. Whatever 'tesseract' resolves to on the system PATH
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

# This file sits at Oris/tesseract_config.py → parent == Oris/
_PROJECT_ROOT = Path(__file__).resolve().parent

_LOCAL_TESS_DIR = _PROJECT_ROOT / "tesseract"            # Oris/tesseract/
_LOCAL_TESS_EXE = _LOCAL_TESS_DIR / "tesseract.exe"     # Windows executable
_LOCAL_TESSDATA = _LOCAL_TESS_DIR / "tessdata"           # language model folder

_CONFIGURED = False   # guard: only run once


def configure_tesseract() -> None:
    """
    Call ONCE at application startup (before any OCR).
    Idempotent – safe to call multiple times.

    Priority order:
      1. Oris/tesseract/tesseract.exe  (project-local portable bundle)
      2. C:/Program Files/Tesseract-OCR/tesseract.exe
      3. C:/Program Files (x86)/Tesseract-OCR/tesseract.exe
      4. Tesseract on system PATH (standard install / Linux / macOS)
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract not installed – OCR will not be available.")
        return

    # 1. Project-local portable binary (highest priority)
    if _LOCAL_TESS_EXE.exists():
        pytesseract.pytesseract.tesseract_cmd = str(_LOCAL_TESS_EXE)
        os.environ.setdefault("TESSDATA_PREFIX", str(_LOCAL_TESSDATA))
        logger.info("Tesseract: using project-local bundle at %s", _LOCAL_TESS_EXE)
        return

    # 2 & 3. Common Windows install paths (no PATH update needed)
    win_paths = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for fp in win_paths:
        if fp.exists():
            pytesseract.pytesseract.tesseract_cmd = str(fp)
            logger.info("Tesseract: found at %s", fp)
            return

    # 4. Fall back to PATH (Linux / macOS / if manually installed on PATH)
    import shutil
    if shutil.which("tesseract"):
        logger.info("Tesseract: using system PATH entry.")
    else:
        logger.warning(
            "Tesseract binary not found.\n"
            "  Option A (easiest): place the portable folder at:\n"
            "    %s\n"
            "  Option B: install from https://github.com/UB-Mannheim/tesseract/wiki\n",
            _LOCAL_TESS_DIR,
        )


def tesseract_status() -> dict:
    """
    Return a diagnostic dict describing the current Tesseract setup.
    Useful for debugging / health checks.
    """
    import shutil

    local_bundle = _LOCAL_TESS_EXE.exists()
    binary_path  = "tesseract (PATH)"

    try:
        import pytesseract
        binary_path = pytesseract.pytesseract.tesseract_cmd
    except ImportError:
        pass

    available = Path(binary_path).exists() or bool(shutil.which("tesseract"))
    tessdata  = os.environ.get("TESSDATA_PREFIX", "")

    return {
        "binary_path":      binary_path,
        "tessdata_prefix":  tessdata,
        "local_bundle":     local_bundle,
        "local_bundle_path": str(_LOCAL_TESS_DIR),
        "available":        available,
    }
