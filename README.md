# Oris: Robust Document Parser

Oris is a production-level, layout-aware document parser designed to extract structured content from **PDF** and **PPT/PPTX** files. It bridges the gap between digital text extraction and OCR fallback, ensuring that even scanned documents or complex slide decks are parsed into a consistent, hierarchical format.

## 🚀 Key Features

- **Multi-Format Support**: Native processing for PDFs (via pdfplumber/PyMuPDF) and PowerPoint files (via python-pptx).
- **Layout-Aware Extraction**: Automatically detects multi-column layouts, headings, subtitles, and logical sections.
- **Smart OCR Fallback**: Automatically detects scanned pages vs. digital pages. It uses a robust OCR pipeline (Tesseract/EasyOCR) with advanced preprocessing (denoising, thresholding, skew correction).
- **Hierarchical Structure**: Captures bullet points (including nesting), tables (converted to structured JSON/Markdown), and inline formatting cues.
- **Consistent Schema**: Every page or slide is exported as a standard `PageRecord` object.
- **Dual Export**: Generate machine-readable **JSON** and human-readable **Markdown** simultaneously.
- **Batch Processing**: Process single files or entire directories with ease.
- **Zero-PATH Configuration**: Dedicated system to use a project-local Tesseract binary without needing to modify system environment variables.

---

## 🛠️ Technology Stack

- **Core**: Python 3.8+
- **PDF Extraction**: `pdfplumber`, `PyMuPDF` (fitz)
- **PPTX Extraction**: `python-pptx`
- **OCR Engine**: `pytesseract` (Tesseract) and `easyocr`
- **Image Processing**: `OpenCV`, `Pillow`
- **Schema & Validation**: `Pydantic v2`
- **CLI & UI**: `Click`, `Rich` (for progress bars and tables)

---

## 📂 Project Structure

```
Oris/
├── main.py                  # CLI entry point & batch runner
├── tesseract_config.py      # Zero-PATH Tesseract configuration
├── core/
│   ├── pdf_extractor.py     # PDF native + OCR pipeline
│   ├── ppt_extractor.py     # PPTX native extraction
│   ├── ocr_pipeline.py      # OCR fallback with preprocessing
│   ├── layout_parser.py     # Bounding box / layout grouping
│   ├── schema.py            # Pydantic output models (Python 3.8 compatible)
│   └── utils.py             # Helpers & text normalization
├── exporters/
│   ├── json_exporter.py     # JSON serialization
│   └── markdown_exporter.py # Markdown rendering
├── requirements.txt         # Project dependencies
└── out/                     # Default output directory
```

---

## ⚙️ Installation

### 1. Clone & Install Dependencies
It is recommended to use the `radioconda` environment as tested:
```bash
c:\users\asus\radioconda\python.exe -m pip install -r requirements.txt
```

### 2. Tesseract OCR (Optional but Recommended)
Oris is designed to work without modifying your system `PATH`.
- **Portable Method**: Download the Tesseract binaries and place them in an `Oris/tesseract/` folder. The system will automatically detect `Oris/tesseract/tesseract.exe`.
- **Standard Method**: Install Tesseract normally. Oris will check common installation paths like `C:\Program Files\Tesseract-OCR\`.

---

## 📖 Usage

### CLI Commands
Run the parser using your Python interpreter:

```bash
# Basic usage (processes folder 'test' and outputs to 'out')
python main.py process test/ --output out/ --markdown

# Force OCR on all pages (even if digital text exists)
python main.py process documents/ --force-ocr --ocr-dpi 300

# Use EasyOCR engine instead of Tesseract
python main.py process test/ --ocr-engine easyocr

# Full list of options
python main.py process --help
```

### CLI Flags
| Flag | Description |
|---|---|
| `--output`, `-o` | Directory to save JSON/MD files. |
| `--markdown`, `-m` | Export Markdown files alongside JSON. |
| `--ocr-engine` | Choose between `tesseract` or `easyocr`. |
| `--ocr-dpi` | Resolution for rendering pages for OCR (Default: 200). |
| `--force-ocr` | Skip native text and use OCR for everything. |
| `--no-preprocess` | Skip image cleanup (denoising/skew correction). |
| `--verbose`, `-v` | Enable detailed debug logging. |

---

## 📊 Output Example (JSON)

Each page is parsed into a structured record:
```json
{
  "page_number": 1,
  "page_type": "digital",
  "title": "Document Introduction",
  "sections": [
    {
      "heading": "Summary",
      "body": "This document contains...",
      "bullets": [{"text": "Key point 1", "level": 0}],
      "tables": []
    }
  ],
  "raw_text": "..."
}
```

---

## 🔍 OCR Pipeline Details
When a page is scanned or lacks text, Oris triggers the following pipeline:
1. **Denoising**: Removes scanner noise using Non-Local Means.
2. **Thresholding**: Adaptive Gaussian thresholding for high contrast.
3. **Skew Correction**: Detects page tilt and rotates it back to horizontal.
4. **Recognition**: Extracts text using the chosen OCR engine.
5. **Layout Parsing**: Reconstructs the reading order from the raw OCR coordinates.

---

## 📜 License
This project is built for high-accuracy document parsing and research applications.
