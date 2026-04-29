"""
main.py – Document Parser CLI
==============================
Batch-processes PDF and PPTX/PPT files, extracting page/slide-wise
structured content and exporting to JSON (and optionally Markdown).

Usage examples
--------------
# Single file, JSON only:
    python main.py process path/to/doc.pdf

# Single file, JSON + Markdown:
    python main.py process path/to/deck.pptx --markdown

# Batch directory, force OCR on everything, EasyOCR engine:
    python main.py process ./docs/ --markdown --ocr-engine easyocr --force-ocr

# Batch directory with custom output directory:
    python main.py process ./docs/ --output ./results/ --markdown

# Show version:
    python main.py --version
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt

__version__ = "1.0.0"

# ── Logging setup ─────────────────────────────────────────────────────────────

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    # Quieten noisy third-party loggers
    for lib in ("pdfminer", "PIL", "easyocr", "torch", "fitz"):
        logging.getLogger(lib).setLevel(logging.WARNING)


console = Console()

# ── Supported extensions ──────────────────────────────────────────────────────

PDF_EXTS  = {".pdf"}
PPTX_EXTS = {".pptx", ".ppt"}
ALL_EXTS  = PDF_EXTS | PPTX_EXTS


def _collect_files(input_path: Path) -> list[Path]:
    """Collect all supported files from a path (file or directory)."""
    if input_path.is_file():
        if input_path.suffix.lower() in ALL_EXTS:
            return [input_path]
        console.print(
            f"[red]Unsupported file type:[/red] {input_path.suffix} "
            f"(supported: {', '.join(sorted(ALL_EXTS))})"
        )
        return []
    elif input_path.is_dir():
        files = sorted(
            f for f in input_path.rglob("*")
            if f.suffix.lower() in ALL_EXTS and f.is_file()
        )
        if not files:
            console.print(f"[yellow]No supported files found in[/yellow] {input_path}")
        return files
    else:
        console.print(f"[red]Path does not exist:[/red] {input_path}")
        return []


def _determine_output_path(input_file: Path, output_dir: Path, ext: str) -> Path:
    """Build the output file path, preserving the source file stem."""
    return output_dir / (input_file.stem + ext)


# ── Processing logic ──────────────────────────────────────────────────────────

def _process_one(
    file_path: Path,
    output_dir: Path,
    *,
    ocr_engine: str,
    ocr_preprocess: bool,
    ocr_dpi: int,
    force_ocr: bool,
    export_markdown: bool,
) -> dict:
    """
    Process a single file and write output(s).

    Returns a summary dict with keys:
      file, pages, errors, json_out, md_out (optional)
    """
    from core.pdf_extractor import extract_pdf
    from core.ppt_extractor import extract_pptx
    from exporters.json_exporter import export_json
    from exporters.markdown_exporter import export_markdown as _export_md

    ext = file_path.suffix.lower()
    summary: dict = {
        "file":   str(file_path),
        "pages":  0,
        "errors": 0,
        "json_out": None,
        "md_out":   None,
    }

    try:
        if ext in PDF_EXTS:
            result = extract_pdf(
                str(file_path),
                ocr_engine=ocr_engine,
                ocr_preprocess=ocr_preprocess,
                ocr_dpi=ocr_dpi,
                force_ocr=force_ocr,
            )
        elif ext in PPTX_EXTS:
            result = extract_pptx(
                str(file_path),
                ocr_engine=ocr_engine,
                ocr_preprocess=ocr_preprocess,
            )
        else:
            summary["errors"] = 1
            return summary

        # JSON export (always)
        json_out = _determine_output_path(file_path, output_dir, ".json")
        export_json(result, str(json_out))
        summary["json_out"] = str(json_out)

        # Markdown export (optional)
        if export_markdown:
            md_out = _determine_output_path(file_path, output_dir, ".md")
            _export_md(result, str(md_out))
            summary["md_out"] = str(md_out)

        summary["pages"]  = len(result.pages)
        summary["errors"] = len(result.errors)

    except Exception as exc:
        logging.getLogger(__name__).error(
            "Fatal error processing %s: %s", file_path, exc, exc_info=True
        )
        summary["errors"] += 1

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(__version__, prog_name="docparser")
def cli() -> None:
    """Document Parser – extract structured text from PDFs and PPTX files."""


@cli.command("process")
@click.argument(
    "input_path",
    type=click.Path(exists=True, readable=True, path_type=Path),
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for JSON/Markdown files. "
         "Defaults to '<input_dir>/output/' or './<file_stem>_output/'.",
)
@click.option(
    "--markdown", "-m",
    is_flag=True,
    default=False,
    help="Also export a Markdown file alongside the JSON.",
)
@click.option(
    "--ocr-engine",
    type=click.Choice(["tesseract", "easyocr"], case_sensitive=False),
    default="tesseract",
    show_default=True,
    help="OCR engine to use for scanned/image pages.",
)
@click.option(
    "--no-preprocess",
    is_flag=True,
    default=False,
    help="Skip image preprocessing (denoising, thresholding, skew correction) "
         "before OCR.  Faster but less accurate for low-quality scans.",
)
@click.option(
    "--ocr-dpi",
    type=int,
    default=200,
    show_default=True,
    help="DPI for rendering PDF pages to images for OCR.",
)
@click.option(
    "--force-ocr",
    is_flag=True,
    default=False,
    help="Force OCR on every page even if native text is available.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable DEBUG-level logging.",
)
def process_command(
    input_path: Path,
    output: Path | None,
    markdown: bool,
    ocr_engine: str,
    no_preprocess: bool,
    ocr_dpi: int,
    force_ocr: bool,
    verbose: bool,
) -> None:
    """
    Extract structured content from INPUT_PATH (file or directory).

    Supported formats:
      - PDF   (.pdf)
      - PowerPoint (.pptx, .ppt)

    Outputs are written to JSON, and optionally Markdown (--markdown).
    """
    _configure_logging(verbose)

    # ── Resolve output directory ───────────────────────────────────────────────
    if output is None:
        if input_path.is_dir():
            output = input_path / "output"
        else:
            output = input_path.parent / (input_path.stem + "_output")
    output.mkdir(parents=True, exist_ok=True)

    # ── Collect files ──────────────────────────────────────────────────────────
    files = _collect_files(input_path)
    if not files:
        sys.exit(1)

    console.print(
        f"\n[bold cyan]Document Parser[/bold cyan] v{__version__}\n"
        f"  Files found : [yellow]{len(files)}[/yellow]\n"
        f"  Output dir  : [green]{output}[/green]\n"
        f"  OCR engine  : [blue]{ocr_engine}[/blue]\n"
        f"  Markdown    : {'[green]yes[/green]' if markdown else '[dim]no[/dim]'}\n"
    )

    summaries: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Processing files…", total=len(files))

        for file_path in files:
            progress.update(task, description=f"[cyan]{file_path.name}[/cyan]")
            summary = _process_one(
                file_path,
                output,
                ocr_engine=ocr_engine,
                ocr_preprocess=not no_preprocess,
                ocr_dpi=ocr_dpi,
                force_ocr=force_ocr,
                export_markdown=markdown,
            )
            summaries.append(summary)
            progress.advance(task)

    # ── Summary table ──────────────────────────────────────────────────────────
    console.print()
    tbl = Table(title="Extraction Summary", show_lines=True)
    tbl.add_column("File",    style="cyan",  no_wrap=True)
    tbl.add_column("Pages",   justify="right")
    tbl.add_column("Errors",  justify="right")
    tbl.add_column("JSON",    style="green")
    tbl.add_column("Markdown", style="magenta")

    total_pages  = 0
    total_errors = 0
    for s in summaries:
        err_style = "red" if s["errors"] else "green"
        tbl.add_row(
            Path(s["file"]).name,
            str(s["pages"]),
            f"[{err_style}]{s['errors']}[/{err_style}]",
            Path(s["json_out"]).name if s.get("json_out") else "-",
            Path(s["md_out"]).name   if s.get("md_out")   else "-",
        )
        total_pages  += s["pages"]
        total_errors += s["errors"]

    console.print(tbl)
    console.print(
        f"\n[bold]Total[/bold]: {len(summaries)} file(s), "
        f"{total_pages} page(s), "
        f"{'[red]' if total_errors else '[green]'}{total_errors} error(s)"
        f"{'[/red]' if total_errors else '[/green]'}\n"
    )

    if total_errors:
        sys.exit(2)


@cli.command("status")
def status_command() -> None:
    """Check the health and configuration of the document parser."""
    from tesseract_config import tesseract_status
    
    console.print("\n[bold cyan]System Status[/bold cyan]")
    
    # OCR Status
    stat = tesseract_status()
    t_style = "green" if stat["available"] else "red"
    
    console.print(f"\n[bold]OCR Engine (Tesseract):[/bold]")
    console.print(f"  Available     : [{t_style}]{stat['available']}[/{t_style}]")
    console.print(f"  Binary Path   : [blue]{stat['binary_path']}[/blue]")
    console.print(f"  Tessdata Pref : [blue]{stat['tessdata_prefix'] or 'Not Set'}[/blue]")
    console.print(f"  Local Bundle  : {'[green]Yes[/green]' if stat['local_bundle'] else '[yellow]No (Using System)[/yellow]'}")
    
    if not stat["available"]:
        console.print(f"\n[red]Warning:[/red] Tesseract not found. Native PDF/PPTX will work, but OCR fallback will fail.")
        console.print(f"To fix, place Tesseract in: [blue]{stat['local_bundle_path']}[/blue]")

    # Dependencies
    console.print(f"\n[bold]Supported Formats:[/bold]")
    console.print(f"  PDF           : [green]pdfplumber, PyMuPDF[/green]")
    console.print(f"  PowerPoint    : [green]python-pptx[/green]")
    console.print("")


@cli.command("interactive")
@click.pass_context
def interactive_command(ctx: click.Context) -> None:
    """Launch a guided terminal interface to parse documents."""
    console.print("\n[bold magenta]Welcome to the Oris Interactive Parser[/bold magenta]")
    console.print("[dim]This wizard will guide you through the document parsing process.[/dim]\n")

    # 1. Input Path
    while True:
        raw_path = Prompt.ask("[bold]Enter the path to a file or directory[/bold]")
        input_path = Path(raw_path).expanduser().resolve()
        if input_path.exists():
            break
        console.print(f"[red]Error:[/red] Path [yellow]{input_path}[/yellow] does not exist. Try again.")

    # 2. Output Path
    default_out = "out"
    raw_out = Prompt.ask("[bold]Enter output directory[/bold]", default=default_out)
    output_path = Path(raw_out).expanduser().resolve()

    # 3. Features
    export_markdown = Confirm.ask("Export to Markdown as well as JSON?", default=True)
    
    # 4. OCR Settings
    use_ocr = Confirm.ask("Enable OCR fallback for scanned pages?", default=True)
    ocr_engine = "tesseract"
    ocr_dpi = 200
    force_ocr = False
    
    if use_ocr:
        ocr_engine = Prompt.ask(
            "Choose OCR Engine", 
            choices=["tesseract", "easyocr"], 
            default="tesseract"
        )
        ocr_dpi = IntPrompt.ask("OCR Resolution (DPI)", default=200)
        force_ocr = Confirm.ask("Force OCR on all pages (even digital)?", default=False)

    verbose = Confirm.ask("Enable detailed debug logging?", default=False)

    console.print("\n[bold cyan]Starting process...[/bold cyan]\n")
    
    # Forward to the process command
    ctx.invoke(
        process_command,
        input_path=input_path,
        output=output_path,
        markdown=export_markdown,
        ocr_engine=ocr_engine,
        no_preprocess=False,
        ocr_dpi=ocr_dpi,
        force_ocr=force_ocr,
        verbose=verbose
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
