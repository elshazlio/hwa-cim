#!/usr/bin/env python3
"""
Extract: (1) raster XObjects, (2) full-page PNGs, (3) full-page SVGs (vector
schematics), (4) pdfplumber tables — into an output directory with manifest.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber


def table_to_markdown(table: list[list[str | None]]) -> str:
    if not table or not table[0]:
        return ""
    rows = [[(c or "").replace("|", "\\|").strip() for c in row] for row in table]
    widths = [max(len(rows[r][c]) for r in range(len(rows))) for c in range(len(rows[0]))]
    header = "| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(len(rows[0]))) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = "\n".join(
        "| " + " | ".join(rows[r][c].ljust(widths[c]) for c in range(len(rows[r]))) + " |"
        for r in range(1, len(rows))
    )
    return header + "\n" + sep + ("\n" + body if body else "")


def extract(
    pdf_path: Path,
    out_root: Path,
    repo_root: Path,
    page_dpi: float = 300,
    write_page_svg: bool = True,
) -> None:
    pdf_path = pdf_path.resolve()
    out_root = out_root.resolve()
    repo_root = repo_root.resolve()
    images_dir = out_root / "embedded_images"
    pages_dir = out_root / "page_renders"
    svg_dir = out_root / "vector_pages"
    tables_dir = out_root / "tables"
    dirs = [images_dir, pages_dir, tables_dir]
    if write_page_svg:
        dirs.append(svg_dir)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    def rel(p: Path) -> str:
        return str(p.resolve().relative_to(repo_root))

    manifest: dict = {
        "source_pdf": rel(pdf_path),
        "output_root": rel(out_root),
        "page_dpi": page_dpi,
        "note": (
            "embedded_images/ lists only raster XObjects. "
            "Schematics and line art are usually vector paths — use vector_pages/*.svg "
            "or high-DPI page_renders/*.png."
        ),
        "embedded_images": [],
        "page_renders": [],
        "vector_pages_svg": [],
        "tables": [],
    }

    doc = fitz.open(pdf_path)
    zoom = page_dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_no = page_index + 1

        pix_page = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
        page_png = pages_dir / f"page_{page_no:03d}.png"
        pix_page.save(page_png.as_posix())
        manifest["page_renders"].append(
            {
                "page": page_no,
                "path": rel(page_png),
            }
        )

        if write_page_svg:
            svg_text = page.get_svg_image()
            svg_path = svg_dir / f"page_{page_no:03d}.svg"
            svg_path.write_text(svg_text, encoding="utf-8")
            manifest["vector_pages_svg"].append(
                {
                    "page": page_no,
                    "path": rel(svg_path),
                    "path_elements": svg_text.count("<path"),
                }
            )

        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception as e:  # noqa: BLE001
                manifest["embedded_images"].append(
                    {
                        "page": page_no,
                        "index": img_index,
                        "error": str(e),
                    }
                )
                continue
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            out_name = images_dir / f"p{page_no:03d}_img{img_index:02d}.png"
            pix.save(out_name.as_posix())
            manifest["embedded_images"].append(
                {
                    "page": page_no,
                    "index": img_index,
                    "path": rel(out_name),
                    "width": pix.width,
                    "height": pix.height,
                }
            )

    doc.close()

    with pdfplumber.open(pdf_path) as plumber:
        for page_index, page in enumerate(plumber.pages):
            page_no = page_index + 1
            try:
                tables = page.extract_tables() or []
            except Exception as e:  # noqa: BLE001
                manifest["tables"].append(
                    {"page": page_no, "error": str(e)},
                )
                continue
            for t_index, table in enumerate(tables, start=1):
                if not table:
                    continue
                base = f"p{page_no:03d}_table{t_index:02d}"
                md_path = tables_dir / f"{base}.md"
                csv_path = tables_dir / f"{base}.csv"
                md_path.write_text(table_to_markdown(table) + "\n", encoding="utf-8")
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    for row in table:
                        w.writerow([(c or "") for c in row])
                manifest["tables"].append(
                    {
                        "page": page_no,
                        "index": t_index,
                        "markdown": rel(md_path),
                        "csv": rel(csv_path),
                        "rows": len(table),
                        "cols": len(table[0]) if table else 0,
                    }
                )

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_pdf = (
        root
        / "background_info"
        / "Reference_Paper"
        / "A_Charge_Domain_SRAM_Compute-in-Memory_Macro_With_C-2C_Ladder-Based_8-Bit_MAC_Unit_in_22-nm_FinFET_Process_for_Edge_Inference.pdf"
    )
    default_out = root / "background_info" / "Reference_Paper" / "extracted_for_agents"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", nargs="?", type=Path, default=default_pdf)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_out,
        help="Output directory (default: Reference_Paper/extracted_for_agents)",
    )
    ap.add_argument(
        "--dpi",
        type=float,
        default=300.0,
        help="Raster resolution for page_renders (default 300).",
    )
    ap.add_argument(
        "--no-svg",
        action="store_true",
        help="Skip vector_pages/*.svg (full-page SVG, preserves schematic line art).",
    )
    args = ap.parse_args()

    if not args.pdf.is_file():
        raise SystemExit(f"PDF not found: {args.pdf}")
    extract(
        args.pdf,
        args.output,
        repo_root=root,
        page_dpi=args.dpi,
        write_page_svg=not args.no_svg,
    )
    print(f"Wrote assets under {args.output}")


if __name__ == "__main__":
    main()
