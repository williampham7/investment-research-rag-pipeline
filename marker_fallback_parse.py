"""
Re-parse the papers pdf_parser.py flagged as needs_fallback_parser=True using
marker instead of PyMuPDF. Marker's layout model correctly separates headings,
equations, and table rows (PyMuPDF's font-heuristic conflates them), and its
table recognizer finds borderless/booktabs-style tables PyMuPDF's grid-line
detector misses.

Converts marker's block tree into the same schema pdf_parser.py produces
(pages/sections/tables/diagnostics) so parsed_papers/ stays uniform regardless
of which parser produced a given file, and overwrites just the flagged papers'
JSON files in place -- the clean PyMuPDF ones are left untouched.

These are all born-digital LaTeX PDFs, not scans, so --disable_ocr-equivalent
config is used: full-page OCR was taking ~13 min for a 10-page paper (vs ~20s
with direct text-layer extraction) because marker's OCR-error heuristic
flagged dense math notation as "garbled" and re-OCR'd it for no benefit.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

from marker.config.parser import ConfigParser
from marker.models import create_model_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
ARXIV_DIR = PROJECT_ROOT / "arxiv_papers"
OUTPUT_DIR = PROJECT_ROOT / "parsed_papers"

HEADING_TAG_RE = re.compile(r"^<h([1-6])")
CONTENT_BLOCK_TYPES = {"Text", "ListGroup", "Footnote", "Equation", "Caption"}


def html_to_text(html_str: str) -> str:
    return BeautifulSoup(html_str or "", "html.parser").get_text(" ", strip=True)


def html_table_to_rows(html_str: str) -> list[list[str]]:
    soup = BeautifulSoup(html_str or "", "html.parser")
    return [
        [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
        for tr in soup.find_all("tr")
    ]


def page_num_from_block_id(block_id: str) -> int:
    # ids look like "/page/6/Table/4" -> page index 6, 1-indexed page 7
    return int(block_id.strip("/").split("/")[1]) + 1


def heading_level(html_str: str) -> int:
    m = HEADING_TAG_RE.match((html_str or "").strip())
    return int(m.group(1)) if m else 1


def walk(node):
    yield node
    for child in node.get("children") or []:
        yield from walk(child)


def find_tables(node: dict) -> list[dict]:
    tables = []
    children = node.get("children") or []
    for i, child in enumerate(children):
        if child.get("block_type") == "Table":
            caption = None
            if i > 0 and children[i - 1].get("block_type") == "Caption":
                caption = html_to_text(children[i - 1]["html"])
            elif i + 1 < len(children) and children[i + 1].get("block_type") == "Caption":
                caption = html_to_text(children[i + 1]["html"])
            tables.append(
                {
                    "table_id": child["id"],
                    "page": page_num_from_block_id(child["id"]),
                    "caption": caption or "",
                    "html": child.get("html", ""),
                    "rows": html_table_to_rows(child.get("html", "")),
                }
            )
        tables.extend(find_tables(child))
    return tables


def build_sections_and_pages(doc: dict) -> tuple[list[dict], list[dict]]:
    sections = []
    pages = []
    current = None

    def flush():
        nonlocal current
        if current is not None:
            current["text"] = " ".join(current["text_parts"]).strip()
            del current["text_parts"]
            sections.append(current)

    for page_node in doc.get("children") or []:
        if page_node.get("block_type") != "Page":
            continue
        page_num = page_num_from_block_id(page_node["id"])
        page_text_parts = []

        for block in walk(page_node):
            bt = block.get("block_type")
            if bt in ("Page", "Table", "TableGroup"):
                continue
            if bt == "SectionHeader":
                title = html_to_text(block.get("html", ""))
                if not title:
                    continue
                flush()
                current = {
                    "title": title,
                    "level": heading_level(block.get("html", "")),
                    "page_start": page_num,
                    "page_end": page_num,
                    "text_parts": [],
                }
                page_text_parts.append(title)
            elif bt in CONTENT_BLOCK_TYPES:
                text = html_to_text(block.get("html", ""))
                if not text:
                    continue
                page_text_parts.append(text)
                if current is None:
                    current = {
                        "title": "Front Matter",
                        "level": 0,
                        "page_start": page_num,
                        "page_end": page_num,
                        "text_parts": [],
                    }
                current["page_end"] = page_num
                current["text_parts"].append(text)

        pages.append({"page_number": page_num, "text": " ".join(page_text_parts)})

    flush()
    return sections, pages


def convert_with_marker(pdf_path: Path, converter) -> dict:
    rendered = converter(str(pdf_path))
    doc = rendered.model_dump()

    sections, pages = build_sections_and_pages(doc)
    tables = find_tables(doc)

    return {
        "num_pages": len(pages),
        "parser": "marker",
        "pages": pages,
        "sections": sections,
        "tables": tables,
        "diagnostics": {
            "num_headings_detected": len([s for s in sections if s["level"] > 0]),
            "captured_table_count": len(tables),
            "needs_fallback_parser": False,
        },
    }


def build_converter():
    config_parser = ConfigParser(
        {
            "output_format": "json",
            "disable_ocr": True,
            "disable_image_extraction": True,
        }
    )
    models = create_model_dict()
    converter_cls = config_parser.get_converter_cls()
    return converter_cls(
        config=config_parser.generate_config_dict(),
        artifact_dict=models,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )


def load_metadata_by_id(arxiv_dir: Path) -> dict:
    by_id = {}
    with (arxiv_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                by_id[rec["paper_id"]] = rec
    return by_id


def safe_filename(paper_id: str) -> str:
    return paper_id.removeprefix("arxiv:").replace("/", "_")


def find_flagged_papers(output_dir: Path) -> list[str]:
    flagged = []
    for fp in sorted(output_dir.glob("*.json")):
        with fp.open("r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("diagnostics", {}).get("needs_fallback_parser"):
            flagged.append(d["paper_id"])
    return flagged


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-parse flagged papers with marker")
    parser.add_argument("--arxiv-dir", type=Path, default=ARXIV_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    flagged_ids = find_flagged_papers(args.output_dir)
    if args.limit:
        flagged_ids = flagged_ids[: args.limit]
    logger.info("Found %d papers flagged for fallback parsing", len(flagged_ids))

    metadata_by_id = load_metadata_by_id(args.arxiv_dir)
    logger.info("Loading marker models (one-time)...")
    converter = build_converter()
    logger.info("Models loaded, starting conversion")

    succeeded, failed = 0, []
    for i, paper_id in enumerate(flagged_ids, start=1):
        rec = metadata_by_id[paper_id]
        pdf_path = Path(rec["pdf_path"])
        out_path = args.output_dir / f"{safe_filename(paper_id)}.json"
        try:
            parsed = convert_with_marker(pdf_path, converter)
        except Exception as e:
            logger.error("(%d/%d) FAILED %s: %s", i, len(flagged_ids), paper_id, e)
            failed.append(paper_id)
            continue

        output = {"paper_id": paper_id, "title": rec["title"], "source_pdf": str(pdf_path)}
        output.update(parsed)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False)

        succeeded += 1
        logger.info(
            "(%d/%d) %s -> %d pages, %d sections, %d tables",
            i,
            len(flagged_ids),
            paper_id,
            parsed["num_pages"],
            len(parsed["sections"]),
            len(parsed["tables"]),
        )

    logger.info("Done. succeeded=%d failed=%d", succeeded, len(failed))
    if failed:
        logger.info("Failed: %s", failed)


if __name__ == "__main__":
    main()
