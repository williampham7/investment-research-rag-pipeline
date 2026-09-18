"""
PDF -> structured JSON parsing for papers in arxiv_papers/.

Extracts per-page text, detects section headers from font size / bold
spans, and pulls out tables (PyMuPDF's find_tables) with their captions.
Writes one JSON file per paper into parsed_papers/.

If a paper's diagnostics show more "Table N" mentions in the text than
tables we actually captured, it's flagged needs_fallback_parser=True --
that's the signal to re-parse it with marker or GROBID instead.
"""

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
ARXIV_DIR = PROJECT_ROOT / "arxiv_papers"
OUTPUT_DIR = PROJECT_ROOT / "parsed_papers"

BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit for bold

NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
BARE_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*\.?$")
ARXIV_STAMP_RE = re.compile(r"^arXiv:\d{4}\.\d+")
TABLE_CAPTION_RE = re.compile(r"^\s*(TABLE|Table)\s+([IVXLCDM]+|\d+)\b")
FIGURE_CAPTION_RE = re.compile(r"^\s*(FIGURE|Figure|Fig\.?)\s+([IVXLCDM]+|\d+)\b")
TABLE_MENTION_RE = re.compile(r"\bTable\s+([IVXLCDM]+|\d+)\b")

SECTION_KEYWORDS = {
    "abstract", "introduction", "related work", "literature review", "background",
    "data", "data description", "methodology", "method", "methods", "model",
    "empirical results", "results", "discussion", "conclusion", "conclusions",
    "references", "appendix", "acknowledgements", "acknowledgments",
    "robustness checks", "robustness", "limitations", "summary",
}

CAPTION_SEARCH_WINDOW = 120  # points; how far to look for a caption near a table


def extract_lines(page) -> list[dict]:
    """One entry per visual line: merged span text, max font size, bold flag, bbox."""
    lines = []
    d = page.get_text("dict")
    for block in d["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            lines.append(
                {
                    "text": text,
                    "size": max(s["size"] for s in spans),
                    "bold": any(s["flags"] & BOLD_FLAG for s in spans),
                    "bbox": line["bbox"],
                }
            )
    return [ln for ln in lines if not ARXIV_STAMP_RE.match(ln["text"])]


def merge_split_numbering(lines: list[dict]) -> list[dict]:
    """Some templates put a bare section number on its own line, with the
    title on the next line (e.g. "1" then "Introduction"). Merge those pairs
    so heading detection sees them as a single "1 Introduction" line."""
    merged = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if (
            i + 1 < len(lines)
            and BARE_NUMBER_RE.match(ln["text"])
            and (ln["bold"] or ln["size"] > lines[i + 1]["size"])
            and len(lines[i + 1]["text"].split()) <= 10
        ):
            nxt = lines[i + 1]
            merged.append(
                {
                    "text": f"{ln['text']} {nxt['text']}",
                    "size": max(ln["size"], nxt["size"]),
                    "bold": ln["bold"] or nxt["bold"],
                    "bbox": nxt["bbox"],
                }
            )
            i += 2
        else:
            merged.append(ln)
            i += 1
    return merged


def merge_caption_blocks(lines: list[dict]) -> list[dict]:
    """Figure/table captions often wrap across several visual lines, each
    bold and short -- which otherwise look exactly like a run of section
    headings. Collapse each caption into a single entry (keeping the full
    text for table captions, since find_nearest_caption needs it; figure
    captions are dropped entirely since nothing in our schema uses them)."""
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        is_table_cap = bool(TABLE_CAPTION_RE.match(ln["text"]))
        is_figure_cap = bool(FIGURE_CAPTION_RE.match(ln["text"]))
        if is_table_cap or is_figure_cap:
            j = i + 1
            full_text = ln["text"]
            while (
                j < len(lines)
                and lines[j]["bold"] == ln["bold"]
                and abs(lines[j]["size"] - ln["size"]) < 0.5
                and not NUMBERED_HEADING_RE.match(lines[j]["text"])
                and len(lines[j]["text"].split()) <= 30
            ):
                full_text += " " + lines[j]["text"]
                j += 1
            if is_table_cap:
                out.append({**ln, "text": full_text})
            i = j
        else:
            out.append(ln)
            i += 1
    return out


def compute_body_font_size(pages_lines: list[list[dict]]) -> float:
    weighted = Counter()
    for lines in pages_lines:
        for ln in lines:
            weighted[round(ln["size"] * 2) / 2] += len(ln["text"])
    if not weighted:
        return 10.0
    return weighted.most_common(1)[0][0]


def looks_like_prose(text: str) -> bool:
    """Rejects math-typeset lines (italic unicode math alphanumerics, glyph
    placeholders) that would otherwise pass as short bold "headings"."""
    if any(ord(c) < 32 for c in text):
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return not text or non_ascii / len(text) <= 0.15


def classify_heading(line: dict, body_size: float) -> dict | None:
    text = line["text"]
    word_count = len(text.split())
    if word_count == 0 or word_count > 15 or len(text) > 140:
        return None
    if not looks_like_prose(text):
        return None
    if TABLE_CAPTION_RE.match(text) or FIGURE_CAPTION_RE.match(text):
        return None
    size_ratio = line["size"] / body_size if body_size else 1.0

    m = NUMBERED_HEADING_RE.match(text)
    if m and re.search(r"[A-Za-z]{2,}", m.group(2)) and (line["bold"] or size_ratio >= 1.1):
        level = m.group(1).count(".") + 1
        return {"level": min(level, 3), "title": text}

    stripped_keyword = re.sub(r"^\d+(\.\d+)*\.?\s+", "", text).strip().lower().rstrip(":")
    if stripped_keyword in SECTION_KEYWORDS and (line["bold"] or size_ratio >= 1.05):
        return {"level": 1, "title": text}

    if size_ratio >= 1.3:
        return {"level": 1, "title": text}
    if line["bold"] and 0.95 <= size_ratio < 1.3:
        return {"level": 2, "title": text}

    return None


def build_sections(pages_lines: list[list[dict]], body_size: float) -> list[dict]:
    sections = []
    current = None

    def flush():
        if current is not None and current["text_parts"]:
            current["text"] = " ".join(current["text_parts"]).strip()
            del current["text_parts"]
            sections.append(current)
        elif current is not None:
            del current["text_parts"]
            current["text"] = ""
            sections.append(current)

    for page_num, lines in enumerate(pages_lines, start=1):
        for ln in lines:
            heading = classify_heading(ln, body_size)
            if heading:
                flush()
                current = {
                    "title": heading["title"],
                    "level": heading["level"],
                    "page_start": page_num,
                    "page_end": page_num,
                    "text_parts": [],
                }
            else:
                if current is None:
                    current = {
                        "title": "Front Matter",
                        "level": 0,
                        "page_start": page_num,
                        "page_end": page_num,
                        "text_parts": [],
                    }
                current["page_end"] = page_num
                current["text_parts"].append(ln["text"])
    flush()
    return sections


def find_nearest_caption(table_bbox, page_lines: list[dict]) -> tuple[dict | None, str | None]:
    """Returns (caption_line, kind) where kind is 'table' or 'figure', whichever is nearest."""
    best = None
    best_kind = None
    best_dist = CAPTION_SEARCH_WINDOW
    tx0, ty0, tx1, ty1 = table_bbox
    for ln in page_lines:
        lx0, ly0, lx1, ly1 = ln["bbox"]
        if ly1 <= ty0:
            dist = ty0 - ly1
        elif ly0 >= ty1:
            dist = ly0 - ty1
        else:
            continue  # vertically overlapping the table itself, not a caption
        kind = None
        if TABLE_CAPTION_RE.match(ln["text"]):
            kind = "table"
        elif FIGURE_CAPTION_RE.match(ln["text"]):
            kind = "figure"
        if kind and dist < best_dist:
            best, best_kind, best_dist = ln, kind, dist
    return best, best_kind


def extract_tables(doc, pages_lines: list[list[dict]]) -> tuple[list[dict], list[dict]]:
    tables = []
    rejected = []
    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        try:
            found = page.find_tables()
        except Exception as e:
            logger.warning("find_tables failed on page %d: %s", page_num, e)
            continue
        for t_idx, table in enumerate(found.tables):
            caption_line, kind = find_nearest_caption(table.bbox, pages_lines[page_index])
            if kind == "table":
                tables.append(
                    {
                        "table_id": f"p{page_num}_t{t_idx + 1}",
                        "page": page_num,
                        "caption": caption_line["text"],
                        "bbox": list(table.bbox),
                        "rows": table.extract(),
                    }
                )
            else:
                rejected.append(
                    {
                        "page": page_num,
                        "bbox": list(table.bbox),
                        "reason": (
                            f"nearest caption is a Figure, not a Table: {caption_line['text']!r}"
                            if kind == "figure"
                            else "no Table/Figure caption found nearby"
                        ),
                    }
                )
    return tables, rejected


def parse_pdf(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    pages_lines = [
        merge_caption_blocks(merge_split_numbering(extract_lines(page))) for page in doc
    ]
    body_size = compute_body_font_size(pages_lines)

    pages = [
        {"page_number": i + 1, "text": doc[i].get_text("text")} for i in range(len(doc))
    ]
    sections = build_sections(pages_lines, body_size)
    tables, rejected = extract_tables(doc, pages_lines)

    full_text = "\n".join(p["text"] for p in pages)
    mentioned_numbers = {m.group(1) for m in TABLE_MENTION_RE.finditer(full_text)}
    captured_numbers = set()
    for t in tables:
        m = TABLE_CAPTION_RE.match(t["caption"])
        if m:
            captured_numbers.add(m.group(2))

    num_headings = len([s for s in sections if s["level"] > 0])
    heading_density = num_headings / len(pages) if pages else 0.0
    tables_garbled = len(mentioned_numbers) > len(captured_numbers)
    headings_garbled = heading_density > 3.5

    diagnostics = {
        "body_font_size": body_size,
        "num_headings_detected": num_headings,
        "heading_density_per_page": round(heading_density, 2),
        "mentioned_table_count": len(mentioned_numbers),
        "captured_table_count": len(tables),
        "rejected_table_candidates": rejected,
        "tables_garbled": tables_garbled,
        "headings_garbled": headings_garbled,
        "needs_fallback_parser": tables_garbled or headings_garbled,
    }

    doc.close()
    return {
        "num_pages": len(pages),
        "parser": "pymupdf",
        "pages": pages,
        "sections": sections,
        "tables": tables,
        "diagnostics": diagnostics,
    }


def load_metadata(arxiv_dir: Path) -> list[dict]:
    metadata_path = arxiv_dir / "metadata.jsonl"
    records = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_filename(paper_id: str) -> str:
    return paper_id.removeprefix("arxiv:").replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse arXiv PDFs into structured JSON")
    parser.add_argument("--arxiv-dir", type=Path, default=ARXIV_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Only parse the first N papers")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_metadata(args.arxiv_dir)
    if args.limit:
        records = records[: args.limit]

    needs_fallback = []
    for i, rec in enumerate(records, start=1):
        pdf_path = Path(rec["pdf_path"])
        out_path = args.output_dir / f"{safe_filename(rec['paper_id'])}.json"
        try:
            parsed = parse_pdf(pdf_path)
        except Exception as e:
            logger.error("Failed to parse %s: %s", rec["paper_id"], e)
            continue

        output = {"paper_id": rec["paper_id"], "title": rec["title"], "source_pdf": str(pdf_path)}
        output.update(parsed)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False)

        if parsed["diagnostics"]["needs_fallback_parser"]:
            needs_fallback.append(rec["paper_id"])

        logger.info(
            "(%d/%d) %s -> %d pages, %d sections, %d tables%s",
            i,
            len(records),
            rec["paper_id"],
            parsed["num_pages"],
            len(parsed["sections"]),
            len(parsed["tables"]),
            " [NEEDS FALLBACK PARSER]" if parsed["diagnostics"]["needs_fallback_parser"] else "",
        )

    logger.info("Done. %d/%d papers flagged for fallback parsing (marker/GROBID).", len(needs_fallback), len(records))
    if needs_fallback:
        logger.info("Flagged: %s", needs_fallback)


if __name__ == "__main__":
    main()
