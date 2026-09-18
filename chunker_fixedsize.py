"""
Naive fixed-size chunking baseline over parsed_papers/*.json -> chunks/chunks_fixedsize.jsonl.

Unlike chunker.py, this ignores section/table structure entirely: each
paper's page text is concatenated into one token stream and sliced into
fixed windows with overlap, so a chunk can cut mid-sentence, mid-table, or
across a section boundary. Same size/overlap constants as chunker.py so the
only variable that differs between the two chunk sets is chunking strategy.
"""

import argparse
import json
import logging
from pathlib import Path

import tiktoken

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
PARSED_DIR = PROJECT_ROOT / "parsed_papers"
OUTPUT_DIR = PROJECT_ROOT / "chunks"

WINDOW_TOKENS = 650  # midpoint of chunker.py's 500-800 target range
OVERLAP_TOKENS = 90  # same overlap budget as chunker.py

_encoding = tiktoken.get_encoding("cl100k_base")


def flatten_pages_to_tokens(pages: list[dict]) -> list[tuple[int, int]]:
    """Returns a flat list of (token_id, source_page_number)."""
    flat = []
    for page in pages:
        for tok in _encoding.encode(page["text"], disallowed_special=()):
            flat.append((tok, page["page_number"]))
    return flat


def chunk_paper(doc: dict) -> list[dict]:
    paper_id = doc["paper_id"]
    title = doc["title"]
    pages = doc.get("pages", [])
    flat = flatten_pages_to_tokens(pages)
    if not flat:
        return []

    chunks = []
    step = WINDOW_TOKENS - OVERLAP_TOKENS
    chunk_idx = 0
    for start in range(0, len(flat), step):
        window = flat[start : start + WINDOW_TOKENS]
        if not window:
            break
        token_ids = [t for t, _ in window]
        page_nums = [p for _, p in window]
        text = _encoding.decode(token_ids).strip()
        if not text:
            continue
        chunks.append(
            {
                "chunk_id": f"{paper_id}::fixed::chunk{chunk_idx}",
                "paper_id": paper_id,
                "title": title,
                "section_title": "N/A (fixed-size chunk)",
                "chunk_type": "text",
                "page_start": min(page_nums),
                "page_end": max(page_nums),
                "token_count": len(token_ids),
                "text": text,
            }
        )
        chunk_idx += 1
        if start + WINDOW_TOKENS >= len(flat):
            break

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-size chunking baseline")
    parser.add_argument("--parsed-dir", type=Path, default=PARSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "chunks_fixedsize.jsonl"

    paper_files = sorted(args.parsed_dir.glob("*.json"))
    logger.info("Chunking %d parsed papers (fixed-size)", len(paper_files))

    total = 0
    token_counts = []
    with out_path.open("w", encoding="utf-8") as out_f:
        for i, fp in enumerate(paper_files, start=1):
            with fp.open("r", encoding="utf-8") as f:
                doc = json.load(f)
            chunks = chunk_paper(doc)
            for c in chunks:
                out_f.write(json.dumps(c, ensure_ascii=False) + "\n")
                token_counts.append(c["token_count"])
            total += len(chunks)
            logger.info("(%d/%d) %s -> %d chunks", i, len(paper_files), doc["paper_id"], len(chunks))

    avg = sum(token_counts) / len(token_counts) if token_counts else 0
    logger.info("Done. %d total chunks across %d papers", total, len(paper_files))
    logger.info("Avg tokens/chunk: %.1f, min=%d, max=%d", avg, min(token_counts, default=0), max(token_counts, default=0))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
