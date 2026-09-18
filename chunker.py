"""
Section-aware chunking over parsed_papers/*.json -> chunks/chunks.jsonl.

- Splits on section headers first, then packs sentences into ~500-800 token
  chunks with ~12% overlap within each section.
- Tables become their own standalone chunk (caption + flattened rows),
  never split, tagged with the section they fall under.
- The Abstract section is tagged chunk_type="abstract" instead of "text" so
  retrieval can boost/filter on it.
- Reference lists and acknowledgements are skipped: high volume, ~no
  retrieval value for "find a tradable signal" style queries.
- Text/abstract fragments under MIN_CHUNK_TOKENS are dropped (stray page
  numbers and layout artifacts, not real content).
- A table chunk over MAX_TABLE_TOKENS is split into multiple row-batches
  (caption repeated on each) so nothing exceeds common embedding API input
  limits (e.g. 8191 tokens) -- a handful of very wide results tables run to
  5000-9800 tokens flattened as a single chunk otherwise.
"""

import argparse
import json
import logging
import re
from pathlib import Path

import tiktoken

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
PARSED_DIR = PROJECT_ROOT / "parsed_papers"
OUTPUT_DIR = PROJECT_ROOT / "chunks"

TARGET_MIN_TOKENS = 500
TARGET_MAX_TOKENS = 800
OVERLAP_TOKENS = 90  # ~12-15% of the 500-800 target range
MIN_CHUNK_TOKENS = 10  # drop text/abstract fragments smaller than this
MAX_TABLE_TOKENS = 4000  # split a table into row-batches above this

EXCLUDED_SECTIONS = {"references", "bibliography", "acknowledgements", "acknowledgments"}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
LEADING_NUMBERING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+")

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text, disallowed_special=()))


def normalize_title(title: str) -> str:
    return LEADING_NUMBERING_RE.sub("", title).strip().lower().rstrip(":")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def hard_split_long_sentence(sentence: str, max_tokens: int) -> list[str]:
    tokens = _encoding.encode(sentence, disallowed_special=())
    return [
        _encoding.decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)
    ]


def pack_sentences(sentences: list[str]) -> list[str]:
    """Greedy-pack sentences into chunks within [TARGET_MIN, TARGET_MAX] tokens,
    carrying the trailing ~OVERLAP_TOKENS worth of sentences into the next chunk."""
    units = []  # (sentence_text, token_count)
    for s in sentences:
        n = count_tokens(s)
        if n > TARGET_MAX_TOKENS:
            units.extend((piece, count_tokens(piece)) for piece in hard_split_long_sentence(s, TARGET_MAX_TOKENS))
        else:
            units.append((s, n))

    chunks = []
    current: list[tuple[str, int]] = []
    current_tokens = 0
    has_new_content = False  # False right after flush() until a real sentence is added

    def flush() -> None:
        """Appends the current chunk and seeds the next one with trailing overlap."""
        nonlocal current, current_tokens, has_new_content
        chunks.append(" ".join(s for s, _ in current))
        overlap_seed = []
        overlap_sum = 0
        for s, n in reversed(current):
            if overlap_sum >= OVERLAP_TOKENS:
                break
            overlap_seed.insert(0, (s, n))
            overlap_sum += n
        current = list(overlap_seed)
        current_tokens = overlap_sum
        has_new_content = False

    for sentence, n in units:
        if current_tokens + n > TARGET_MAX_TOKENS and current_tokens > 0:
            flush()
        current.append((sentence, n))
        current_tokens += n
        has_new_content = True
        if current_tokens >= TARGET_MAX_TOKENS:
            flush()

    if current and has_new_content:
        chunks.append(" ".join(s for s, _ in current))

    return chunks


def row_to_line(row: list[str]) -> str:
    cells = [c.strip() for c in row if c and c.strip()]
    return " | ".join(cells)


def table_to_text(table: dict) -> str:
    caption = table.get("caption", "").strip()
    lines = [row_to_line(row) for row in table.get("rows", [])]
    body = "\n".join(line for line in lines if line)
    return f"{caption}\n\n{body}".strip() if caption else body


def table_to_texts(table: dict) -> list[str]:
    """One text blob per table, unless it's too big for embedding limits --
    then split into row-batches, each carrying the caption for context."""
    full_text = table_to_text(table)
    if count_tokens(full_text) <= MAX_TABLE_TOKENS:
        return [full_text] if full_text else []

    caption = table.get("caption", "").strip()
    caption_tokens = count_tokens(caption)
    budget = MAX_TABLE_TOKENS - caption_tokens

    batches = []
    current_lines: list[str] = []
    current_tokens = 0
    for row in table.get("rows", []):
        line = row_to_line(row)
        if not line:
            continue
        n = count_tokens(line)
        if current_tokens + n > budget and current_lines:
            batches.append(current_lines)
            current_lines, current_tokens = [], 0
        current_lines.append(line)
        current_tokens += n
    if current_lines:
        batches.append(current_lines)

    return [f"{caption}\n\n{chr(10).join(lines)}".strip() for lines in batches]


def find_enclosing_section_title(page: int, sections: list[dict]) -> str:
    for s in sections:
        if s["page_start"] <= page <= s["page_end"]:
            return s["title"]
    return "Unknown"


def chunk_paper(doc: dict) -> list[dict]:
    paper_id = doc["paper_id"]
    title = doc["title"]
    sections = doc.get("sections", [])
    chunks = []

    for sec_idx, section in enumerate(sections):
        norm = normalize_title(section["title"])
        if norm in EXCLUDED_SECTIONS:
            continue
        text = section.get("text", "").strip()
        if not text:
            continue

        chunk_type = "abstract" if norm == "abstract" else "text"
        pieces = pack_sentences(split_sentences(text))

        chunk_idx = 0
        for piece in pieces:
            n = count_tokens(piece)
            if n < MIN_CHUNK_TOKENS:
                continue
            chunks.append(
                {
                    "chunk_id": f"{paper_id}::sec{sec_idx}::chunk{chunk_idx}",
                    "paper_id": paper_id,
                    "title": title,
                    "section_title": section["title"],
                    "chunk_type": chunk_type,
                    "page_start": section["page_start"],
                    "page_end": section["page_end"],
                    "token_count": n,
                    "text": piece,
                }
            )
            chunk_idx += 1

    for table in doc.get("tables", []):
        texts = table_to_texts(table)
        safe_table_id = table["table_id"].replace("/", "_")
        multi_part = len(texts) > 1
        for part_idx, text in enumerate(texts):
            chunk_id = f"{paper_id}::table::{safe_table_id}"
            if multi_part:
                chunk_id += f"::part{part_idx}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "title": title,
                    "section_title": find_enclosing_section_title(table["page"], sections),
                    "chunk_type": "table",
                    "page_start": table["page"],
                    "page_end": table["page"],
                    "token_count": count_tokens(text),
                    "text": text,
                }
            )

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk parsed papers for the RAG index")
    parser.add_argument("--parsed-dir", type=Path, default=PARSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "chunks.jsonl"

    paper_files = sorted(args.parsed_dir.glob("*.json"))
    logger.info("Chunking %d parsed papers", len(paper_files))

    total_chunks = 0
    type_counts: dict[str, int] = {}
    token_counts = []

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, fp in enumerate(paper_files, start=1):
            with fp.open("r", encoding="utf-8") as f:
                doc = json.load(f)
            chunks = chunk_paper(doc)
            for c in chunks:
                out_f.write(json.dumps(c, ensure_ascii=False) + "\n")
                type_counts[c["chunk_type"]] = type_counts.get(c["chunk_type"], 0) + 1
                token_counts.append(c["token_count"])
            total_chunks += len(chunks)
            logger.info("(%d/%d) %s -> %d chunks", i, len(paper_files), doc["paper_id"], len(chunks))

    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    logger.info("Done. %d total chunks across %d papers", total_chunks, len(paper_files))
    logger.info("By type: %s", type_counts)
    logger.info("Avg tokens/chunk: %.1f, min=%d, max=%d", avg_tokens, min(token_counts, default=0), max(token_counts, default=0))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
