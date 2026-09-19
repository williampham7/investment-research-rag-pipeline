"""
Structured extraction: parsed_papers/*.json -> extracted/<safe_id>.json

For each paper, sends the full non-reference paper text (all sections except
EXCLUDED_SECTIONS, plus tables) to Gemini with extraction_schema.StructuredExtraction
enforced as a response_schema, so the API itself guarantees field/enum shape.
A second, independent pass then:

  1. checks every performance_metrics[].evidence_quote actually appears (as a
     bag-of-words match) in the text we sent -- catching hallucinated numbers
     that satisfy the schema but aren't grounded in the paper;
  2. sanity-range-checks numeric values per unit;
  3. cross-checks signal_type/asset_class against the coarse tags already
     assigned during curation (arxiv_papers/curation_notes.jsonl) and flags
     disagreement instead of silently trusting either source.

None of this is silently "fixed" -- flags are attached to the saved record
under extraction_meta.validation_flags for a human to review.

Resumable: skips a paper if extracted/<safe_id>.json already exists (--force
to redo it). Progress is checkpointed one paper at a time to disk, so hitting
a daily free-tier quota partway through loses no completed work -- rerunning
later picks up where it left off. Papers that fail after retries are recorded
in extracted/_failed.jsonl (not silently dropped) and do not stop the run.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment. Get a free
key at https://aistudio.google.com/apikey.

Usage:
    python structured_extract.py                        # all papers
    python structured_extract.py --only arxiv:1234.5678v1  # single paper
    python structured_extract.py --limit 5               # smoke test
    python structured_extract.py --force                 # redo everything
    python structured_extract.py --model gemini-2.0-flash --sleep 6
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from google import genai

from chunker import EXCLUDED_SECTIONS, count_tokens, normalize_title, table_to_text
from extraction_schema import StructuredExtraction
from llm_utils import DEFAULT_MODEL, DailyQuotaExhausted, generate_structured, get_client
from pdf_parser import safe_filename

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
PARSED_DIR = PROJECT_ROOT / "parsed_papers"
OUTPUT_DIR = PROJECT_ROOT / "extracted"
CURATION_NOTES_PATH = PROJECT_ROOT / "arxiv_papers" / "curation_notes.jsonl"

MAX_INPUT_TOKENS = 250_000  # observed corpus max is ~60k; this is a safety net, not a real limit
DEFAULT_SLEEP_SECONDS = 4.5  # ~13 requests/min, conservative for the free tier

QUOTE_WORD_RE = re.compile(r"[a-z0-9]+")

SYSTEM_INSTRUCTION = """You are a meticulous quantitative-finance research analyst extracting \
structured facts from an academic paper for a research database. Follow these rules strictly:

1. Base every field ONLY on the paper text provided. Never use outside knowledge about the paper, \
its authors, or the topic in general.
2. Never invent or compute numbers. Only report a performance metric if it is explicitly stated in \
the text. If you are not sure a number is a performance metric (as opposed to, e.g., a p-value or a \
sample count), leave it out. Statistical diagnostic/goodness-of-fit tests (Jarque-Bera, ADF, KPSS, \
ARCH-LM, Durbin-Watson, regression R-squared, etc.) describe the data or model fit, not the \
strategy's performance -- do not include them in performance_metrics even if quantitative.
2b. Numeric findings are often reported as PROSE DESCRIBING A FIGURE rather than in a clean table, \
e.g. "CAAR is recorded at 0.041 by day -6" or "AAR was positive for 20 of 30 days (66%)". These count \
as performance_metrics just as much as a table cell does -- read the full results/discussion text \
carefully for numbers like this, not only formal tables. If a paper truly reports zero quantitative \
results anywhere (purely theoretical/methodological), performance_metrics should be an empty list.
3. Every performance_metrics[].evidence_quote MUST be copied verbatim (or near-verbatim, e.g. across a \
line break) from the supplied text -- not paraphrased, not reconstructed from memory.
4. If a string field genuinely has no answer in the text, write "not reported" rather than guessing.
5. signal_type and asset_class must be chosen from the fixed enums given in the schema. Use "other" \
only when nothing else fits.
6. Set extraction_confidence to "low" if the paper is ambiguous, mostly theoretical, or you had to \
stretch to fill required fields; use extraction_notes to say why."""


def build_paper_text(doc: dict, max_tokens: int = MAX_INPUT_TOKENS) -> str:
    parts = [f"Title: {doc['title']}"]
    for section in doc.get("sections", []):
        if normalize_title(section["title"]) in EXCLUDED_SECTIONS:
            continue
        text = section.get("text", "").strip()
        if not text:
            continue
        parts.append(f"## {section['title']}\n{text}")
    for table in doc.get("tables", []):
        text = table_to_text(table).strip()
        if text:
            parts.append(f"## Table {table.get('table_id', '')} (page {table.get('page', '?')})\n{text}")

    full = "\n\n".join(parts)
    if count_tokens(full) <= max_tokens:
        return full

    logger.warning("Paper %s exceeds %d tokens; truncating (keeping the start)", doc.get("paper_id"), max_tokens)
    encoding = __import__("tiktoken").get_encoding("cl100k_base")
    tokens = encoding.encode(full, disallowed_special=())
    return encoding.decode(tokens[:max_tokens])


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_curation_tags() -> dict[str, dict]:
    return {rec["paper_id"]: rec for rec in load_jsonl(CURATION_NOTES_PATH)}


def _normalize_words(text: str) -> list[str]:
    return QUOTE_WORD_RE.findall(text.lower())


def quote_is_grounded(quote: str, source_text: str, threshold: float = 0.7) -> bool:
    """Bag-of-words overlap check: catches fabricated/hallucinated quotes while
    tolerating whitespace/line-break/OCR-artifact differences from the source."""
    quote_words = [w for w in _normalize_words(quote) if len(w) >= 3]
    if not quote_words:
        return True
    source_words = set(_normalize_words(source_text))
    matched = sum(1 for w in quote_words if w in source_words)
    return (matched / len(quote_words)) >= threshold


METRIC_RANGES: dict[str, tuple[float, float]] = {
    "ratio": (-15.0, 20.0),
    "percent": (-500.0, 5000.0),
    "bps": (-50000.0, 500000.0),
    "stddev": (0.0, 500.0),
    "count": (0.0, 1e12),
    "currency": (-1e15, 1e15),
    "other": (-1e12, 1e12),
}

# t-statistics/z-statistics aren't economically bounded like a Sharpe ratio --
# with a large enough sample (millions of panel observations is common in
# modern empirical finance) a legitimate t-stat can run into the hundreds.
# Checked by name instead of unit since the model reports them as "ratio".
WIDE_RANGE_METRIC_NAME_RE = re.compile(r"t[\s-]?stat|z[\s-]?stat|f[\s-]?stat|chi[\s-]?square", re.I)
WIDE_RATIO_RANGE = (-1000.0, 1000.0)


def _tag_mismatch(extracted: list[str], curated: list[str]) -> bool:
    """True only when the two tag sets share nothing AND neither is a bare
    ['other'] -- a real disagreement, not one side reasonably hedging."""
    if not curated or set(extracted) & set(curated):
        return False
    return extracted != ["other"] and curated != ["other"]


def validate_extraction(
    extraction: StructuredExtraction, source_text: str, curation_tags: dict | None
) -> list[str]:
    flags: list[str] = []

    if not extraction.performance_metrics:
        flags.append("no_performance_metrics_found")

    for metric in extraction.performance_metrics:
        if not quote_is_grounded(metric.evidence_quote, source_text):
            flags.append(f"unverified_quote:{metric.metric_name!r}")
        if metric.unit == "ratio" and WIDE_RANGE_METRIC_NAME_RE.search(metric.metric_name):
            lo, hi = WIDE_RATIO_RANGE
        else:
            lo, hi = METRIC_RANGES.get(metric.unit, (-1e12, 1e12))
        if not (lo <= metric.value <= hi):
            flags.append(f"metric_out_of_range:{metric.metric_name!r}={metric.value}{metric.unit}")

    if curation_tags:
        if _tag_mismatch(extraction.signal_type, curation_tags.get("signal_type", [])):
            flags.append("signal_type_disagrees_with_curation")
        if _tag_mismatch(extraction.asset_class, curation_tags.get("asset_class", [])):
            flags.append("asset_class_disagrees_with_curation")
    else:
        flags.append("no_curation_tags_to_cross_check")

    return flags


def process_paper(
    client: genai.Client,
    model: str,
    doc: dict,
    curation_tags: dict[str, dict],
) -> dict:
    paper_id = doc["paper_id"]
    paper_text = build_paper_text(doc)
    input_tokens = count_tokens(paper_text)

    extraction = generate_structured(
        client, model,
        contents=f"Extract structured facts from this paper.\n\n{paper_text}",
        response_schema=StructuredExtraction,
        system_instruction=SYSTEM_INSTRUCTION,
    )
    tags = curation_tags.get(paper_id)
    flags = validate_extraction(extraction, paper_text, tags)

    return {
        "paper_id": paper_id,
        "title": doc["title"],
        "extraction": extraction.model_dump(),
        "extraction_meta": {
            "model": model,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "input_tokens": input_tokens,
            "validation_flags": flags,
            "curation_signal_type": tags.get("signal_type", []) if tags else [],
            "curation_asset_class": tags.get("asset_class", []) if tags else [],
        },
    }


def write_summary_csv(output_dir: Path, records: list[dict]) -> Path:
    csv_path = output_dir / "extraction_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "paper_id", "title", "signal_type", "asset_class", "num_performance_metrics",
            "sample_metric", "extraction_confidence", "num_validation_flags", "validation_flags",
        ])
        for rec in records:
            ext = rec["extraction"]
            metrics = ext.get("performance_metrics", [])
            sample_metric = ""
            if metrics:
                m = next((m for m in metrics if "sharpe" in m["metric_name"].lower()), metrics[0])
                sample_metric = f"{m['metric_name']}={m['value']}{m['unit']}"
            flags = rec["extraction_meta"]["validation_flags"]
            writer.writerow([
                rec["paper_id"],
                rec["title"],
                ";".join(ext.get("signal_type", [])),
                ";".join(ext.get("asset_class", [])),
                len(metrics),
                sample_metric,
                ext.get("extraction_confidence", ""),
                len(flags),
                ";".join(flags),
            ])
    return csv_path


def revalidate_only(output_dir: Path) -> None:
    """Re-derive validation_flags for every existing extracted/*.json against the
    current validate_extraction() logic, with zero API calls. Rebuilds each
    paper's source text from parsed_papers/ so the grounding check still runs."""
    curation_tags = load_curation_tags()
    paper_files = sorted(output_dir.glob("*.json"))
    changed = 0
    all_records = []
    for fp in paper_files:
        if fp.name.startswith("_"):
            continue
        with fp.open("r", encoding="utf-8") as f:
            record = json.load(f)
        parsed_path = PARSED_DIR / fp.name
        if not parsed_path.exists():
            logger.warning("No parsed_papers/%s found, skipping revalidation for it", fp.name)
            all_records.append(record)
            continue
        with parsed_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        paper_text = build_paper_text(doc)
        extraction = StructuredExtraction.model_validate(record["extraction"])
        new_flags = validate_extraction(extraction, paper_text, curation_tags.get(record["paper_id"]))
        if new_flags != record["extraction_meta"]["validation_flags"]:
            logger.info("%s: flags changed %s -> %s", record["paper_id"], record["extraction_meta"]["validation_flags"], new_flags)
            changed += 1
        record["extraction_meta"]["validation_flags"] = new_flags
        with fp.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        all_records.append(record)

    csv_path = write_summary_csv(output_dir, all_records)
    logger.info("Revalidated %d records, %d had flag changes. Wrote %s", len(all_records), changed, csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured extraction over parsed papers via Gemini")
    parser.add_argument("--parsed-dir", type=Path, default=PARSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--only", default=None, help="Only process this one paper_id (e.g. arxiv:1234.5678v1)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N papers (smoke test)")
    parser.add_argument("--force", action="store_true", help="Re-extract papers that already have output")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="Seconds to sleep between calls")
    parser.add_argument(
        "--revalidate-only", action="store_true",
        help="Re-run validate_extraction() over existing extracted/*.json with no API calls -- "
        "use after changing validation_extraction()/METRIC_RANGES to refresh flags for free.",
    )
    args = parser.parse_args()

    if args.revalidate_only:
        revalidate_only(args.output_dir)
        return

    try:
        client = get_client()
    except RuntimeError as e:
        logger.error(str(e))
        raise SystemExit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curation_tags = load_curation_tags()

    paper_files = sorted(args.parsed_dir.glob("*.json"))
    if args.only:
        target = safe_filename(args.only) + ".json"
        paper_files = [fp for fp in paper_files if fp.name == target]
        if not paper_files:
            logger.error("No parsed file found for %s", args.only)
            raise SystemExit(1)
    if args.limit:
        paper_files = paper_files[: args.limit]

    failed_path = args.output_dir / "_failed.jsonl"

    logger.info("Model: %s | %d parsed papers to consider", args.model, len(paper_files))

    processed, skipped, failed = 0, 0, 0
    for i, fp in enumerate(paper_files, start=1):
        with fp.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        paper_id = doc["paper_id"]
        out_path = args.output_dir / fp.name

        if out_path.exists() and not args.force:
            logger.info("(%d/%d) %s -> already extracted, skipping", i, len(paper_files), paper_id)
            skipped += 1
            continue

        logger.info("(%d/%d) %s -> extracting...", i, len(paper_files), paper_id)
        try:
            record = process_paper(client, args.model, doc, curation_tags)
        except DailyQuotaExhausted:
            logger.error(
                "Daily free-tier quota for model '%s' is exhausted. Stopping here instead of "
                "retrying every remaining paper against the same wall -- %d done, %d left. "
                "Quotas reset daily; rerun this exact command later (or pass --model to use a "
                "different model) and it will resume from where it left off.",
                args.model, processed + skipped, len(paper_files) - i + 1,
            )
            break
        except Exception as e:  # noqa: BLE001 - a single paper's failure must not kill the run
            logger.error("(%d/%d) %s -> FAILED: %s", i, len(paper_files), paper_id, e)
            with failed_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "paper_id": paper_id,
                    "error": str(e),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
            failed += 1
            time.sleep(args.sleep)
            continue

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        flags = record["extraction_meta"]["validation_flags"]
        flag_note = f" [flags: {', '.join(flags)}]" if flags else ""
        logger.info(
            "(%d/%d) %s -> OK, %d metrics, confidence=%s%s",
            i, len(paper_files), paper_id,
            len(record["extraction"]["performance_metrics"]),
            record["extraction"]["extraction_confidence"],
            flag_note,
        )
        processed += 1
        time.sleep(args.sleep)

    all_records = []
    for fp in sorted(args.output_dir.glob("*.json")):
        if fp.name.startswith("_"):
            continue
        with fp.open("r", encoding="utf-8") as f:
            all_records.append(json.load(f))
    csv_path = write_summary_csv(args.output_dir, all_records)

    logger.info(
        "Done. %d extracted this run, %d skipped (already done), %d failed. %d total extracted records.",
        processed, skipped, failed, len(all_records),
    )
    logger.info("Wrote %s", csv_path)
    if failed:
        logger.warning("See %s for failures -- rerun the script (same args) to retry them.", failed_path)


if __name__ == "__main__":
    main()
