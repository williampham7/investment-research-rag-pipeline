"""
Exact, deterministic filtering over extracted/*.json -- the "structured" side
of qa.py's dual-path Q&A. No LLM involved in the matching itself: the
classifier (qa_schema.QueryClassification) only turns the user's question into
filter parameters, and this module does plain Python comparisons over
already-validated data. That's the whole point of building structured
extraction -- exact answers to filter/aggregate questions instead of an LLM
eyeballing whatever chunks happen to surface from retrieval.
"""

import json
from pathlib import Path

from qa_schema import QueryClassification

PROJECT_ROOT = Path(__file__).parent
EXTRACTED_DIR = PROJECT_ROOT / "extracted"

_OPS = {
    ">": lambda v, t: v > t,
    ">=": lambda v, t: v >= t,
    "<": lambda v, t: v < t,
    "<=": lambda v, t: v <= t,
}


def load_extracted_records(extracted_dir: Path = EXTRACTED_DIR) -> list[dict]:
    records = []
    for fp in sorted(extracted_dir.glob("*.json")):
        if fp.name.startswith("_"):
            continue
        with fp.open("r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def _best_matching_metric(record: dict, classification: QueryClassification) -> dict | None:
    """Returns the metric entry that best satisfies metric_keyword/operator/threshold,
    or None if this record has no metric satisfying the filter."""
    keyword = classification.metric_keyword.lower()
    op = _OPS.get(classification.metric_operator)
    candidates = [
        m for m in record["extraction"]["performance_metrics"]
        if keyword in m["metric_name"].lower()
    ]
    if not candidates:
        return None
    if op is None:
        return candidates[0]
    passing = [m for m in candidates if op(m["value"], classification.metric_threshold)]
    if not passing:
        return None
    # "best" = furthest past the threshold in the direction the operator implies
    reverse = classification.metric_operator in (">", ">=")
    return sorted(passing, key=lambda m: m["value"], reverse=reverse)[0]


def run_structured_query(
    classification: QueryClassification,
    records: list[dict] | None = None,
    paper_id_allowlist: set[str] | None = None,
    limit: int = 25,
) -> list[dict]:
    """Filters extracted records by signal_type/asset_class/metric, returning
    the matches sorted by the matching metric's value (best first) when a
    metric filter is present, else in corpus order. Each result carries the
    specific matching metric (with its evidence_quote/source_location) so the
    caller can cite it, not just the paper as a whole."""
    if records is None:
        records = load_extracted_records()

    sig_filter = set(classification.signal_type_filter)
    asset_filter = set(classification.asset_class_filter)
    has_metric_filter = bool(classification.metric_keyword)

    results = []
    for record in records:
        if paper_id_allowlist is not None and record["paper_id"] not in paper_id_allowlist:
            continue
        ext = record["extraction"]
        if sig_filter and not (sig_filter & set(ext["signal_type"])):
            continue
        if asset_filter and not (asset_filter & set(ext["asset_class"])):
            continue

        matching_metric = None
        if has_metric_filter:
            matching_metric = _best_matching_metric(record, classification)
            if matching_metric is None:
                continue

        results.append({
            "paper_id": record["paper_id"],
            "title": record["title"],
            "signal_type": ext["signal_type"],
            "asset_class": ext["asset_class"],
            "methodology_summary": ext["methodology_summary"],
            "matching_metric": matching_metric,
        })

    if has_metric_filter:
        reverse = classification.metric_operator in (">", ">=", "none")
        results.sort(key=lambda r: r["matching_metric"]["value"], reverse=reverse)

    return results[:limit]
