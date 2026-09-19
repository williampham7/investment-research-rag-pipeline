"""
End-to-end Q&A: classifies a question, then answers it via one of two paths
(or both):

  structured -- exact filter/aggregate questions ("which papers report Sharpe
                > 2 in FX carry?") are answered by plain Python filtering over
                extracted/*.json (structured_query.py). No LLM in the loop for
                the actual matching -- deterministic, can't hallucinate a
                number, but can't explain mechanism either.
  retrieval  -- open-ended "how/why/explain" questions go through
                HybridRetriever.search() and an LLM synthesizes a cited
                answer from the retrieved chunks only.
  hybrid     -- both: the structured filter finds the matching papers, then
                retrieval (scoped to just those papers) explains them.

A single classifier call (qa_schema.QueryClassification) decides the mode and
extracts filter parameters in one shot.

Usage:
    python qa.py --query "Which papers report a Sharpe ratio above 2 in FX carry?"
    python qa.py --query "How do papers construct a momentum signal in FX?"
    python qa.py --query "What's the best performing stat arb strategy and how does it work?"

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment, and an index
already built by embed.py (see retrieve.py).
"""

import argparse
import logging
import sys
from typing import Literal

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pydantic import BaseModel, Field

from llm_utils import DEFAULT_MODEL, generate_structured, get_client
from qa_schema import QueryClassification
from retrieve import HybridRetriever
from structured_query import load_extracted_records, run_structured_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLASSIFIER_SYSTEM_INSTRUCTION = """You classify a user's question about a corpus of empirical \
quantitative-finance research papers into a query mode and filter parameters, per the schema.

Use the fixed signal_type/asset_class enums -- only set a filter dimension if the question actually \
implies it; don't force a guess. metric_keyword is a generic lowercase word (e.g. "sharpe", "return", \
"drawdown", "alpha"), not an exact metric name, since it is matched by substring against many different \
phrasings of the same metric across papers -- it MUST be a word that could plausibly appear inside an \
actual metric name. If the question says "best performing"/"highest returns"/"top strategy" WITHOUT \
naming a specific metric, use "sharpe" as metric_keyword (the standard cross-strategy performance \
yardstick reported by nearly every paper) rather than a vague word like "performance" or "best", which \
will never match any real metric name and silently return zero results."""

SYNTHESIS_SYSTEM_INSTRUCTION = """You are a research assistant answering questions about a corpus of \
empirical quantitative-finance papers using ONLY the provided source excerpts. Rules:

1. Base your answer only on the provided sources. Never use outside knowledge about specific papers, \
authors, or findings not shown here.
2. Cite every factual claim with the bracketed source number(s) it came from, e.g. "Momentum returns \
are persistent across asset classes [2][5]." Don't cite a source for general framing/transition text.
3. If the sources don't fully answer the question, say so explicitly in the answer instead of filling \
gaps with outside knowledge or speculation.
4. used_source_indices must list every source number you actually cited, in ascending order, with no \
duplicates.
5. Be concise and precise -- this is for a researcher, not a general audience."""

RETRIEVAL_TOP_K = 8
MIN_CHUNKS_BEFORE_WIDENING = 4  # if a signal_type/asset_class filter leaves fewer than this, drop it
HYBRID_EXPLAIN_TOP_N = 3  # explain only the top-ranked structured matches, not the whole match set --
# otherwise generic relevance ranking can surface a lower-ranked paper's chunks over the
# actual top-ranked (e.g. highest-Sharpe) match, describing the wrong paper as "the best".


class SynthesizedAnswer(BaseModel):
    answer: str = Field(description="The answer, with inline [N] citation markers referencing the numbered sources.")
    used_source_indices: list[int] = Field(description="Every source number actually cited, ascending, no duplicates.")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence the sources adequately answer the question.")
    caveat: str = Field(description="What's missing/uncertain, if anything. Empty string if none.")


def classify_query(client, model: str, query: str) -> QueryClassification:
    return generate_structured(
        client, model,
        contents=f"Classify this question:\n\n{query}",
        response_schema=QueryClassification,
        system_instruction=CLASSIFIER_SYSTEM_INSTRUCTION,
    )


def _format_sources(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({c['paper_id']}, section \"{c['section_title']}\", pp{c['page_start']}-{c['page_end']}):\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )


def synthesize_answer(client, model: str, query: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "answer": "No relevant passages were found in the corpus for this question.",
            "sources": [],
            "confidence": "low",
            "caveat": "Retrieval returned zero results.",
        }
    result = generate_structured(
        client, model,
        contents=f"Question: {query}\n\nSources:\n{_format_sources(chunks)}",
        response_schema=SynthesizedAnswer,
        system_instruction=SYNTHESIS_SYSTEM_INSTRUCTION,
    )
    cited = [chunks[i - 1] for i in result.used_source_indices if 1 <= i <= len(chunks)]
    return {
        "answer": result.answer,
        "sources": cited,
        "confidence": result.confidence,
        "caveat": result.caveat,
    }


def answer_structured(classification: QueryClassification, records: list[dict] | None = None) -> dict:
    matches = run_structured_query(classification, records=records)
    if not matches:
        summary = "No papers in the corpus match those filters."
    else:
        filt_bits = []
        if classification.signal_type_filter:
            filt_bits.append("signal_type in " + ", ".join(classification.signal_type_filter))
        if classification.asset_class_filter:
            filt_bits.append("asset_class in " + ", ".join(classification.asset_class_filter))
        if classification.metric_keyword:
            op = classification.metric_operator
            thr = classification.metric_threshold
            filt_bits.append(f"{classification.metric_keyword} {op} {thr}" if op != "none" else classification.metric_keyword)
        summary = f"Found {len(matches)} paper(s)" + (f" matching {'; '.join(filt_bits)}." if filt_bits else ".")
    return {"summary": summary, "matches": matches}


def answer_retrieval(client, model: str, retriever: HybridRetriever, query: str, classification: QueryClassification) -> dict:
    signal_type = classification.signal_type_filter[0] if classification.signal_type_filter else None
    asset_class = classification.asset_class_filter[0] if classification.asset_class_filter else None
    chunks = retriever.search(query, top_k=RETRIEVAL_TOP_K, signal_type=signal_type, asset_class=asset_class)

    widened = False
    if len(chunks) < MIN_CHUNKS_BEFORE_WIDENING and (signal_type or asset_class):
        # An inferred signal_type+asset_class filter is a hard AND that can leave
        # almost nothing to answer from (e.g. only 3 papers in the whole corpus
        # are tagged both momentum AND fx) -- degrade to an unfiltered search
        # rather than starve the LLM of context for a prose explanation question.
        logger.info("Only %d chunks with filter (signal_type=%s, asset_class=%s); widening to unfiltered search", len(chunks), signal_type, asset_class)
        chunks = retriever.search(query, top_k=RETRIEVAL_TOP_K)
        widened = True

    result = synthesize_answer(client, model, query, chunks)
    if widened:
        note = "Filtered to too few results, so this searched the full corpus instead of just the inferred filter."
        result["caveat"] = f"{result['caveat']} {note}".strip() if result.get("caveat") else note
    return result


def answer_hybrid(client, model: str, retriever: HybridRetriever, query: str, classification: QueryClassification, records: list[dict] | None = None) -> dict:
    structured = answer_structured(classification, records=records)
    if not structured["matches"]:
        return {**structured, "answer": None, "sources": [], "confidence": "low",
                "caveat": "No papers matched the filters, so there is nothing to explain."}
    # Only explain the top-ranked matches -- when matches are sorted by a metric
    # (e.g. "best Sharpe"), scoping retrieval to the whole match set lets generic
    # relevance ranking surface a lower-ranked paper's chunks over the actual
    # top-ranked one, describing the wrong paper as "the best".
    paper_ids = {m["paper_id"] for m in structured["matches"][:HYBRID_EXPLAIN_TOP_N]}
    chunks = retriever.search(query, top_k=RETRIEVAL_TOP_K, paper_ids=paper_ids)
    synthesis = synthesize_answer(client, model, query, chunks)
    return {**structured, **synthesis}


def answer_question(
    query: str,
    client=None,
    model: str = DEFAULT_MODEL,
    retriever: HybridRetriever | None = None,
    records: list[dict] | None = None,
) -> dict:
    """Top-level entry point: classifies the query, routes it, and returns a
    dict with at least {mode, classification}. Structured/hybrid results also
    carry {summary, matches}; retrieval/hybrid results also carry
    {answer, sources, confidence, caveat}."""
    client = client or get_client()
    classification = classify_query(client, model, query)
    result: dict = {"mode": classification.mode, "classification": classification.model_dump()}

    if classification.mode == "structured":
        result.update(answer_structured(classification, records=records))
    else:
        if retriever is None:
            retriever = HybridRetriever(model_slug="bge-large", verbose=False)
        if classification.mode == "retrieval":
            result.update(answer_retrieval(client, model, retriever, query, classification))
        else:
            result.update(answer_hybrid(client, model, retriever, query, classification, records=records))

    return result


def _print_result(result: dict) -> None:
    print(f"\nMode: {result['mode']}  ({result['classification']['reasoning']})\n")
    if "summary" in result:
        print(result["summary"])
        for m in result["matches"][:15]:
            mm = m.get("matching_metric")
            metric_str = f" | {mm['metric_name']}={mm['value']}{mm['unit']}" if mm else ""
            print(f"  - {m['paper_id']} | {m['title'][:60]}{metric_str}")
        print()
    if result.get("answer"):
        print(result["answer"])
        print(f"\n[confidence: {result['confidence']}]" + (f"  caveat: {result['caveat']}" if result.get("caveat") else ""))
        if result.get("sources"):
            print("\nSources:")
            for s in result["sources"]:
                print(f"  - {s['paper_id']} | {s['section_title']} (pp{s['page_start']}-{s['page_end']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question over the paper corpus")
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    result = answer_question(args.query, model=args.model)
    _print_result(result)


if __name__ == "__main__":
    main()
