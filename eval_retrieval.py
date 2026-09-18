"""
Retrieval evaluation harness: recall@5, recall@10, and MRR@10 for each of the
4 requested ablations, isolating one variable at a time from a shared
baseline (section-aware chunks, bge-large, hybrid dense+BM25, reranked):

    1. chunking strategy : section-aware (baseline)  vs  fixed-size
    2. embedding model   : bge-large (baseline)       vs  e5-large
    3. retrieval mode    : hybrid (baseline)          vs  dense-only
    4. reranking         : with reranker (baseline)   vs  without

Relevance is judged at the PAPER level (a hit if the retrieved chunk's
paper_id is in the query's gold set) rather than chunk level, since gold
chunk IDs from section-aware chunking don't even exist in the fixed-size
index -- paper-level relevance is what makes the chunking comparison valid
in the first place.

Saves results/eval_results.csv and results/eval_results.md.
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from retrieve import HybridRetriever, load_jsonl

PROJECT_ROOT = Path(__file__).parent
EVAL_QUERIES_PATH = PROJECT_ROOT / "eval" / "eval_queries.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval"

TOP_K = 10  # retrieval depth; recall@5 and recall@10 both read from this same run

# (config_name, chunks_path, index_dir, model_slug, use_bm25, rerank)
RUN_CONFIGS = [
    ("section-aware + bge-large + hybrid + rerank (baseline)",
     PROJECT_ROOT / "chunks" / "chunks.jsonl", PROJECT_ROOT / "index", "bge-large", True, True),
    ("fixed-size + bge-large + hybrid + rerank",
     PROJECT_ROOT / "chunks" / "chunks_fixedsize.jsonl", PROJECT_ROOT / "index_fixedsize", "bge-large", True, True),
    ("section-aware + e5-large + hybrid + rerank",
     PROJECT_ROOT / "chunks" / "chunks.jsonl", PROJECT_ROOT / "index", "e5-large", True, True),
    ("section-aware + bge-large + dense-only + rerank",
     PROJECT_ROOT / "chunks" / "chunks.jsonl", PROJECT_ROOT / "index", "bge-large", False, True),
    ("section-aware + bge-large + hybrid + no-rerank",
     PROJECT_ROOT / "chunks" / "chunks.jsonl", PROJECT_ROOT / "index", "bge-large", True, False),
]

# Which two rows to compare under each named ablation, and which row is "baseline" in that pair.
COMPARISONS = [
    ("Chunking strategy", "section-aware + bge-large + hybrid + rerank (baseline)", "fixed-size + bge-large + hybrid + rerank"),
    ("Embedding model", "section-aware + bge-large + hybrid + rerank (baseline)", "section-aware + e5-large + hybrid + rerank"),
    ("Retrieval mode", "section-aware + bge-large + hybrid + rerank (baseline)", "section-aware + bge-large + dense-only + rerank"),
    ("Reranking", "section-aware + bge-large + hybrid + rerank (baseline)", "section-aware + bge-large + hybrid + no-rerank"),
]


def load_queries() -> list[dict]:
    return load_jsonl(EVAL_QUERIES_PATH)


def evaluate_run(retriever: HybridRetriever, queries: list[dict], use_bm25: bool, rerank: bool) -> dict:
    recall5_hits, recall10_hits, rr_sum = 0, 0, 0.0
    n = len(queries)
    for q in queries:
        gold = set(q["relevant_paper_ids"])
        results = retriever.search(q["query"], top_k=TOP_K, rerank=rerank, use_bm25=use_bm25)
        ranked_paper_ids = [r["paper_id"] for r in results]

        if any(pid in gold for pid in ranked_paper_ids[:5]):
            recall5_hits += 1
        if any(pid in gold for pid in ranked_paper_ids[:10]):
            recall10_hits += 1
        for rank, pid in enumerate(ranked_paper_ids, start=1):
            if pid in gold:
                rr_sum += 1.0 / rank
                break

    return {
        "recall@5": recall5_hits / n,
        "recall@10": recall10_hits / n,
        "mrr@10": rr_sum / n,
        "n_queries": n,
    }


def main() -> None:
    queries = load_queries()
    print(f"Loaded {len(queries)} eval queries\n")

    results_by_name = {}
    for name, chunks_path, index_dir, model_slug, use_bm25, rerank in RUN_CONFIGS:
        print(f"=== Running: {name} ===")
        retriever = HybridRetriever(model_slug, chunks_path=chunks_path, index_dir=index_dir, verbose=True)
        metrics = evaluate_run(retriever, queries, use_bm25=use_bm25, rerank=rerank)
        results_by_name[name] = metrics
        print(f"  recall@5={metrics['recall@5']:.3f}  recall@10={metrics['recall@10']:.3f}  mrr@10={metrics['mrr@10']:.3f}\n")
        del retriever  # free GPU memory before loading the next model

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- CSV ---
    csv_path = RESULTS_DIR / "eval_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("config,recall@5,recall@10,mrr@10,n_queries\n")
        for name, m in results_by_name.items():
            f.write(f'"{name}",{m["recall@5"]:.4f},{m["recall@10"]:.4f},{m["mrr@10"]:.4f},{m["n_queries"]}\n')

    # --- Markdown ---
    md_path = RESULTS_DIR / "eval_results.md"
    lines = [
        "# Retrieval evaluation results",
        "",
        f"{len(queries)} hand-labeled queries, paper-level relevance, retrieval depth = {TOP_K}.",
        "",
        "## All runs",
        "",
        "| Config | recall@5 | recall@10 | MRR@10 |",
        "|---|---|---|---|",
    ]
    for name, m in results_by_name.items():
        lines.append(f"| {name} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | {m['mrr@10']:.3f} |")

    lines += ["", "## Ablations (baseline vs. one changed variable)", ""]
    for comparison_name, base_name, variant_name in COMPARISONS:
        base, variant = results_by_name[base_name], results_by_name[variant_name]
        lines += [
            f"### {comparison_name}",
            "",
            "| Config | recall@5 | recall@10 | MRR@10 |",
            "|---|---|---|---|",
            f"| {base_name} | {base['recall@5']:.3f} | {base['recall@10']:.3f} | {base['mrr@10']:.3f} |",
            f"| {variant_name} | {variant['recall@5']:.3f} | {variant['recall@10']:.3f} | {variant['mrr@10']:.3f} |",
            "",
        ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
