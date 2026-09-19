"""
Hybrid retrieval over chunks/chunks.jsonl: dense (FAISS) + lexical (BM25),
merged with reciprocal rank fusion, then cross-encoder reranked.

    python retrieve.py --model bge-large --query "momentum in commodities"

Requires an index already built by embed.py for the chosen --model.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from embed import MODEL_REGISTRY, build_embed_text

# faiss is imported lazily inside HybridRetriever.__init__, AFTER the CUDA
# embedding model is constructed -- not here, and not reordered above. On
# Windows, faiss bundles an OpenMP runtime that conflicts with torch's CUDA
# runtime and segfaults the process the moment a CUDA context initializes
# *after* faiss has loaded. Importing faiss merely before the `sentence_transformers`
# package (as opposed to before the model is actually constructed on CUDA) does
# NOT avoid this -- the crash is tied to CUDA context init order, not import
# statement order, so faiss must not be imported until CUDA is already live.

PROJECT_ROOT = Path(__file__).parent
CHUNKS_PATH = PROJECT_ROOT / "chunks" / "chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "index"
CURATION_NOTES_PATH = PROJECT_ROOT / "arxiv_papers" / "curation_notes.jsonl"

RRF_K = 60
DENSE_TOP_N = 30
BM25_TOP_N = 30
RERANK_TOP_N = 20
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_paper_tags(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {rec["paper_id"]: rec for rec in load_jsonl(path)}


class HybridRetriever:
    def __init__(
        self,
        model_slug: str,
        chunks_path: Path = CHUNKS_PATH,
        index_dir: Path = INDEX_DIR,
        verbose: bool = True,
    ):
        self._log = print if verbose else (lambda *a, **k: None)
        self.chunks = load_jsonl(chunks_path)
        self.chunks_by_id = {c["chunk_id"]: c for c in self.chunks}
        self.paper_tags = load_paper_tags(CURATION_NOTES_PATH)

        self._log(f"Tokenizing {len(self.chunks)} chunks for BM25...")
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

        index_dir = index_dir / model_slug
        with (index_dir / "model_info.json").open("r", encoding="utf-8") as f:
            self.model_info = json.load(f)

        # Load the CUDA embedding model BEFORE faiss is ever imported -- see
        # the module-level note above on why the order matters here.
        self._log(f"Loading embedding model {self.model_info['hf_name']}...")
        self.embed_model = SentenceTransformer(self.model_info["hf_name"])
        self.reranker = None  # lazy-loaded, only needed if reranking is requested

        import faiss
        self.faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
        with (index_dir / "chunk_ids.json").open("r", encoding="utf-8") as f:
            self.faiss_chunk_ids = json.load(f)

    def _dense_search(self, query: str, top_n: int) -> list[tuple[str, float]]:
        text = self.model_info["query_prefix"] + query
        vec = self.embed_model.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        scores, indices = self.faiss_index.search(vec, top_n)
        return [
            (self.faiss_chunk_ids[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]

    def _bm25_search(self, query: str, top_n: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(self.chunks[i]["chunk_id"], float(scores[i])) for i in top_indices if scores[i] > 0]

    def _apply_metadata_filter(
        self,
        results: list[tuple[str, float]],
        signal_type: str | None,
        asset_class: str | None,
        paper_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        if not signal_type and not asset_class and paper_ids is None:
            return results

        def matches(chunk_id: str) -> bool:
            paper_id = self.chunks_by_id[chunk_id]["paper_id"]
            if paper_ids is not None and paper_id not in paper_ids:
                return False
            tags = self.paper_tags.get(paper_id)
            if not tags:
                return paper_ids is not None  # allow an explicit paper_id match even with no tags
            if signal_type and signal_type not in tags.get("signal_type", []):
                return False
            if asset_class and asset_class not in tags.get("asset_class", []):
                return False
            return True

        return [(cid, score) for cid, score in results if matches(cid)]

    @staticmethod
    def _reciprocal_rank_fusion(*ranked_lists: list[tuple[str, float]]) -> list[tuple[str, float]]:
        fused: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, (chunk_id, _score) in enumerate(ranked, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    def _rerank(self, query: str, chunk_ids: list[str]) -> list[tuple[str, float]]:
        if self.reranker is None:
            self._log(f"Loading reranker {RERANK_MODEL}...")
            self.reranker = CrossEncoder(RERANK_MODEL)
        pairs = [(query, build_embed_text(self.chunks_by_id[cid])) for cid in chunk_ids]
        scores = self.reranker.predict(pairs)
        return sorted(zip(chunk_ids, (float(s) for s in scores)), key=lambda kv: kv[1], reverse=True)

    def search(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True,
        use_bm25: bool = True,
        signal_type: str | None = None,
        asset_class: str | None = None,
        paper_ids: set[str] | None = None,
    ) -> list[dict]:
        """paper_ids, if given, restricts results to that set of papers -- used
        by qa.py's hybrid mode to explain mechanism for a structured-query
        match set rather than searching the whole corpus."""
        dense = self._apply_metadata_filter(self._dense_search(query, DENSE_TOP_N), signal_type, asset_class, paper_ids)
        if use_bm25:
            bm25 = self._apply_metadata_filter(self._bm25_search(query, BM25_TOP_N), signal_type, asset_class, paper_ids)
            fused = self._reciprocal_rank_fusion(dense, bm25)
        else:
            fused = dense

        if rerank and fused:
            candidate_ids = [cid for cid, _ in fused[:RERANK_TOP_N]]
            final = self._rerank(query, candidate_ids)[:top_k]
        else:
            final = fused[:top_k]

        results = []
        for chunk_id, score in final:
            chunk = self.chunks_by_id[chunk_id]
            results.append(
                {
                    "score": score,
                    "chunk_id": chunk_id,
                    "paper_id": chunk["paper_id"],
                    "title": chunk["title"],
                    "section_title": chunk["section_title"],
                    "chunk_type": chunk["chunk_type"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "text": chunk["text"],
                }
            )
        return results


def format_result(rank: int, r: dict) -> str:
    pages = f"p{r['page_start']}" if r["page_start"] == r["page_end"] else f"pp{r['page_start']}-{r['page_end']}"
    snippet = r["text"][:280].replace("\n", " ")
    return (
        f"[{rank}] score={r['score']:.4f}  {r['title']}\n"
        f"    {r['section_title']} ({pages}) -- {r['chunk_type']}\n"
        f"    {snippet}...\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the hybrid retrieval index")
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="bge-large")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--dense-only", action="store_true", help="Skip BM25/RRF fusion")
    parser.add_argument("--signal-type", default=None)
    parser.add_argument("--asset-class", default=None)
    args = parser.parse_args()

    retriever = HybridRetriever(args.model)
    results = retriever.search(
        args.query,
        top_k=args.top_k,
        rerank=not args.no_rerank,
        use_bm25=not args.dense_only,
        signal_type=args.signal_type,
        asset_class=args.asset_class,
    )

    print(f"\nQuery: {args.query!r}  (model={args.model}, rerank={not args.no_rerank})\n")
    for i, r in enumerate(results, start=1):
        print(format_result(i, r))


if __name__ == "__main__":
    main()
