"""
Embed chunks/chunks.jsonl with a chosen sentence-transformers model and build
a FAISS index. Supports multiple models side by side (index/<model_slug>/) so
they can be compared later on a retrieval eval set (recall@k).

Each embedded model needs its own query/passage prefix convention -- BGE and
E5 give noticeably worse results if you skip theirs.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# faiss is imported lazily in main(), AFTER the CUDA embedding model is
# constructed -- not here. On Windows, faiss bundles an OpenMP runtime that
# conflicts with torch's CUDA runtime and segfaults the process the moment a
# CUDA context initializes *after* faiss has already loaded. What matters is
# the order CUDA-init vs. faiss-load actually happen at runtime, not the
# order of import statements in the file -- faiss must not be imported until
# the SentenceTransformer model is already constructed on CUDA.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
CHUNKS_PATH = PROJECT_ROOT / "chunks" / "chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "index"

# hf_name: the sentence-transformers model id.
# query_prefix / passage_prefix: instruction strings these models expect
# prepended at inference time (both models perform noticeably worse without).
MODEL_REGISTRY = {
    "bge-large": {
        "hf_name": "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    "e5-large": {
        "hf_name": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_embed_text(chunk: dict) -> str:
    """Light context augmentation: bare 500-800 token passages lose the paper
    title/section framing that helps disambiguate short or table-only chunks."""
    return f"{chunk['title']}\n{chunk['section_title']}\n\n{chunk['text']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks and build a FAISS index")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), required=True)
    parser.add_argument("--chunks-path", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    spec = MODEL_REGISTRY[args.model]
    out_dir = args.index_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(args.chunks_path)
    logger.info("Loaded %d chunks", len(chunks))

    logger.info("Loading model %s", spec["hf_name"])
    model = SentenceTransformer(spec["hf_name"])
    logger.info("Model loaded on device: %s", model.device)

    import faiss  # deferred -- see module-level note on import order

    texts = [spec["passage_prefix"] + build_embed_text(c) for c in chunks]
    logger.info("Encoding %d passages (batch_size=%d)...", len(texts), args.batch_size)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("Built FAISS index: %d vectors, dim=%d", index.ntotal, dim)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    with (out_dir / "chunk_ids.json").open("w", encoding="utf-8") as f:
        json.dump([c["chunk_id"] for c in chunks], f)
    with (out_dir / "model_info.json").open("w", encoding="utf-8") as f:
        json.dump({"slug": args.model, **spec, "dim": dim, "num_vectors": index.ntotal}, f, indent=2)

    logger.info("Saved index to %s", out_dir)


if __name__ == "__main__":
    main()
