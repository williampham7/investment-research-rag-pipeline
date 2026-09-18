"""
arXiv ingestion: fetch quant finance papers via the arXiv API and download
their PDFs into arxiv_papers/, alongside a metadata.jsonl sidecar.
"""

import argparse
import json
import logging
from pathlib import Path

import arxiv
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# arXiv q-fin categories relevant to tradable signals/strategies (see project spec §3)
DEFAULT_CATEGORIES = ["q-fin.PM", "q-fin.TR", "q-fin.ST", "q-fin.CP"]

OUTPUT_DIR = Path(__file__).parent / "arxiv_papers"
METADATA_FILE = OUTPUT_DIR / "metadata.jsonl"


def build_query(categories: list[str], search_terms: str | None) -> str:
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    if search_terms:
        return f"({cat_query}) AND ({search_terms})"
    return cat_query


def short_id(result: arxiv.Result) -> str:
    # entry_id looks like "https://arxiv.org/abs/2401.01234v1" (new-style) or
    # "https://arxiv.org/abs/math/0703743v1" (old-style, pre-2007) — split on
    # "/abs/" rather than the last "/" so the old-style archive prefix survives.
    return result.entry_id.split("/abs/", 1)[-1]


def download_pdf(result: arxiv.Result, pdf_path: Path) -> None:
    if not result.pdf_url:
        raise ValueError("result has no pdf_url")
    response = requests.get(result.pdf_url, timeout=30)
    response.raise_for_status()
    pdf_path.write_bytes(response.content)


def paper_to_metadata(result: arxiv.Result, pdf_path: Path) -> dict:
    return {
        "paper_id": f"arxiv:{short_id(result)}",
        "title": result.title,
        "authors": [a.name for a in result.authors],
        "published": result.published.isoformat(),
        "updated": result.updated.isoformat(),
        "categories": result.categories,
        "primary_category": result.primary_category,
        "url": result.entry_id,
        "pdf_url": result.pdf_url,
        "abstract": result.summary,
        "pdf_path": str(pdf_path),
    }


def load_existing_ids(metadata_file: Path) -> set[str]:
    if not metadata_file.exists():
        return set()
    ids = set()
    with metadata_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["paper_id"])
    return ids


def fetch_papers(
    categories: list[str],
    search_terms: str | None,
    max_results: int,
    output_dir: Path,
    delay_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_dir / "metadata.jsonl"
    existing_ids = load_existing_ids(metadata_file)

    query = build_query(categories, search_terms)
    logger.info("Query: %s", query)

    client = arxiv.Client(page_size=100, delay_seconds=delay_seconds, num_retries=5)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    downloaded = 0
    skipped = 0
    failed = 0

    with metadata_file.open("a", encoding="utf-8") as meta_out:
        for result in client.results(search):
            sid = short_id(result)
            paper_id = f"arxiv:{sid}"

            if paper_id in existing_ids:
                skipped += 1
                continue

            safe_id = sid.replace("/", "_")
            pdf_path = output_dir / f"{safe_id}.pdf"

            try:
                download_pdf(result, pdf_path)
            except Exception as e:
                logger.warning("Failed to download %s: %s", paper_id, e)
                failed += 1
                continue

            record = paper_to_metadata(result, pdf_path)
            meta_out.write(json.dumps(record) + "\n")
            meta_out.flush()
            existing_ids.add(paper_id)
            downloaded += 1
            logger.info("Downloaded (%d): %s", downloaded, result.title)

    logger.info(
        "Done. downloaded=%d skipped(already present)=%d failed=%d",
        downloaded,
        skipped,
        failed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch quant finance papers from arXiv")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help=f"arXiv categories to search (default: {DEFAULT_CATEGORIES})",
    )
    parser.add_argument(
        "--search-terms",
        default=None,
        help='Additional arXiv query terms, e.g. \'abs:"momentum" OR abs:"Sharpe"\'',
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=150,
        help="Maximum number of papers to fetch",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to save PDFs and metadata into",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=3.0,
        help="Delay between arXiv API requests (politeness)",
    )
    args = parser.parse_args()

    fetch_papers(
        categories=args.categories,
        search_terms=args.search_terms,
        max_results=args.max_results,
        output_dir=args.output_dir,
        delay_seconds=args.delay_seconds,
    )


if __name__ == "__main__":
    main()
