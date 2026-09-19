"""
Streamlit Q&A UI over the paper corpus. Routes every question through
qa.answer_question(), which classifies it as structured (exact filter/sort
over extracted/*.json), retrieval (cited prose answer from HybridRetriever
chunks), or hybrid (both).

    streamlit run app.py

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment, chunks/ +
index/ built (chunker.py, embed.py), and extracted/ populated
(structured_extract.py).
"""

import os

import streamlit as st

from llm_utils import DEFAULT_MODEL, get_client
from qa import answer_question
from retrieve import HybridRetriever
from structured_query import load_extracted_records

st.set_page_config(page_title="Quant Research Q&A", page_icon="\U0001F4CA", layout="wide")


@st.cache_resource(show_spinner="Connecting to Gemini...")
def _get_client():
    return get_client()


@st.cache_resource(show_spinner="Loading retrieval index (embedding model + FAISS + BM25)...")
def _get_retriever():
    return HybridRetriever(model_slug="bge-large", verbose=False)


@st.cache_data(show_spinner=False)
def _get_records():
    return load_extracted_records()


def render_structured(result: dict) -> None:
    st.markdown(f"**{result['summary']}**")
    if not result["matches"]:
        return
    rows = []
    for m in result["matches"]:
        mm = m.get("matching_metric")
        rows.append({
            "Paper": m["paper_id"],
            "Title": m["title"],
            "Signal type": ", ".join(m["signal_type"]),
            "Asset class": ", ".join(m["asset_class"]),
            "Matching metric": f"{mm['metric_name']} = {mm['value']} {mm['unit']}" if mm else "",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    with st.expander("Methodology summaries"):
        for m in result["matches"]:
            st.markdown(f"**{m['paper_id']}** -- {m['title']}")
            st.caption(m["methodology_summary"])
            if m.get("matching_metric"):
                mm = m["matching_metric"]
                st.caption(f"Evidence ({mm['source_location']}): “{mm['evidence_quote']}”")


def render_answer(result: dict) -> None:
    if result.get("answer") is None:
        return
    st.markdown(result["answer"])
    badge = {"high": "\U0001F7E2", "medium": "\U0001F7E1", "low": "\U0001F534"}.get(result.get("confidence"), "")
    st.caption(f"{badge} confidence: {result.get('confidence', 'n/a')}" + (f"  |  {result['caveat']}" if result.get("caveat") else ""))
    if result.get("sources"):
        with st.expander(f"Sources ({len(result['sources'])})"):
            for s in result["sources"]:
                st.markdown(f"**{s['paper_id']}** -- {s['title']}")
                st.caption(f"{s['section_title']} (pp{s['page_start']}-{s['page_end']}, {s['chunk_type']})")
                st.text(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))
                st.divider()


def main() -> None:
    st.title("Quant Research Q&A")
    st.caption("Ask about the corpus of empirical quant-finance papers. Questions are routed automatically to an exact structured filter, a cited retrieval answer, or both.")

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        st.error("No GEMINI_API_KEY (or GOOGLE_API_KEY) found in the environment. Set it and restart.")
        st.stop()

    with st.sidebar:
        st.subheader("About")
        st.markdown(
            "- **Structured** mode: exact filter/sort over validated extracted facts "
            "(signal type, asset class, performance metrics). No LLM guessing on numbers.\n"
            "- **Retrieval** mode: cited answer synthesized from the paper text most relevant to your question.\n"
            "- **Hybrid** mode: both -- filters to matching papers, then explains them."
        )
        st.divider()
        model = st.text_input("Gemini model", value=DEFAULT_MODEL)
        records = _get_records()
        st.caption(f"{len(records)} papers with structured extractions loaded.")

    examples = [
        "Which papers report a Sharpe ratio above 2 in FX carry strategies?",
        "How do papers construct time-series momentum trading signals?",
        "What's the best performing statistical arbitrage strategy and how does it work?",
    ]
    query = st.text_input("Your question", placeholder=examples[0])
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            query = ex
            st.session_state["_query_override"] = ex
    query = st.session_state.get("_query_override", query)

    if not query:
        return

    client = _get_client()

    with st.spinner("Classifying and answering..."):
        try:
            retriever = _get_retriever()
            result = answer_question(query, client=client, model=model, retriever=retriever, records=records)
        except Exception as e:  # noqa: BLE001 - surface any failure to the user instead of a blank page
            st.error(f"Something went wrong answering this question: {e}")
            return

    mode = result["mode"]
    mode_label = {"structured": "\U0001F522 Structured filter", "retrieval": "\U0001F4D6 Retrieval", "hybrid": "\U0001F500 Hybrid"}[mode]
    st.markdown(f"### {mode_label}")
    st.caption(result["classification"]["reasoning"])

    if mode in ("structured", "hybrid"):
        render_structured(result)
    if mode in ("retrieval", "hybrid"):
        render_answer(result)


if __name__ == "__main__":
    main()
