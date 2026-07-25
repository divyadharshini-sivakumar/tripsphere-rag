"""
TripSphere Multi-Document RAG — Streamlit Web Interface
"""
import streamlit as st
from pathlib import Path
import tempfile
import os

from config import DATA_DIR, TOP_K, SUPPORTED_EXTENSIONS
from ingest import ingest_files, get_embeddings, get_vectorstore
from rag_pipeline import answer_question, get_collection_stats, retrieve
from loaders import load_document

st.set_page_config(
    page_title="TripSphere RAG Assistant",
    page_icon="✈️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resources (avoid reloading model / reconnecting on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource
def cached_embeddings():
    return get_embeddings()


@st.cache_resource
def cached_vectorstore():
    return get_vectorstore(cached_embeddings())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("✈️ TripSphere RAG")
    st.caption("Multi-Document Retrieval-Augmented Generation")
    st.markdown("---")

    st.subheader("1. Upload Documents")
    uploaded = st.file_uploader(
        "PDF, CSV, or TXT",
        type=["pdf", "csv", "txt"],
        accept_multiple_files=True,
    )

    reset_db = st.checkbox("Reset vector store before ingest", value=False)

    if st.button("Index Uploaded Files", type="primary", use_container_width=True):
        if not uploaded:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Ingesting..."):
                paths = []
                for f in uploaded:
                    dest = DATA_DIR / f.name
                    dest.write_bytes(f.getvalue())
                    paths.append(dest)
                try:
                    summary = ingest_files(paths, reset=reset_db)
                    st.success(
                        f"Ingested {summary.get('chunks', 0)} chunks "
                        f"from {summary.get('documents', 0)} docs."
                    )
                    st.json(summary)
                    # Clear cache so new data is visible
                    cached_vectorstore.clear()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.markdown("---")
    st.subheader("2. Collection Stats")
    if st.button("Refresh Stats", use_container_width=True):
        st.session_state["stats"] = get_collection_stats()

    stats = st.session_state.get("stats") or get_collection_stats()
    if "error" in stats:
        st.warning("Vector store empty or unavailable.")
    else:
        st.metric("Vectors", stats.get("count", 0))
        st.write("**Types:**", stats.get("document_types", {}))
        st.write("**Sources:**", stats.get("sources", []))
        st.caption(
            f"Model: {stats.get('embedding_model', 'n/a')} "
            f"({stats.get('embedding_dimension', '?')}d)"
        )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("TripSphere Knowledge Assistant")
st.markdown(
    "Ask questions about hotel policies, pricing, bookings, and travel FAQs. "
    "Answers are grounded in your uploaded documents."
)

col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input(
        "Your question",
        placeholder="e.g. What is the cancellation policy for flexible rates?",
    )
with col2:
    filter_type = st.selectbox(
        "Filter by type",
        options=["all", "pdf", "csv", "txt"],
        index=0,
    )
    top_k = st.slider("Top-K chunks", min_value=1, max_value=8, value=TOP_K)

ask = st.button("Ask", type="primary")

if ask and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        try:
            result = answer_question(
                question.strip(), k=top_k, filter_type=filter_type
            )
            st.session_state["last_result"] = result
            st.session_state["last_question"] = question.strip()
        except Exception as e:
            st.error(f"Error: {e}")
            st.info(
            "Make sure OPENROUTER_API_KEY is set in your environment or .env file."
            )

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    st.subheader("Answer")
    st.write(result["answer"])
    if result.get("sources"):
        st.caption("Sources: " + ", ".join(result["sources"]))

    with st.expander("Retrieved passages (evidence)", expanded=False):
        for i, doc in enumerate(result.get("documents", []), 1):
            meta = doc.metadata
            st.markdown(
                f"**[{i}]** `{meta.get('source')}` · "
                f"type=`{meta.get('document_type')}` · "
                f"chunk=`{meta.get('chunk_id')}` · "
                f"loc=`{meta.get('page', meta.get('row', '-'))}`"
            )
            st.text(doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""))
            st.markdown("---")

st.markdown("---")
st.caption(
    "TripSphere RAG Lab · Local HuggingFace embeddings · "
    "OpenRouter LLM · Persistent ChromaDB"
)

