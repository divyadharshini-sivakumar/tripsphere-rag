"""
TripSphere Multi-Document RAG — Streamlit Web Interface
"""

import streamlit as st

from config import DATA_DIR, TOP_K
from ingest import (
    get_embeddings,
    get_vectorstore,
    ingest_data_directory,
    ingest_files,
)
from rag_pipeline import answer_question, get_collection_stats


st.set_page_config(
    page_title="TripSphere RAG Assistant",
    page_icon="✈️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def cached_embeddings():
    """Load the embedding model only once."""
    return get_embeddings()


@st.cache_resource
def cached_vectorstore():
    """Connect to ChromaDB only once."""
    return get_vectorstore(cached_embeddings())


@st.cache_resource
def initialize_vector_database():
    """
    Automatically build the vector database when the deployed app starts.

    Streamlit Cloud uses a fresh environment, so the local ChromaDB must be
    regenerated from the files stored inside the data folder.
    """
    stats = get_collection_stats()

    if stats.get("count", 0) > 0:
        return {
            "status": "existing",
            "chunks": stats.get("count", 0),
        }

    return ingest_data_directory(reset=False)


# ---------------------------------------------------------------------------
# Automatically initialize ChromaDB
# ---------------------------------------------------------------------------
try:
    with st.spinner("Preparing the TripSphere knowledge base..."):
        initialization = initialize_vector_database()

    if initialization.get("status") == "empty":
        st.warning(
            "No documents were found in the data folder. "
            "Upload PDF, CSV, or TXT files from the sidebar."
        )

except Exception as error:
    st.error(f"Knowledge-base initialization failed: {error}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("✈️ TripSphere RAG")
    st.caption("Multi-Document Retrieval-Augmented Generation")
    st.markdown("---")

    st.subheader("1. Upload Documents")

    uploaded_files = st.file_uploader(
        "PDF, CSV, or TXT",
        type=["pdf", "csv", "txt"],
        accept_multiple_files=True,
    )

    reset_database = st.checkbox(
        "Reset vector store before ingest",
        value=False,
    )

    if st.button(
        "Index Uploaded Files",
        type="primary",
        use_container_width=True,
    ):
        if not uploaded_files:
            st.warning("Please upload at least one file.")

        else:
            with st.spinner("Ingesting uploaded documents..."):
                try:
                    file_paths = []

                    for uploaded_file in uploaded_files:
                        destination = DATA_DIR / uploaded_file.name
                        destination.write_bytes(uploaded_file.getvalue())
                        file_paths.append(destination)

                    summary = ingest_files(
                        file_paths=file_paths,
                        reset=reset_database,
                    )

                    st.success(
                        f"Ingested {summary.get('chunks', 0)} chunks "
                        f"from {summary.get('documents', 0)} documents."
                    )

                    st.json(summary)

                    cached_vectorstore.clear()
                    st.session_state["stats"] = get_collection_stats()

                except Exception as error:
                    st.error(f"Ingestion failed: {error}")

    st.markdown("---")
    st.subheader("2. Collection Stats")

    if st.button(
        "Refresh Stats",
        use_container_width=True,
    ):
        st.session_state["stats"] = get_collection_stats()

    stats = st.session_state.get("stats")

    if stats is None:
        stats = get_collection_stats()

    if "error" in stats:
        st.warning("Vector store is empty or unavailable.")

    else:
        st.metric(
            "Vectors",
            stats.get("count", 0),
        )

        st.write(
            "**Types:**",
            stats.get("document_types", {}),
        )

        st.write(
            "**Sources:**",
            stats.get("sources", []),
        )

        st.caption(
            f"Model: {stats.get('embedding_model', 'n/a')} "
            f"({stats.get('embedding_dimension', '?')} dimensions)"
        )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("TripSphere Knowledge Assistant")

st.markdown(
    "Ask questions about hotel policies, pricing, bookings, and travel FAQs. "
    "Answers are grounded in the uploaded TripSphere documents."
)

question_column, settings_column = st.columns([3, 1])

with question_column:
    question = st.text_input(
        "Your question",
        placeholder=(
            "Example: What is the cancellation policy "
            "for flexible rates?"
        ),
    )

with settings_column:
    filter_type = st.selectbox(
        "Filter by type",
        options=["all", "pdf", "csv", "txt"],
        index=0,
    )

    top_k = st.slider(
        "Top-K chunks",
        min_value=1,
        max_value=8,
        value=TOP_K,
    )


ask_button = st.button(
    "Ask",
    type="primary",
)


if ask_button:
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Retrieving documents and generating an answer..."):
            try:
                result = answer_question(
                    question=question.strip(),
                    k=top_k,
                    filter_type=filter_type,
                )

                st.session_state["last_result"] = result
                st.session_state["last_question"] = question.strip()

            except Exception as error:
                st.error(f"Error: {error}")

                st.info(
                    "Make sure OPENROUTER_API_KEY and OPENROUTER_MODEL "
                    "are configured in Streamlit Secrets."
                )


# ---------------------------------------------------------------------------
# Display result
# ---------------------------------------------------------------------------
if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    st.subheader("Answer")
    st.write(result.get("answer", "No answer was generated."))

    if result.get("sources"):
        st.caption(
            "Sources: " + ", ".join(result["sources"])
        )

    with st.expander(
        "Retrieved passages (evidence)",
        expanded=False,
    ):
        documents = result.get("documents", [])

        if not documents:
            st.info("No passages were retrieved.")

        for index, document in enumerate(documents, start=1):
            metadata = document.metadata

            location = metadata.get(
                "page",
                metadata.get("row", "-"),
            )

            st.markdown(
                f"**[{index}]** `{metadata.get('source', 'unknown')}` · "
                f"type=`{metadata.get('document_type', 'unknown')}` · "
                f"chunk=`{metadata.get('chunk_id', '-')}` · "
                f"location=`{location}`"
            )

            content = document.page_content

            st.text(
                content[:600]
                + ("..." if len(content) > 600 else "")
            )

            st.markdown("---")


st.markdown("---")

st.caption(
    "TripSphere RAG Lab · Local Hugging Face embeddings · "
    "OpenRouter LLM · Persistent ChromaDB"
)
