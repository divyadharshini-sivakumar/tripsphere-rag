"""
TripSphere Multi-Document RAG — Ingestion Pipeline
Loads documents, chunks them, embeds with local HuggingFace model,
and persists to ChromaDB. Never calls Groq.
"""
from pathlib import Path
from typing import List, Optional
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    DATA_DIR,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
)
from loaders import load_directory, load_document


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a local sentence-transformer embedding model (no API key)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(embeddings: Optional[HuggingFaceEmbeddings] = None) -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    if embeddings is None:
        embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks and assign chunk IDs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(uuid.uuid4())[:8]
        chunk.metadata["chunk_index"] = i
    return chunks


def ingest_files(file_paths: List[Path], reset: bool = False) -> dict:
    """
    Ingest a list of files into ChromaDB.
    Returns a summary dict.
    """
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)

    if reset:
        # Delete and recreate collection
        try:
            vectorstore.delete_collection()
        except Exception:
            pass
        vectorstore = get_vectorstore(embeddings)

    all_docs: List[Document] = []
    for fp in file_paths:
        docs = load_document(Path(fp))
        all_docs.extend(docs)

    if not all_docs:
        return {"status": "empty", "documents": 0, "chunks": 0}

    chunks = chunk_documents(all_docs)
    # Add to Chroma (ids auto-generated if not provided)
    ids = [c.metadata["chunk_id"] for c in chunks]
    vectorstore.add_documents(documents=chunks, ids=ids)

    return {
        "status": "ok",
        "documents": len(all_docs),
        "chunks": len(chunks),
        "sources": list({d.metadata.get("source", "unknown") for d in all_docs}),
    }


def ingest_data_directory(reset: bool = False) -> dict:
    """Ingest every supported file found in DATA_DIR."""
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)

    if reset:
        try:
            vectorstore.delete_collection()
        except Exception:
            pass
        vectorstore = get_vectorstore(embeddings)

    print(f"Scanning {DATA_DIR} ...")
    docs = load_directory(DATA_DIR)
    if not docs:
        print("No documents found.")
        return {"status": "empty", "documents": 0, "chunks": 0}

    print(f"Chunking {len(docs)} document(s) ...")
    chunks = chunk_documents(docs)
    ids = [c.metadata["chunk_id"] for c in chunks]
    vectorstore.add_documents(documents=chunks, ids=ids)
    print(f"Ingested {len(chunks)} chunks into ChromaDB at {CHROMA_DIR}")

    return {
        "status": "ok",
        "documents": len(docs),
        "chunks": len(chunks),
        "sources": list({d.metadata.get("source", "unknown") for d in docs}),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest TripSphere documents into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Wipe existing collection first")
    args = parser.parse_args()
    summary = ingest_data_directory(reset=args.reset)
    print(summary)
