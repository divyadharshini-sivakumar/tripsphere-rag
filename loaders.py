"""
TripSphere Multi-Document RAG — Document Loaders
Handles PDF, CSV, and plain-text files with rich metadata.
"""
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
)

from config import SUPPORTED_EXTENSIONS


def detect_document_type(file_path: Path) -> str:
    """Return a human-readable document type from file extension."""
    ext = file_path.suffix.lower()
    mapping = {
        ".pdf": "pdf",
        ".csv": "csv",
        ".txt": "txt",
    }
    return mapping.get(ext, "unknown")


def load_pdf(file_path: Path) -> List[Document]:
    """Load a PDF and attach source + page metadata."""
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    for i, doc in enumerate(docs):
        # Overwrite any absolute path that PyPDFLoader may have set
        doc.metadata["source"] = file_path.name
        doc.metadata["document_type"] = "pdf"
        doc.metadata["page"] = doc.metadata.get("page", i)
        doc.metadata["category"] = "hotel_policy"
    return docs


def load_csv(file_path: Path) -> List[Document]:
    """
    Load a CSV; each row becomes a Document.
    Metadata includes row index and source.
    """
    loader = CSVLoader(
        file_path=str(file_path),
        encoding="utf-8",
        csv_args={"delimiter": ","},
    )
    docs = loader.load()
    for i, doc in enumerate(docs):
        doc.metadata["source"] = file_path.name
        doc.metadata["document_type"] = "csv"
        doc.metadata["row"] = i
        doc.metadata["category"] = "booking_pricing"
        # CSVLoader may put source as full path
        if "source" in doc.metadata and str(file_path) in str(doc.metadata["source"]):
            doc.metadata["source"] = file_path.name
    return docs


def load_txt(file_path: Path) -> List[Document]:
    """Load a plain-text FAQ file."""
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = file_path.name
        doc.metadata["document_type"] = "txt"
        doc.metadata["category"] = "travel_faq"
        doc.metadata["page"] = 0
    return docs


def load_document(file_path: Path) -> List[Document]:
    """
    Dispatch to the correct loader based on extension.
    Raises ValueError for unsupported types.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    if ext == ".pdf":
        return load_pdf(file_path)
    if ext == ".csv":
        return load_csv(file_path)
    if ext == ".txt":
        return load_txt(file_path)

    raise ValueError(f"No loader implemented for {ext}")


def load_directory(directory: Path) -> List[Document]:
    """Load every supported file in a directory (non-recursive)."""
    directory = Path(directory)
    all_docs: List[Document] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                docs = load_document(path)
                all_docs.extend(docs)
                print(f"  ✓ Loaded {path.name} → {len(docs)} document(s)")
            except Exception as e:
                print(f"  ✗ Failed to load {path.name}: {e}")
    return all_docs
