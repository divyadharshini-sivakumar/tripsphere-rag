"""
TripSphere Multi-Document RAG — Configuration
Loads environment variables and defines project paths.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Load values from .env
load_dotenv(BASE_DIR / ".env")


# Folder paths
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
REPORTS_DIR = BASE_DIR / "reports"


# Supported document types
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".csv",
    ".txt",
}


# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "",
).strip()

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.1-8b-instruct:free",
).strip()


# Embedding configuration
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
).strip()


# Chunking configuration
CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "800")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "150")
)


# Retrieval configuration
TOP_K = int(
    os.getenv("TOP_K", "4")
)


# ChromaDB collection
COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "tripsphere_documents",
).strip()


# Create folders if they do not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)