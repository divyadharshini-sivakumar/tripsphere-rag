"""
TripSphere Multi-Document RAG — Vector DB Inspector
Run from terminal to inspect ChromaDB contents without Streamlit.
"""
from rag_pipeline import get_vectorstore, get_collection_stats
from config import CHROMA_DIR, COLLECTION_NAME


def main():
    print("=" * 60)
    print("TripSphere ChromaDB Inspector")
    print("=" * 60)
    print(f"Persist directory : {CHROMA_DIR}")
    print(f"Collection name   : {COLLECTION_NAME}")
    print()

    stats = get_collection_stats()
    if "error" in stats:
        print(f"Error reading collection: {stats['error']}")
        print("Have you run ingest.py yet?")
        return

    print(f"Total vectors     : {stats['count']}")
    print(f"Embedding model   : {stats['embedding_model']}")
    print(f"Embedding dim     : {stats['embedding_dimension']}")
    print(f"Document types    : {stats['document_types']}")
    print(f"Sources           : {stats['sources']}")
    print()

    if stats["count"] == 0:
        print("Collection is empty.")
        return

    vs = get_vectorstore()
    data = vs.get(include=["metadatas", "documents"], limit=5)
    print("--- Sample of first 5 vectors ---")
    for i, (vid, meta, doc) in enumerate(
        zip(data["ids"], data["metadatas"], data["documents"])
    ):
        print(f"\n[{i+1}] id={vid}")
        print(f"    metadata={meta}")
        preview = (doc or "")[:120].replace("\n", " ")
        print(f"    text={preview}...")


if __name__ == "__main__":
    main()
