"""
Database inspection script for TechMart Hybrid Search System.
Connects to persistent ChromaDB storage and displays collection statistics,
sample document contents, metadata, vector IDs, and embedding dimensionality.
"""

import logging
import sys
import chromadb
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def inspect_database():
    """
    Connects to persistent ChromaDB and inspects collection information.
    """
    logging.info(f"Connecting to ChromaDB at: '{config.CHROMA_PERSIST_DIR}'...")
    
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
    except Exception as e:
        print(f"\n[ERROR] Failed to connect to ChromaDB at '{config.CHROMA_PERSIST_DIR}': {e}")
        sys.exit(1)

    existing_collections = [col.name for col in client.list_collections()]
    
    if config.CHROMA_COLLECTION_NAME not in existing_collections:
        print("\n" + "=" * 60)
        print("CHROMADB INSPECTION SUMMARY")
        print("=" * 60)
        print(f"Target Collection:   '{config.CHROMA_COLLECTION_NAME}'")
        print(f"Status:              NOT FOUND / MISSING")
        print(f"Available Collections: {existing_collections if existing_collections else 'None'}")
        print(f"Persistence Path:    {config.CHROMA_PERSIST_DIR.resolve()}")
        print("=" * 60)
        print("Tip: Run 'python ingest.py' first to populate the vector store.\n")
        return

    try:
        collection = client.get_collection(config.CHROMA_COLLECTION_NAME)
        total_count = collection.count()

        # Fetch first 5 records including embeddings to verify dimensionality
        limit = min(5, total_count)
        data = collection.get(limit=limit, include=["documents", "metadatas", "embeddings"])

        ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        embeddings = data.get("embeddings", None)

        embedding_dim = "N/A"
        if embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None:
            embedding_dim = len(embeddings[0])

        print("\n" + "=" * 60)
        print("CHROMADB COLLECTION INSPECTION SUMMARY")
        print("=" * 60)
        print(f"Collection Name:     {config.CHROMA_COLLECTION_NAME}")
        print(f"Persistence Path:    {config.CHROMA_PERSIST_DIR.resolve()}")
        print(f"Total Documents:     {total_count}")
        print(f"Embedding Dimension: {embedding_dim}")
        print("=" * 60)

        print(f"\n--- FIRST {len(ids)} VECTOR IDs ---")
        for idx, doc_id in enumerate(ids, 1):
            print(f" {idx}. {doc_id}")

        print(f"\n--- FIRST {len(metadatas)} METADATA RECORDS ---")
        for idx, meta in enumerate(metadatas, 1):
            print(f" [{idx}] ID: {meta.get('product_id', 'N/A')} | Name: {meta.get('name', 'N/A')}")
            print(f"     Category: {meta.get('category', 'N/A')} | Brand: {meta.get('brand', 'N/A')}")
            print(f"     Price: ${meta.get('price', 0.0):.2f} | Rating: {meta.get('rating', 0.0)} | Available: {meta.get('availability', False)}")

        print(f"\n--- FIRST {len(documents)} DOCUMENTS (PAGE CONTENT SNIPPETS) ---")
        for idx, doc in enumerate(documents, 1):
            lines = doc.strip().split("\n")
            preview = " | ".join(lines[:3])
            print(f" [{idx}] {preview[:120]}...")

        print("\n" + "=" * 60 + "\n")

    except Exception as e:
        logging.error(f"Error inspecting collection '{config.CHROMA_COLLECTION_NAME}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    inspect_database()
