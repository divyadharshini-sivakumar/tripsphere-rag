import os
import json
import shutil

from config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    FAQ_JSON_PATH,
    validate_config
)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def run_ingestion():
    """
    Loads the QuickBite FAQ JSON file, creates text chunks,
    generates embeddings and stores them in ChromaDB.
    """

    print("[*] Validating environment configuration...")

    validate_config(require_api_key=False)

    # Check whether the FAQ JSON file exists
    if not os.path.exists(FAQ_JSON_PATH):
        raise FileNotFoundError(
            f"FAQ data file was not found at: {FAQ_JSON_PATH}. "
            "Make sure data/quickbite_faq.json is uploaded to GitHub."
        )

    # -----------------------------------------------------
    # 1. Load JSON Dataset
    # -----------------------------------------------------
    print(
        f"[*] Loading FAQ entries from '{FAQ_JSON_PATH}'..."
    )

    with open(
        FAQ_JSON_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        faq_data = json.load(file)

    if not isinstance(faq_data, list):
        raise ValueError(
            "The FAQ JSON file must contain a list of FAQ objects."
        )

    if len(faq_data) == 0:
        raise ValueError(
            "The FAQ JSON file is empty."
        )

    # -----------------------------------------------------
    # 2. Create LangChain Documents
    # -----------------------------------------------------
    raw_documents = []

    for item in faq_data:
        required_fields = [
            "faq_id",
            "category",
            "question",
            "answer"
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in item
        ]

        if missing_fields:
            raise ValueError(
                f"FAQ item is missing fields: {missing_fields}"
            )

        content = (
            f"Category: {item['category']}\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}"
        )

        metadata = {
            "faq_id": item["faq_id"],
            "category": item["category"],
            "question": item["question"],
            "source": "quickbite_faq.json"
        }

        document = Document(
            page_content=content,
            metadata=metadata
        )

        raw_documents.append(document)

    print(
        f"[+] Total raw documents loaded: "
        f"{len(raw_documents)}"
    )

    # -----------------------------------------------------
    # 3. Split Documents into Chunks
    # -----------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(
        raw_documents
    )

    for index, chunk in enumerate(chunks):
        faq_id = chunk.metadata.get(
            "faq_id",
            "UNK"
        )

        chunk.metadata["chunk_id"] = (
            f"{faq_id}_chunk_{index + 1}"
        )

    print(
        f"[+] Total text chunks ready for indexing: "
        f"{len(chunks)}"
    )

    # -----------------------------------------------------
    # 4. Load HuggingFace Embeddings
    # -----------------------------------------------------
    print(
        f"[*] Loading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        }
    )

    # -----------------------------------------------------
    # 5. Reset Existing ChromaDB Directory
    # -----------------------------------------------------
    if os.path.exists(CHROMA_PERSIST_DIR):
        try:
            shutil.rmtree(
                CHROMA_PERSIST_DIR
            )

            print(
                "[*] Existing ChromaDB directory removed."
            )

        except Exception as error:
            print(
                f"[!] Warning while removing old database: "
                f"{error}"
            )

    os.makedirs(
        CHROMA_PERSIST_DIR,
        exist_ok=True
    )

    # -----------------------------------------------------
    # 6. Create and Store Embeddings in ChromaDB
    # -----------------------------------------------------
    print(
        f"[*] Indexing {len(chunks)} chunks into "
        f"collection '{COLLECTION_NAME}'..."
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR
    )

    final_count = vectorstore._collection.count()

    print("\n" + "=" * 70)
    print("SUCCESS: INGESTION COMPLETE")
    print(f"Collection Name: {COLLECTION_NAME}")
    print(f"Persist Directory: {CHROMA_PERSIST_DIR}")
    print(f"Raw Documents: {len(raw_documents)}")
    print(f"Chunks Indexed: {len(chunks)}")
    print(f"Final Vector Count: {final_count}")
    print("=" * 70 + "\n")

    if final_count == 0:
        raise RuntimeError(
            "Ingestion completed, but no vectors were indexed."
        )

    return final_count


if __name__ == "__main__":
    run_ingestion()