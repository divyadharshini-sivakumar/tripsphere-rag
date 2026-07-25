"""
TripSphere Multi-Document RAG — Query Pipeline

Retrieves relevant chunks from ChromaDB and generates answers
using an OpenRouter model through the OpenAI-compatible API.
"""

from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    TOP_K,
)


SYSTEM_PROMPT = """
You are TripSphere Assistant, a helpful travel and hospitality expert.

Answer only using the provided context.

If the answer is not available in the context, respond with:
"I don't have that information in the TripSphere documents."

Keep the answer concise, with a maximum of 4 to 5 sentences.
Mention the source filename when relevant.
"""


HUMAN_PROMPT = """
Context:
{context}

Question:
{question}

Answer:
"""


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the Hugging Face embedding model used during ingestion
    and retrieval.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(
    embeddings: Optional[HuggingFaceEmbeddings] = None,
) -> Chroma:
    """
    Connect to the existing persistent ChromaDB vector store.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def get_llm() -> ChatOpenAI:
    """
    Create the OpenRouter LLM connection.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. "
            "Add your OpenRouter key to the .env file."
        )

    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=512,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-OpenRouter-Title": "TripSphere Multi-Document RAG",
        },
    )


def format_docs(docs: List[Document]) -> str:
    """
    Convert retrieved documents into formatted context for the LLM.
    """
    parts = []

    for index, document in enumerate(docs, start=1):
        source = document.metadata.get("source", "unknown")
        document_type = document.metadata.get("document_type", "unknown")

        location = document.metadata.get(
            "page",
            document.metadata.get("row", ""),
        )

        content = document.page_content.strip()

        parts.append(
            f"[{index}] "
            f"(source={source}, type={document_type}, location={location})\n"
            f"{content}"
        )

    return "\n\n".join(parts)


def retrieve(
    query: str,
    k: int = TOP_K,
    filter_type: Optional[str] = None,
) -> List[Document]:
    """
    Retrieve the most relevant chunks from ChromaDB.

    A document-type filter can optionally be applied for:
    pdf, csv or txt.
    """
    vectorstore = get_vectorstore()

    search_kwargs: Dict[str, Any] = {
        "k": k,
    }

    if filter_type and filter_type.lower() != "all":
        search_kwargs["filter"] = {
            "document_type": filter_type.lower()
        }

    return vectorstore.similarity_search(
        query,
        **search_kwargs,
    )


def answer_question(
    question: str,
    k: int = TOP_K,
    filter_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete RAG workflow:

    1. Retrieve documents from ChromaDB.
    2. Format the retrieved context.
    3. Send the context and question to OpenRouter.
    4. Return the answer and evidence.
    """
    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "documents": [],
        }

    documents = retrieve(
        query=question,
        k=k,
        filter_type=filter_type,
    )

    if not documents:
        return {
            "answer": (
                "No relevant documents were found in the vector store. "
                "Please run the ingestion script first."
            ),
            "sources": [],
            "documents": [],
        }

    context = format_docs(documents)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )

    llm = get_llm()
    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    answer = response.content.strip()

    sources = sorted(
        {
            document.metadata.get("source", "unknown")
            for document in documents
        }
    )

    return {
        "answer": answer,
        "sources": sources,
        "documents": documents,
    }


def get_collection_stats() -> Dict[str, Any]:
    """
    Return information about the ChromaDB collection.
    """
    vectorstore = get_vectorstore()

    try:
        data = vectorstore.get(
            include=["metadatas"]
        )

        ids = data.get("ids", [])
        metadatas = data.get("metadatas", [])

        document_types: Dict[str, int] = {}
        sources = set()

        for metadata in metadatas or []:
            if not metadata:
                continue

            document_type = metadata.get(
                "document_type",
                "unknown",
            )

            document_types[document_type] = (
                document_types.get(document_type, 0) + 1
            )

            source = metadata.get("source", "unknown")
            sources.add(source)

        return {
            "collection_name": COLLECTION_NAME,
            "count": len(ids),
            "document_types": document_types,
            "sources": sorted(sources),
            "embedding_dimension": 384,
            "embedding_model": EMBEDDING_MODEL,
        }

    except Exception as error:
        return {
            "error": str(error),
            "count": 0,
        }