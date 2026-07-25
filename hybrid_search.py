import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


class TechMartHybridSearch:
    """
    TechMart hybrid search engine.

    Combines:
    1. Semantic vector search using ChromaDB
    2. BM25 keyword search
    3. Metadata filtering

    The ChromaDB collection is created in memory, making it suitable
    for Streamlit Community Cloud.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        if data_path is None:
            data_path = os.path.join(
                base_dir,
                "data",
                "techmart_products.json"
            )

        self.data_path = data_path
        self.embedding_model = embedding_model

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Product dataset was not found at: {self.data_path}. "
                "Make sure data/techmart_products.json exists in GitHub."
            )

        self.products = self._load_products()

        if not self.products:
            raise ValueError(
                "The TechMart product dataset is empty."
            )

        self.documents = self._create_documents()

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"}
        )

        # In-memory ChromaDB: no persist_directory
        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            collection_name="techmart_products"
        )

        self.bm25_documents = [
            self._tokenize(document.page_content)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(self.bm25_documents)

    def _load_products(self) -> List[Dict[str, Any]]:
        """Load products from the JSON dataset."""

        with open(
            self.data_path,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        # Support either a direct list or {"products": [...]}
        if isinstance(data, dict):
            data = data.get("products", [])

        if not isinstance(data, list):
            raise ValueError(
                "techmart_products.json must contain a list of products."
            )

        cleaned_products = []

        for index, product in enumerate(data):
            if not isinstance(product, dict):
                continue

            cleaned_product = {
                "product_id": str(
                    product.get(
                        "product_id",
                        product.get("id", f"TM-{index + 1:03d}")
                    )
                ),
                "name": str(
                    product.get(
                        "name",
                        product.get("product_name", "Unnamed Product")
                    )
                ),
                "category": str(
                    product.get("category", "Uncategorized")
                ),
                "brand": str(
                    product.get("brand", "Unknown")
                ),
                "price": self._safe_float(
                    product.get("price", 0)
                ),
                "rating": self._safe_float(
                    product.get("rating", 0)
                ),
                "description": str(
                    product.get(
                        "description",
                        product.get("details", "")
                    )
                ),
                "stock": self._safe_int(
                    product.get(
                        "stock",
                        product.get("stock_quantity", 0)
                    )
                )
            }

            cleaned_products.append(cleaned_product)

        return cleaned_products

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Safely convert a value to float."""

        try:
            if isinstance(value, str):
                value = value.replace("$", "").replace(",", "").strip()

            return float(value)

        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_int(value: Any) -> int:
        """Safely convert a value to integer."""

        try:
            return int(float(value))

        except (TypeError, ValueError):
            return 0

    def _create_documents(self) -> List[Document]:
        """Convert product records into LangChain documents."""

        documents = []

        for product in self.products:
            content = (
                f"Product Name: {product['name']}\n"
                f"Category: {product['category']}\n"
                f"Brand: {product['brand']}\n"
                f"Price: ${product['price']:.2f}\n"
                f"Rating: {product['rating']}\n"
                f"Stock: {product['stock']}\n"
                f"Description: {product['description']}"
            )

            metadata = {
                "product_id": product["product_id"],
                "name": product["name"],
                "category": product["category"],
                "brand": product["brand"],
                "price": product["price"],
                "rating": product["rating"],
                "stock": product["stock"]
            }

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )

        return documents

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Convert text into lowercase keyword tokens."""

        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )

    @staticmethod
    def _normalize_scores(
        scores: List[float]
    ) -> List[float]:
        """Normalize scores to a range between 0 and 1."""

        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)

        if maximum == minimum:
            if maximum == 0:
                return [0.0 for _ in scores]

            return [1.0 for _ in scores]

        return [
            (score - minimum) / (maximum - minimum)
            for score in scores
        ]

    @staticmethod
    def _matches_filters(
        metadata: Dict[str, Any],
        category: Optional[str],
        brand: Optional[str],
        min_price: Optional[float],
        max_price: Optional[float],
        min_rating: Optional[float]
    ) -> bool:
        """Check whether a product satisfies the metadata filters."""

        if (
            category
            and category != "All"
            and metadata.get("category") != category
        ):
            return False

        if (
            brand
            and brand != "All"
            and metadata.get("brand") != brand
        ):
            return False

        price = float(metadata.get("price", 0))
        rating = float(metadata.get("rating", 0))

        if min_price is not None and price < min_price:
            return False

        if max_price is not None and price > max_price:
            return False

        if min_rating is not None and rating < min_rating:
            return False

        return True

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        top_k: int = 5,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using vector similarity and BM25.

        Returns products ranked by the combined hybrid score.
        """

        query = query.strip()

        if not query:
            return []

        vector_weight = max(0.0, min(1.0, vector_weight))
        keyword_weight = max(0.0, min(1.0, keyword_weight))

        total_weight = vector_weight + keyword_weight

        if total_weight == 0:
            vector_weight = 0.6
            keyword_weight = 0.4
            total_weight = 1.0

        vector_weight /= total_weight
        keyword_weight /= total_weight

        # -------------------------------------------------
        # 1. Semantic vector scores
        # -------------------------------------------------
        vector_results = self.vectorstore.similarity_search_with_score(
            query,
            k=len(self.documents)
        )

        vector_score_by_id = {}

        for document, distance in vector_results:
            product_id = document.metadata["product_id"]

            # Chroma returns distance; lower distance is better
            similarity = 1.0 / (1.0 + float(distance))

            vector_score_by_id[product_id] = similarity

        # -------------------------------------------------
        # 2. BM25 keyword scores
        # -------------------------------------------------
        query_tokens = self._tokenize(query)

        raw_keyword_scores = self.bm25.get_scores(
            query_tokens
        ).tolist()

        normalized_keyword_scores = self._normalize_scores(
            raw_keyword_scores
        )

        keyword_score_by_id = {}

        for index, score in enumerate(
            normalized_keyword_scores
        ):
            product_id = self.products[index]["product_id"]
            keyword_score_by_id[product_id] = score

        # -------------------------------------------------
        # 3. Combine and filter results
        # -------------------------------------------------
        results = []

        for product in self.products:
            product_id = product["product_id"]

            if not self._matches_filters(
                metadata=product,
                category=category,
                brand=brand,
                min_price=min_price,
                max_price=max_price,
                min_rating=min_rating
            ):
                continue

            vector_score = vector_score_by_id.get(
                product_id,
                0.0
            )

            keyword_score = keyword_score_by_id.get(
                product_id,
                0.0
            )

            hybrid_score = (
                vector_weight * vector_score
                + keyword_weight * keyword_score
            )

            result = product.copy()

            result.update(
                {
                    "vector_score": round(
                        vector_score,
                        4
                    ),
                    "keyword_score": round(
                        keyword_score,
                        4
                    ),
                    "hybrid_score": round(
                        hybrid_score,
                        4
                    )
                }
            )

            results.append(result)

        results.sort(
            key=lambda item: item["hybrid_score"],
            reverse=True
        )

        return results[:top_k]

    def get_categories(self) -> List[str]:
        """Return the available product categories."""

        categories = {
            product["category"]
            for product in self.products
        }

        return sorted(categories)

    def get_brands(self) -> List[str]:
        """Return the available product brands."""

        brands = {
            product["brand"]
            for product in self.products
        }

        return sorted(brands)

    def get_product_count(self) -> int:
        """Return the number of indexed products."""

        return len(self.products)

    def get_vector_count(self) -> int:
        """Return the number of vectors in ChromaDB."""

        return self.vectorstore._collection.count()