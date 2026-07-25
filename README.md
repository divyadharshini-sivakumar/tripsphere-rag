# TechMart Hybrid Search System

## Overview

TechMart Hybrid Search is a production-style electronics product search system that combines:

- Semantic vector search
- BM25 keyword search
- Weighted hybrid score fusion
- Metadata filtering
- Persistent ChromaDB storage
- Streamlit web interface
- ChromaDB inspection
- PDF report generation

The application searches a realistic dataset of 60 electronics products across categories such as laptops, smartphones, tablets, gaming products, accessories, and smart-home devices.

---

## Features

- 60 realistic TechMart product records
- Local HuggingFace embeddings
- Persistent ChromaDB vector database
- BM25 keyword retrieval
- Configurable vector and keyword weights
- Metadata filters for:
  - Category
  - Brand
  - Price range
  - Minimum rating
  - Availability
- Ranked hybrid search results
- Streamlit web application
- ChromaDB database inspector
- Automatic PDF report generation

---

## Project Structure

```text
techmart_hybrid_search/
│
├── app.py
├── config.py
├── generate_data.py
├── ingest.py
├── hybrid_search.py
├── inspect_db.py
├── report_generator.py
├── requirements.txt
├── .env.example
├── README.md
│
├── data/
│   └── techmart_products.json
│
├── chroma_db/
│
└── reports/
    └── techmart_hybrid_search_report.pdf