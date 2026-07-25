"""
Configuration module for TechMart Hybrid Search System.
Handles environment loading, path setup, model settings, and default parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# File & Directory Paths
DATASET_PATH = BASE_DIR / Path(os.getenv("DATASET_PATH", "data/techmart_products.json"))
CHROMA_PERSIST_DIR = BASE_DIR / Path(os.getenv("CHROMA_PERSIST_DIR", "chroma_db"))
REPORTS_DIR = BASE_DIR / Path(os.getenv("REPORTS_DIR", "reports"))

# Vector Database Configuration
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "techmart_products")

# Embedding Model Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Hybrid Search Default Parameters
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_ALPHA = float(os.getenv("DEFAULT_ALPHA", "0.5"))  # 1.0 = Pure Vector Search, 0.0 = Pure BM25 Keyword Search

# Product Categories
CATEGORIES = [
    "Laptops & Computers",
    "Smartphones & Tablets",
    "Audio & Headphones",
    "Wearables & Smartwatches",
    "Smart Home & IoT",
    "Gaming & Accessories",
    "Cameras & Photography",
    "Monitors & Displays"
]

# Supported Brands
BRANDS = [
    "Apple", "Samsung", "Sony", "Dell", "HP", "Lenovo", "Asus",
    "Logitech", "Bose", "Anker", "Canon", "LG", "Corsair", "Google",
    "Garmin", "DJI", "GoPro", "Philips"
]

# Ensure required runtime directories exist
DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
