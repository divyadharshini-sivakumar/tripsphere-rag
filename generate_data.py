import os
import sys
from dotenv import load_dotenv

# Ensure user site-packages are accessible on Windows environments
user_site = os.path.expanduser(r"~\AppData\Roaming\Python\Python310\site-packages")
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

# Load environment variables from .env file
load_dotenv()

# Read API key from Streamlit Secrets (Cloud) or .env (Local)
try:
    import streamlit as st

    OPENAI_API_KEY = st.secrets.get(
        "OPENAI_API_KEY",
        os.getenv("OPENAI_API_KEY")
    )

    OPENAI_API_BASE = st.secrets.get(
        "OPENAI_API_BASE",
        os.getenv("OPENAI_API_BASE")
    )

except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

# Auto-detect OpenRouter API keys starting with sk-or-v1-
if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-or-v1-"):
    if not OPENAI_API_BASE:
        OPENAI_API_BASE = "https://openrouter.ai/api/v1"
    DEFAULT_LLM = "openai/gpt-4o-mini"
else:
    DEFAULT_LLM = "gpt-4o-mini"

LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_LLM)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Directory & Collection Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    "chroma_db"
)

if not os.path.isabs(CHROMA_PERSIST_DIR):
    CHROMA_PERSIST_DIR = os.path.join(
        BASE_DIR,
        CHROMA_PERSIST_DIR
    )

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "quickbite_faqs"
)

DATA_DIR = os.path.join(BASE_DIR, "data")
FAQ_JSON_PATH = os.path.join(DATA_DIR, "quickbite_faq.json")
FAQ_TXT_PATH = os.path.join(DATA_DIR, "quickbite_faq.txt")


def validate_config(require_api_key: bool = False):
    """Validates directory setup."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return True