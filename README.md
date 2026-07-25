# TripSphere Multi-Document RAG System

**Lab 4** — A production-style Retrieval-Augmented Generation (RAG) application for a fictional smart travel & hospitality company **TripSphere**.

The system ingests **PDFs** (hotel policies), **CSVs** (bookings & pricing), and **TXT** files (travel FAQs), stores embeddings in a **persistent ChromaDB**, and answers questions via **Groq** through a clean **Streamlit** interface.

---

## Architecture

```
File Upload (Streamlit)
        │
        ▼
Document-type detection (pdf / csv / txt)
        │
        ▼
Specialized Loaders (PyPDFLoader, CSVLoader, TextLoader)
        │
        ▼
Metadata enrichment
  (source, document_type, page/row, category, chunk_id)
        │
        ▼
RecursiveCharacterTextSplitter (chunk_size=800, overlap=150)
        │
        ▼
Local HuggingFace Embeddings
  (sentence-transformers/all-MiniLM-L6-v2, 384-d)
        │
        ▼
Persistent ChromaDB (chroma_db/)
        │
        ▼
Similarity retrieval (top-k = 3–5, optional type filter)
        │
        ▼
Concise prompt → Groq LLM → Grounded answer + evidence
```

**Groq free-tier optimizations**
- Never call Groq during ingestion
- Retrieve only top 3–5 chunks
- Short system prompt + `max_tokens=512`
- `@st.cache_resource` for embeddings & vectorstore
- Session-state caching of last answer

---

## Project Structure

```
tripsphere-rag/
├── requirements.txt
├── .env.example
├── .gitignore
├── config.py                 # paths, models, chunk settings
├── loaders.py                # PDF / CSV / TXT loaders + metadata
├── ingest.py                 # chunk → embed → Chroma (no Groq)
├── rag_pipeline.py           # retrieve + Groq answer generation
├── inspect_db.py             # terminal inspector for Chroma
├── app.py                    # Streamlit UI
├── generate_sample_data.py   # creates demo PDF, CSV, TXT
├── report_generator.py       # builds PDF project report
├── README.md
├── data/                     # place or generate documents here
├── chroma_db/                # persistent vector store (gitignored)
└── reports/                  # generated PDF reports
```

---

## Quick Start (Local)

### 1. Clone & environment

```bash
git clone https://github.com/<your-username>/tripsphere-rag.git
cd tripsphere-rag
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API key

```bash
cp .env.example .env
# Edit .env and set:
# GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Sample data & ingest

```bash
python generate_sample_data.py
python ingest.py --reset
python inspect_db.py
```

Expected `inspect_db.py` output (approximate):

```
============================================================
TripSphere ChromaDB Inspector
============================================================
Persist directory : .../chroma_db
Collection name   : tripsphere_docs

Total vectors     : ~25–40
Embedding model   : sentence-transformers/all-MiniLM-L6-v2
Embedding dim     : 384
Document types    : {'pdf': N, 'csv': 12, 'txt': M}
Sources           : ['tripsphere_bookings_pricing.csv', 'tripsphere_hotel_policies.pdf', 'tripsphere_travel_faqs.txt']
```

### 4. Run the app

```bash
streamlit run app.py
```

Open the URL shown (usually http://localhost:8501).

---

## Streamlit UI Features

| Feature | Description |
|---------|-------------|
| Upload | Multi-file PDF / CSV / TXT |
| Index | Ingest with optional full reset |
| Ask | Question box + document-type filter + top-k slider |
| Answer | Grounded response + source filenames |
| Evidence | Expandable retrieved passages with metadata |
| Stats | Live collection size, types, sources, embedding dim |

---

## Test Queries

1. *What is the cancellation policy for flexible rates?*
2. *How much does a Deluxe Suite cost at Sphere Grand Downtown in New York?*
3. *Are pets allowed? What is the fee?*
4. *How do I redeem Sphere Points for free nights?*
5. *What time is check-out?*
6. *Does Sphere Airport Hub in Chicago have a shuttle?*

---

## Generate Project Report (PDF)

```bash
python report_generator.py
# → reports/TripSphere_RAG_Lab_Report.pdf
```

The report contains architecture, setup, sample queries, vector-store stats, deployment steps, and screenshot placeholders.

---

## GitHub & Streamlit Community Cloud

### Push to GitHub

```bash
git init
git add .
git commit -m "Initial TripSphere Multi-Document RAG"
git branch -M main
git remote add origin https://github.com/<your-username>/tripsphere-rag.git
git push -u origin main
```

### Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Select the repository, branch `main`, main file path `app.py`
3. Under **Advanced settings → Secrets** add:

   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```

4. Click **Deploy**

**Notes for Cloud**
- `chroma_db/` is gitignored → the store starts empty. Users upload & index via the UI, or you can add a one-time ingest on startup.
- First cold start downloads the MiniLM model (~80 MB); subsequent runs are faster.
- Free tier has memory limits; MiniLM-L6-v2 is intentionally lightweight.

---

## Validation Checklist

- [ ] `python generate_sample_data.py` creates 3 files in `data/`
- [ ] `python ingest.py --reset` finishes without error
- [ ] `python inspect_db.py` shows vectors > 0 and correct sources
- [ ] `streamlit run app.py` loads; stats panel shows collection size
- [ ] Asking a policy question returns an answer that cites the PDF
- [ ] Asking a price question returns an answer that cites the CSV
- [ ] Filter by `csv` excludes PDF/TXT chunks
- [ ] `python report_generator.py` produces a PDF in `reports/`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `GROQ_API_KEY is not set` | Add key to `.env` or Streamlit secrets |
| 429 / rate limit from Groq | Lower top-k, wait 60 s, or switch to `llama-3.1-8b-instant` |
| Empty retrieval | Re-run `ingest.py --reset`; check filter is not too strict |
| Embedding / torch errors | `pip install torch sentence-transformers --upgrade` |
| Chroma lock / permission | Delete `chroma_db/` and re-ingest |
| Streamlit Cloud OOM | Confirm you are using MiniLM-L6-v2 (not a larger model) |

---

## Credit-Efficient Prompts (for AI assistants)

**Groq-assisted development**
```
You are helping debug a LangChain + Chroma + Groq RAG app.
Only suggest code changes that reduce API calls.
Never embed or call the LLM inside the ingest path.
Keep prompts under 300 tokens and max_tokens <= 512.
```

**GitHub Copilot**
```
# TripSphere RAG — follow existing patterns in loaders.py / ingest.py.
# Use langchain_huggingface.HuggingFaceEmbeddings and langchain_chroma.Chroma.
# Preserve metadata: source, document_type, page/row, chunk_id.
```

**Antigravity / general**
```
Act as a senior RAG engineer. Prefer local embeddings, persistent Chroma,
and Groq only at query time. Produce minimal, working diffs only.
```

---

## License

MIT — educational use for Lab 4.
