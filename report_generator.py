"""
Generate a PDF project report for the TripSphere Multi-Document RAG lab.
"""
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    ListFlowable,
    ListItem,
)

from config import REPORTS_DIR, COLLECTION_NAME, EMBEDDING_MODEL, GROQ_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from rag_pipeline import get_collection_stats

REPORTS_DIR.mkdir(exist_ok=True)


def build_report(output_name: str = "TripSphere_RAG_Lab_Report.pdf"):
    path = REPORTS_DIR / output_name
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "T", parent=styles["Heading1"], fontSize=18, spaceAfter=6, textColor=colors.HexColor("#1a365d")
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontSize=14, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#2c5282")
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=11, spaceBefore=10, spaceAfter=4
    )
    body = ParagraphStyle("B", parent=styles["BodyText"], fontSize=10, leading=13)
    code = ParagraphStyle(
        "C", parent=styles["Code"], fontSize=8, leading=10, backColor=colors.HexColor("#f7fafc")
    )

    story = []

    # Cover
    story.append(Paragraph("TripSphere Multi-Document RAG System", title))
    story.append(Paragraph("Lab 4 Project Report", h1))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body))
    story.append(Paragraph("Company concept: TripSphere — smart travel & hospitality", body))
    story.append(Spacer(1, 0.25 * inch))

    # 1. Architecture
    story.append(Paragraph("1. System Architecture", h1))
    story.append(
        Paragraph(
            "The pipeline follows a classic multi-document RAG flow optimized for free-tier usage:",
            body,
        )
    )
    steps = [
        "File upload (Streamlit) → document-type detection (PDF / CSV / TXT)",
        "Loading via specialized LangChain loaders with rich metadata",
        "Metadata assignment (source, document_type, page/row, category, chunk_id)",
        "Recursive character chunking (size={}, overlap={})".format(CHUNK_SIZE, CHUNK_OVERLAP),
        "Local HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2, 384-d)",
        "Persistent ChromaDB storage (collection: {})".format(COLLECTION_NAME),
        "Similarity retrieval (top-k={}) with optional document-type filter".format(TOP_K),
        "Concise prompt → Groq LLM (model: {}) → grounded answer".format(GROQ_MODEL),
    ]
    for s in steps:
        story.append(Paragraph(f"• {s}", body))

    story.append(Paragraph("Key design choices for Groq free-tier efficiency:", h2))
    story.append(Paragraph("• Retrieve only 3–5 chunks; never embed or call Groq during ingestion.", body))
    story.append(Paragraph("• Short system prompt + max_tokens=512.", body))
    story.append(Paragraph("• Streamlit @st.cache_resource for embeddings and vectorstore.", body))
    story.append(Paragraph("• No repeated API calls for unchanged questions (session state).", body))

    # 2. Project Structure
    story.append(Paragraph("2. Project Structure", h1))
    structure = """
tripsphere-rag/
├── requirements.txt
├── .env.example / .env
├── .gitignore
├── config.py
├── loaders.py
├── ingest.py
├── rag_pipeline.py
├── inspect_db.py
├── app.py                  # Streamlit UI
├── generate_sample_data.py
├── report_generator.py
├── README.md
├── data/                   # sample PDFs, CSVs, TXTs
├── chroma_db/              # persistent vector store
└── reports/                # generated PDF reports
"""
    story.append(Paragraph(f"<font face='Courier' size='8'>{structure.replace(chr(10), '<br/>')}</font>", body))

    # 3. Setup
    story.append(Paragraph("3. Setup & Installation", h1))
    story.append(Paragraph("1. Create a virtual environment and install dependencies:", body))
    story.append(Paragraph("python -m venv .venv && source .venv/bin/activate", code))
    story.append(Paragraph("pip install -r requirements.txt", code))
    story.append(Paragraph("2. Copy .env.example → .env and add your GROQ_API_KEY.", body))
    story.append(Paragraph("3. Generate sample data: python generate_sample_data.py", body))
    story.append(Paragraph("4. Ingest: python ingest.py --reset", body))
    story.append(Paragraph("5. Launch UI: streamlit run app.py", body))

    # 4. Sample output / test queries
    story.append(Paragraph("4. Sample Test Queries & Expected Behavior", h1))
    queries = [
        ("What is the cancellation policy for flexible rates?", "Should cite hotel_policies PDF — free cancel up to 24h."),
        ("How much is a Deluxe Suite at Sphere Grand Downtown?", "Should cite CSV pricing row (~$349)."),
        ("Are pets allowed and what is the fee?", "Should cite pet policy section — $50 non-refundable."),
        ("How do I redeem Sphere Points for free nights?", "Should cite FAQs TXT — Gold/Platinum, from 8k points."),
        ("What is the check-out time?", "Should cite PDF — 11:00 AM standard."),
    ]
    data = [["Question", "Expected evidence"]] + [[q, e] for q, e in queries]
    t = Table(data, colWidths=[3.4 * inch, 3.4 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5282")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t)

    # 5. Vector DB stats
    story.append(Paragraph("5. Vector Database Statistics", h1))
    try:
        stats = get_collection_stats()
        if stats.get("count", 0) > 0:
            story.append(Paragraph(f"Collection: {stats.get('collection_name')}", body))
            story.append(Paragraph(f"Total vectors: {stats.get('count')}", body))
            story.append(Paragraph(f"Embedding model: {stats.get('embedding_model')} ({stats.get('embedding_dimension')}d)", body))
            story.append(Paragraph(f"Document types: {stats.get('document_types')}", body))
            story.append(Paragraph(f"Sources: {stats.get('sources')}", body))
        else:
            story.append(
                Paragraph(
                    "Collection empty or not yet ingested. Run python ingest.py --reset then re-generate this report.",
                    body,
                )
            )
    except Exception as e:
        story.append(
            Paragraph(
                f"Could not read vector store (model may need download on first run): {e}",
                body,
            )
        )

    # 6. Deployment
    story.append(Paragraph("6. GitHub & Streamlit Community Cloud Deployment", h1))
    story.append(Paragraph("Git commands:", h2))
    story.append(Paragraph("git init", code))
    story.append(Paragraph("git add .", code))
    story.append(Paragraph('git commit -m "Initial TripSphere Multi-Document RAG"', code))
    story.append(Paragraph("git branch -M main", code))
    story.append(Paragraph("git remote add origin https://github.com/<user>/tripsphere-rag.git", code))
    story.append(Paragraph("git push -u origin main", code))
    story.append(Paragraph("Streamlit Community Cloud:", h2))
    story.append(Paragraph("1. Push repo to GitHub (public or private).", body))
    story.append(Paragraph("2. Go to share.streamlit.io → New app → select repo, branch main, main file app.py.", body))
    story.append(Paragraph("3. In Advanced settings → Secrets, add:", body))
    story.append(Paragraph('GROQ_API_KEY = "gsk_..."', code))
    story.append(Paragraph("4. Deploy. Note: first run will download the embedding model (cold start).", body))
    story.append(
        Paragraph(
            "Important: chroma_db/ is gitignored. On Cloud the vector store is ephemeral unless you "
            "re-ingest on startup or use a remote vector DB. For the lab, users can upload & index via the UI.",
            body,
        )
    )

    # 7. Screenshot placeholders
    story.append(Paragraph("7. Screenshot Placeholders", h1))
    story.append(Paragraph("[Screenshot 1: Streamlit home page with sidebar upload]", body))
    story.append(Paragraph("[Screenshot 2: Successful ingestion metrics]", body))
    story.append(Paragraph("[Screenshot 3: Sample Q&A with retrieved passages expander]", body))
    story.append(Paragraph("[Screenshot 4: Collection stats panel]", body))

    # 8. Validation & Troubleshooting
    story.append(Paragraph("8. Validation & Troubleshooting", h1))
    story.append(Paragraph("• python inspect_db.py — should list vectors and sample metadata.", body))
    story.append(Paragraph("• If Groq returns 429: lower TOP_K, wait, or switch model.", body))
    story.append(Paragraph("• If embeddings fail: ensure torch / sentence-transformers installed; CPU is fine.", body))
    story.append(Paragraph("• Empty answers: re-ingest with --reset; check filter_type is not excluding docs.", body))
    story.append(Paragraph("• Streamlit Cloud memory: MiniLM-L6-v2 is lightweight (~80 MB).", body))

    doc.build(story)
    print(f"Report written to {path}")
    return path


if __name__ == "__main__":
    build_report()
