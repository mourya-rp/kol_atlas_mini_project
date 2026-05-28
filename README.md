# 🔬 KOL Atlas: Entity Extraction & Analysis Dashboard

An automated, end-to-end intelligence pipeline designed to extract, score, and semantically analyze medical Key Opinion Leader (KOL) profiles. This application parses profile data, engineers a custom influence metric, computes semantic research alignment via local sentence embeddings, and generates comparative AI summaries.

##  Key Features & Add-ons
*   **Structured Parsing & Validation:** Processes raw HTML profiles into strictly validated JSON schemas using **Pydantic**, embedding data extraction confidence scores.
*   **Algorithmic Scoring Engine:** Computes an objective, multi-weighted **Influence Score** based on citations, h-index, and publication keyword density.
*   **Semantic Similarity Matrix:** Utilizes **SentenceTransformers** (`all-MiniLM-L6-v2`) to generate vector embeddings of research focus areas, calculating a cosine similarity matrix mapped visually via **Seaborn**.
*   **Interactive Streamlit Dashboard:** A production-ready frontend showroom that displays detailed profile metrics, the visual alignment matrix, and cross-profile insights.
*   **Live Gemini LLM Integration:** Connected securely via environment configurations to Google's **Gemini API** for automated, high-fidelity comparative research summaries.

---

## 🛠️ Tech Stack & Architecture
*   **Frontend:** Streamlit
*   **Data Validation:** Pydantic
*   **AI/Embeddings:** SentenceTransformers, Scikit-learn
*   **LLM Orchestration:** Google GenAI SDK (Gemini 3.5 Free Tier)
*   **Data Vis & Processing:** Seaborn, Matplotlib, Pandas, BeautifulSoup4

---

## 📁 Project Structure
```text
├── data/                  # Local target HTML profile mocks
├── .env                   # Secure environment secrets (API Keys)
├── .gitignore             # Git exclusion mapping (ignores venv, .env, caches)
├── app.py                 # Interactive Streamlit dashboard frontend
├── engine.py              # Extraction, validation, and matrix generation logic
├── main.py                # Core backend runner pipeline script
├── kol_profiles.json      # Structured database deliverable (Generated)
├── similarity_matrix.png  # High-res visual matrix heatmap (Generated)
└── requirements.txt       # Unified environment dependencies
