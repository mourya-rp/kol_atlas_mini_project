#  KOL Atlas Mini-Project

A Streamlit dashboard that extracts Key Opinion Leader (KOL) medical profiles, calculates their influence, and uses AI to compare their research focus. 

![Dashboard Preview](dashboard_preview.png)<img width="1470" height="956" alt="Screenshot 2026-05-28 at 7 33 22 PM" src="https://github.com/user-attachments/assets/7434f349-668b-4bf9-84d0-c9489dd1de01" />


##  Features (Including All Add-ons)
* **Data Extraction & Validation:** Parses HTML profiles into strict JSON schemas using **Pydantic** (includes confidence scores).
* **Influence Scoring:** A custom weighting system ranking KOLs based on citations, h-index, and keyword density.
* **Similarity Matrix:** Uses local **SentenceTransformers** to map semantic research overlap via a visual heatmap.
* **LLM Insights:** Integrates the **Gemini API** to generate automated, cross-profile comparison summaries.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Backend:** Python, Pydantic, BeautifulSoup4
* **AI & Data:** Google GenAI (Gemini), SentenceTransformers, Scikit-learn, Seaborn



---

## Project Structure
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
