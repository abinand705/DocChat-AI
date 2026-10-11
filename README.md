# DocChat-AI 📄🤖

DocChat-AI is an intelligent, retrieval-augmented document question-answering assistant built with **Streamlit**, **Google Gemini**, and **ChromaDB**. It enables users to upload PDF and TXT documents, ask natural language questions, and receive streaming answers with deterministic, page-level citations.

---

## 🔗 Demo

- **Live Application:** <!-- TODO: Add live demo link here --> [Live Demo Link (Insert URL here)](#)
- **Preview:**
  <!-- TODO: Add demo GIF or screenshot below -->
  ![DocChat-AI Demo Preview](docs/demo_preview.gif)

---

## 🌟 Key Capabilities

- **Session-Isolated Vector Storage:** Each user session operates in an isolated ChromaDB collection (`s_<session_id>`), preventing cross-user document leakage with automatic lifecycle cleanup of inactive sessions.
- **Resilient Multi-Tier Generation:** Primary generation with `gemini-2.5-flash`, featuring automatic retry on per-minute rate limits and seamless fallback to `gemini-2.5-flash-lite` if quotas or daily caps are encountered.
- **Deterministic Numbered Citations:** Context chunks are passed as numbered reference tokens (`[1]`, `[2]`), which are programmatically resolved and rewritten into exact document filenames and page numbers (`[file, p.X]`).
- **In-Memory Caching & Rate Limiting:** Sliding-window rate limiter for document embeddings, persistent hash-based embedding cache, and session-aware answer caching for repeated queries.
- **Safety & Input Controls:** Document upload caps (10 MB, 60 pages), session chunk boundaries, daily API call limits with automatic UTC midnight reset, and sanitized HTML rendering.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Ingestion["1. Document Ingestion & Storage"]
        A[User Upload / Demo PDF] --> B[PyPDF Loader]
        B --> C[Header & Footer Cleaning]
        C --> D[Text Chunker<br/>400 chars, 48 overlap]
        D --> E[Gemini Embedding Model<br/>Rate Limiter & Hash Cache]
        E --> F[(ChromaDB<br/>Session Collection s_*)]
    end

    subgraph Retrieval["2. Semantic Retrieval"]
        Q[User Query] --> G[Embed Query]
        G --> H[ChromaDB Vector Search<br/>File & Session Filter]
        F -.-> H
        H --> I{Similarity >= MIN_SCORE?}
        I -- No --> J[Direct Not-Found Refusal]
        I -- Yes --> K[Top-K Context Chunks<br/>Labeled 1, 2, ...]
    end

    subgraph Generation["3. Generation & Citation Resolution"]
        K --> L[Gemini 2.5 Flash]
        L -- 429 / Quota Error --> M[Fallback: Gemini 2.5 Flash-Lite]
        L --> N[Streamed Response with Bracket Citations]
        M --> N
        N --> O[Citation Resolver<br/>Rewrites [1] -> [file, p.X]]
        O --> P[Streamlit UI<br/>Sanitized Cards & Sources Expander]
    end
```

---

## 🛠️ Tech Stack

| Component / Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Web Interface** | [Streamlit](https://streamlit.io/) (`~=1.65.0`) | Interactive chat UI, file uploads, citations expanders, and session management |
| **LLM Generation** | [Google GenAI SDK](https://github.com/google-gemini/generative-ai-python) (`~=2.29.0`) | Primary (`gemini-2.5-flash`) and fallback (`gemini-2.5-flash-lite`) generation models |
| **Embeddings** | Google Gemini (`gemini-embedding-001`) | Dense semantic embeddings with sliding-window rate limiting and on-disk SQLite cache |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) (`~=1.5.9`) | In-process vector database with cosine distance search and per-session collection isolation |
| **Document Processing** | [PyPDF](https://pypdf.readthedocs.io/) (`~=6.19.0`) | Document parsing, text layer inspection, and repetitive running header/footer removal |
| **Environment Management** | [python-dotenv](https://github.com/theskumar/python-dotenv) (`~=1.2.4`) | Environment variable loading for API keys and local configuration |

---

## 🔬 Design Decisions and Findings

### 1. Chunk Size & Retrieval Depth Experiment
<!-- TODO: Fill in findings from chunk_experiment.py (grid search over chunk sizes 400-1000 and top-k 3-6) -->
- *Placeholder: Describe results from running `scripts/chunk_experiment.py`, trade-offs between coverage fraction and retrieval precision, and why `CHUNK_SIZE = 400` with `CHUNK_OVERLAP = 48` and `TOP_K = 6` was chosen.*

### 2. Score Threshold Calibration & Refusal Mechanism
<!-- TODO: Fill in findings from calibrate_threshold.py regarding near-topic separation -->
- *Placeholder: Similarity scores alone could not reliably separate near-topic unanswerable questions from valid answerable queries. As a result, the cosine threshold (`MIN_SCORE = 0.5`) filters completely irrelevant queries, while refusal for out-of-scope or unanswerable queries relies primarily on prompt grounding.*

### 3. Deterministic Citation Approach
<!-- TODO: Fill in rationale for numbered chunk citation pipeline -->
- *Placeholder: Describe how passing context chunks labeled as `[1]`, `[2]`, etc. avoids prompt bloat and prevents hallucinated filenames or fabricated page numbers, followed by programmatic replacement in Python to format `[filename, p.X]`.*

---

## ⚠️ Limitations

- **Scanned & Image PDFs Unsupported:** Documents must contain an extractable text layer. Optical Character Recognition (OCR) is not currently integrated; pages consisting solely of scanned images will trigger a notification.
- **Flattened Table Extraction:** Complex tabular layouts and multi-column tables are extracted as linear plain text, which may diminish semantic table structure.
- **Free-Tier API Rate Limits:** Subject to Google AI Studio free-tier quotas (per-minute RPM and daily RPD thresholds), mitigated by automatic retries, exponential backoffs, and fallback models.
- **Evaluation Dataset Scope:** The automated evaluation suite currently benchmarks against focused test suites; evaluations on larger, heterogeneous multi-domain corpora remain ongoing.

---

## 📁 Project Structure

```
DocChat-AI/
├── .streamlit/
│   └── config.toml             # Streamlit server and upload configuration
├── data/
│   └── sample.pdf              # Demo reference document
├── eval/
│   ├── run_eval.py             # Evaluation runner (retrieval-only & full RAG modes)
│   ├── questions.json          # Curated evaluation benchmark questions
│   └── chunk_experiment.md    # Chunking & top-k experiment methodology
├── scripts/
│   ├── calibrate_threshold.py  # Utility to inspect similarity score distributions
│   └── chunk_experiment.py     # Grid search runner for chunk size & top-k tuning
├── src/
│   ├── chunker.py              # Text chunking with sliding character overlap
│   ├── config.py               # Hyperparameters, safety bounds & daily call limiter
│   ├── embeddings.py           # Gemini embedding client with rate limiter & disk cache
│   ├── loader.py               # PDF/TXT loader with running header/footer filtering
│   ├── prompts.py              # Grounding prompts, untrusted text warnings & citation format
│   ├── rag.py                  # End-to-end RAG orchestrator with quota handling & answer cache
│   └── vector_store.py         # Session-isolated ChromaDB storage and cleanup
├── app.py                      # Interactive Streamlit web application
├── requirements.txt            # Compatible pinned direct dependencies
├── LICENSE                     # MIT License
├── .env.example                # Example environment variables template
├── .gitignore                  # Git ignore rules for private data & artifacts
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- A Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))

### 2. Clone the Repository
```bash
git clone https://github.com/abinand705/DocChat-AI.git
cd DocChat-AI
```

### 3. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env` and configure your API key:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 💻 Usage

### Run the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. You can upload custom documents or click **"Load demo document"** in the sidebar to start asking questions immediately.

### Run Retrieval or Full Evaluation
```bash
# Evaluate retrieval metrics (Hit@k, MRR, threshold) without LLM calls:
python eval/run_eval.py --retrieval-only

# Run evaluation on answerable questions only:
python eval/run_eval.py --only-type answerable

# Run full evaluation across the complete question set:
python eval/run_eval.py
```

### Tune Similarity Threshold
```bash
python scripts/calibrate_threshold.py --pdf data/sample.pdf --questions eval/questions.json
```

### Run Chunking Grid Search
```bash
python scripts/chunk_experiment.py --pdf data/sample.pdf --questions eval/questions.json
```

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.