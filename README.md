# Resume Optimizer & Skill-Gap Career Coach

An intelligent, full-stack application that analyzes candidate resumes against job descriptions, pinpoints technical skill gaps with high-precision NLP, calculates weighted alignment scores, constructs a topological learning roadmap, and recommends targeted portfolio projects using a multi-provider LLM fallback chain.

---

## 🌟 Key Capabilities

### 1. High-Precision NLP Skill Extraction
* **Deterministic Phrase Matching:** Built on `spaCy`'s blank English model with a compiled `PhraseMatcher` and canonical dictionary (`data/skills.json`), loading in **sub-1 millisecond** with zero runtime weight downloads.
* **Exact Boundary Precision:** Uses regex lookarounds `(?<![a-zA-Z0-9+#&])(...) (?![a-zA-Z0-9+#&]|(?:\.[a-zA-Z]))` to eliminate false positives:
  * `Java` is never falsely credited when `JavaScript` is present.
  * `C++`, `C#`, and `.NET` are matched with boundary-aware symbol detection.
  * Common English verbs and acronyms (like *go*, *r&d*) are prevented from triggering false tech matches.
* **Empirical Benchmark:** Evaluated against ground-truth labels across 10 diverse technical domains, achieving **97.0% Micro-F1** and **98.7% Recall**.

### 2. Weighted Scoring Engine v2
* **Section-Aware Weighting:** Intelligently identifies *Requirements* (weight `1.0`) vs. *Nice to Have / Preferred* qualifications (weight `0.5`).
* **Term Frequency & Emphasis:** Rewards recurring core skills with up to a `+0.5` weighting bonus.
* **Match Quality Credit:** Exact matches receive full credit (`1.0`); related/partial matches earn half credit (`0.5`).
* **Semantic Correlation:** Integrates TF-IDF cosine semantic similarity with harmonic penalty scaling.

### 3. Resilient Multi-Provider LLM Fallback Chain
* **Provider Hierarchy:** Sequentially queries **Groq** (`qwen/qwen3.8-27b`) $\to$ **OpenRouter** (`openrouter/free`) $\to$ **Google Gemini** (`gemini-2.5-flash`).
* **Automatic Rate Limit Cooldown:** Intercepts HTTP 429 errors and automatically puts rate-limited providers into a timed cooldown before trying the next adapter.
* **100% Offline Reliability:** If external APIs are exhausted or unconfigured, gracefully falls back to an offline curated project library (`data/projects/`) and deterministic template generators.
* **7-Day Result Caching:** SHA-256 hashed request caching provides instant sub-10ms response times for identical submissions.

### 4. Token-Efficient Portfolio Project Recommendations
* **Targeted Problem Solving:** Generates 2 realistic, practical portfolio projects solving concrete real-world problems.
* **Skill Synergy:** Seamlessly pairs the candidate's existing strengths with target missing skills.
* **Optimized Token Budget:** Strictly focused on `ProjectTitle`, `Description`, and `KeySkills` badges, cutting token consumption by ~75% and eliminating API timeouts.

### 5. Prioritized Learning Roadmap
* **Topological Prerequisite Ordering:** Applies Kahn's algorithm over a technical prerequisite DAG (e.g. *Git $\to$ Python $\to$ FastAPI $\to$ Docker $\to$ Kubernetes*).
* **Difficulty Tiers:** Structured into *Foundational*, *Intermediate*, and *Advanced* phases with realistic completion hour estimates.
* **Verified Documentation:** Features direct links to 44+ verified official documentation resources (`data/resources.json`).

### 6. Defensive Multi-Format Ingestion
* **Format Flexibility:** Accepts PDF uploads, Word documents (`.docx`), or direct plain-text pasting with dynamic UI tab switching.
* **Security & Health Gates:** Magic-byte validation (`%PDF-` and `PK\x03\x04`), 10-page ceiling, scanned PDF image detection, and strict 5MB upload limits.

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python 3.12, Flask 3.1, Werkzeug |
| **AI / LLM** | Groq API, OpenRouter API, Google GenAI SDK (`gemini-2.5-flash`), Pydantic v2 |
| **NLP & Scoring** | spaCy (PhraseMatcher), scikit-learn (TF-IDF & Cosine Similarity) |
| **Document Processing** | PyPDF2, python-docx |
| **Frontend** | HTML5, Tailwind CSS, Vanilla JavaScript, Dynamic SVG Gauges |
| **Testing & QA** | pytest (55 automated tests passing) |

---

## 📁 Repository Structure

```text
├── app/
│   ├── llm/                 # Multi-provider fallback chain & adapters
│   │   ├── adapters.py      # Groq, Gemini, OpenRouter, and Template adapters
│   │   ├── base.py          # LLMClient interface & Pydantic schemas
│   │   ├── chain.py         # Sequential fallback orchestrator & cooldown logic
│   │   ├── parser.py        # Robust JSON response extraction
│   │   └── validator.py     # Schema and skill validation
│   ├── services/            # Business logic & algorithms
│   │   ├── nlp.py           # spaCy PhraseMatcher & skill extraction
│   │   ├── pdf.py           # Defensive PDF/DOCX parsing & validation
│   │   ├── projects.py      # Prompt construction & curated library loader
│   │   ├── roadmap.py       # Topological prerequisite DAG resolver
│   │   ├── scoring.py       # Section weighting & semantic matching
│   │   └── skills.py        # Canonical normalization & boundary matching
│   ├── config.py            # Environment configurations & limits
│   └── routes.py            # Flask endpoints & request pipeline
├── data/
│   ├── projects/            # Offline curated portfolio project library
│   ├── resources.json       # 44+ verified official documentation links
│   └── skills.json          # Canonical technical skill dictionary & aliases
├── eval/                    # Ground-truth evaluation benchmarks
│   ├── data/                # Labeled test job descriptions & resumes
│   ├── results/             # Benchmark evaluation output records
│   └── run_eval.py          # Evaluation runner (Micro-F1, Precision, Recall)
├── scripts/                 # Utility scripts (library builder, link checker)
├── static/                  # CSS stylesheets, logos, and UI assets
├── templates/               # Responsive Jinja2 frontend template (index.html)
├── tests/                   # 55 automated unit & integration tests
│   ├── test_llm_chain.py    # Multi-provider fallback chain tests
│   ├── test_llm_validation.py # Validator & prompt defense tests
│   ├── test_matching.py     # Skill boundary & normalization tests
│   ├── test_projects.py     # Project generation & library tests
│   ├── test_roadmap.py      # Topological sorting & roadmap tests
│   ├── test_scoring.py      # Weighted scoring & keyword tests
│   └── test_uploads.py      # Defensive file upload & parsing tests
├── .env.example             # Template for API keys and configuration
├── .gitignore               # Clean repository exclusions
├── pytest.ini               # Pytest configuration
├── requirements.txt         # Production dependencies
└── run.py                   # Application entry point
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/vaghela-meetraj/resume-optimizer.git
cd resume-optimizer
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Activate on macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env    # On Windows: copy .env.example .env
```

Configure your API keys in `.env`:
```env
# Primary LLM Provider
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b

# Secondary LLM Provider
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openrouter/free

# Tertiary LLM Provider
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Provider Priority
LLM_PROVIDERS=groq,openrouter,gemini
```
*(Note: If no API keys are provided, the system automatically uses its offline curated project library).*

### 5. Run the Application
```bash
python run.py
```
Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## 🧪 Testing & Evaluation

### Run Test Suite
```bash
python -m pytest
```
*All 55 unit and integration tests execute and pass in ~2 seconds.*

### Run NLP Accuracy Benchmark
```bash
python eval/run_eval.py
```
*Evaluates skill extraction precision, recall, and Micro-F1 against ground-truth labeled documents across 10 engineering domains.*

---

## 🔒 Security & Privacy

* **In-Memory File Processing:** Uploaded resumes are parsed strictly in memory and are never saved to disk.
* **Prompt Privacy:** Candidate personal details (names, phone numbers, addresses, emails) are stripped prior to external LLM calls; only sanitized technical tokens are sent.
* **Defensive Boundaries:** Enforces strict 5MB upload ceilings, magic-byte format validation, and limits text extraction to 10 pages max.
