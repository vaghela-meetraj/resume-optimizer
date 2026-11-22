# Resume Optimizer & Skill-Gap Career Coach

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Flask Version](https://img.shields.io/badge/framework-Flask%203.1-green.svg)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/tests-55%20passed-success.svg)](tests/)
[![NLP Engine](https://img.shields.io/badge/NLP%20Engine-PhraseMatcher%20%2B%20Lookarounds-blueviolet.svg)](app/services/nlp.py)
[![Latency](https://img.shields.io/badge/NLP%20Latency-%3C1ms-orange.svg)](app/services/nlp.py)

An intelligent, production-ready career platform that parses resumes against technical job descriptions, extracts competencies with sub-millisecond precision, computes section-weighted alignment scores, constructs a topologically-sorted prerequisite learning roadmap, and suggests tailored portfolio projects via a resilient multi-provider LLM fallback chain.

---

## Key Capabilities

### 1. High-Precision NLP Skill Extraction
* **Zero-Weight Deterministic Phrase Matching:** Combines pre-compiled regex boundary lookarounds with `spaCy`'s blank English `PhraseMatcher` and a canonical dictionary ([data/skills.json](data/skills.json)), executing in **<1ms** with zero neural network weight downloads.
* **Exact Boundary Lookarounds:** Employs `(?<![a-zA-Z0-9+#&])(...) (?![a-zA-Z0-9+#&]|(?:\.[a-zA-Z]))` to eliminate common NLP false positives:
  * Eliminates false substring matches (e.g., `Java` is never falsely extracted when `JavaScript` is present).
  * Safely detects technical symbols (`C++`, `C#`, `.NET`).
  * Isolates single-letter languages (`C`, `R`) and ambiguous terms (`Go`) to prevent collisions with prose verbs or acronyms (e.g., *R&D*).
  * Distinguishes sentence-ending periods (e.g., `"Experience with Docker."`) from file extension dots.
* **Evaluation Suite:** Includes an automated evaluation harness across 10 engineering domains and sample resumes to benchmark extraction precision and recall against labeled references.

### 2. Weighted Scoring Engine v2
* **Section-Aware Weighting:** Automatically identifies *Requirements / Minimum Qualifications* (base weight `1.0`) versus *Nice to Have / Preferred Qualifications* (base weight `0.5`).
* **Term Frequency & Repetition Bonus:** Rewards recurring core skills with up to a `+0.5` weighting bonus.
* **Dialect & Database Credit:** Relational database systems (e.g., PostgreSQL, MySQL) satisfy generic SQL requirements with full credit (`1.0`); general SQL satisfies specific database engine requirements with partial credit (`0.5`).
* **Semantic Correlation:** Integrates TF-IDF unigram/trigram cosine similarity as a dynamic tie-breaker (up to `+5%`).
* **True Percentage Scoring:** Removed artificial score ceilings—matching 100% of requirements delivers an honest 100%.

### 3. Resilient Multi-Provider LLM Fallback Chain
* **Sequential Provider Priority:** Sequentially queries **Groq** (`qwen/qwen3.8-27b`) $\to$ **OpenRouter** (`openrouter/free`) $\to$ **Google Gemini** (`gemini-2.5-flash`).
* **Automatic Rate Limit Cooldown:** Intercepts HTTP 429 errors and places rate-limited adapters into an automatic 60-second cooldown window.
* **Self-Corrective Validation Retry:** Project ideas pass through strict Pydantic and business rule validators. If formatting or overlap criteria fail, validator feedback is injected into a single self-corrective retry.
* **100% Offline Resilience:** If all external APIs are exhausted or unconfigured, seamlessly falls back to a curated offline project library ([data/projects/](data/projects/)) and deterministic template generators.
* **7-Day Result Caching:** SHA-256 hashed request caching provides instant sub-10ms responses for repeated queries.

### 4. Topological Learning Roadmap (DAG Kahn's Algorithm)
* **Prerequisite Graph Resolution:** Analyzes technical skill dependencies from [data/resources.json](data/resources.json) using Kahn's topological sorting algorithm.
* **Pedagogical Ordering:** Ensures foundational technologies (`Linux`, `Git`, `Docker`) strictly precede dependent advanced tools (`Kubernetes`, `CI/CD`).
* **Enriched Guidance:** Provides estimated learning hours, priority weights based on missing dependent skills, and verified official documentation links.

### 5. Defensive Multi-Format Ingestion
* **Format Flexibility:** Ingests digital PDFs, Word documents (`.docx`), or raw pasted text with dynamic UI tab switching.
* **Deep Security Gates:** Validates raw magic bytes (`%PDF-` for PDF, `PK\x03\x04` for DOCX), rejects scanned PDFs lacking selectable text, enforces a 10-page ceiling, and applies a strict 5MB upload limit.

---

## Evaluation Framework

The project includes an evaluation suite in [eval/](eval/) designed to measure skill extraction accuracy across technical domains:

* **Labeled Test Dataset:** 10 diverse job descriptions (Backend, Frontend, Full-Stack, AI/ML, DevOps, Data Engineering, Mobile, Java, Data Science, Embedded Systems) and sample resumes in `eval/data/`.
* **Ground-Truth References:** Standardized reference skill sets defined in `eval/labels.json`.
* **Standard Metrics:** Computes Precision, Recall, Micro-F1, and Macro-F1 across all test documents to guide NLP tuning without relying on subjective impressions.

To run the evaluation suite locally:
```bash
python eval/run_eval.py
```
Results and extraction error analyses are written to `eval/results/baseline.json`.

---

## Tech Stack & Dependencies

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Web Framework** | Flask 3.1.3, Werkzeug | Lightweight, predictable routing, application factory pattern |
| **Rate Limiting** | Flask-Limiter 4.1.1 | In-memory sliding window rate limits (10 req/min on `/analyze`) |
| **Caching** | Flask-Caching 2.4.1 | Low-latency 7-day SHA-256 key-value result cache |
| **NLP & Matching** | spaCy 3.8, scikit-learn 1.9 | Sub-millisecond PhraseMatcher (`attr='LOWER'`) + TF-IDF cosine similarity |
| **LLM Orchestration** | OpenAI SDK, google-genai, Pydantic v2 | Standardized `LLMClient` interface, multi-provider fallback, schema validation |
| **Document Processing** | pypdf 7.1.0, python-docx 1.2.0 | Memory-buffered extraction for PDF, DOCX paragraphs and tables |
| **Frontend** | HTML5, Tailwind CSS, Vanilla JS | Zero npm dependencies, responsive layout, dynamic SVG score gauges |
| **Testing** | pytest 9.2.0 | 55 unit and integration tests passing in ~1.5s |

---

## Repository Structure

```text
├── app/
│   ├── __init__.py          # Flask app factory (create_app), logging & security headers
│   ├── config.py            # Environment configurations with safe defaults
│   ├── extensions.py        # Shared cache & rate limiter singletons
│   ├── routes.py            # HTTP controllers (/, /analyze, /health, /api/cache/clear)
│   ├── llm/                 # Multi-provider LLM fallback chain
│   │   ├── adapters.py      # Groq, Gemini, OpenRouter, and Template adapters
│   │   ├── base.py          # LLMClient interface & Pydantic models
│   │   ├── cache.py         # 7-day SHA-256 request cache
│   │   ├── chain.py         # Sequential fallback orchestrator & cooldown logic
│   │   ├── parser.py        # Resilient JSON substring & markdown fence extraction
│   │   └── validator.py     # Schema rules, overlap checks, and corrective retry feedback
│   └── services/            # Core business logic
│       ├── nlp.py           # NLPProcessor singleton & PhraseMatcher
│       ├── pdf.py           # Defensive PDF/DOCX ingestion & magic-byte checking
│       ├── projects.py      # Prompt construction & curated library loader
│       ├── roadmap.py       # Kahn's topological sort prerequisite resolver
│       ├── scoring.py       # Section weighting v2 & semantic matching
│       └── skills.py        # Canonical normalization & SQL heuristics
├── data/
│   ├── projects/            # Curated offline portfolio project library
│   ├── resources.json       # 44+ verified learning resources with prerequisite graph
│   └── skills.json          # 1,842-line canonical skills dictionary & aliases
├── eval/
│   ├── data/                # 10 labeled job descriptions & 3 resumes
│   ├── results/             # Benchmark verification records
│   └── run_eval.py          # Automated evaluation runner (Micro-F1, Precision, Recall)
├── static/                  # Style sheets and branding assets
├── templates/               # Responsive Jinja2 frontend template (index.html)
├── tests/                   # 55 automated unit & integration tests
├── .env.example             # Documented configuration template
├── pytest.ini               # Pytest configuration with pythonpath
├── requirements.txt         # Pinned production dependencies
└── run.py                   # Application entry point (configurable PORT)
```

---

## Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/vaghela-meetraj/resume-optimizer.git
cd resume-optimizer
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate on macOS / Linux:
source .venv/bin/activate

# Activate on Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Copy [.env.example](.env.example) to `.env`:
```bash
cp .env.example .env
```

Configure API keys in `.env` (optional — offline mode activates automatically if unconfigured):
```env
PORT=5001
LLM_PROVIDERS=groq,openrouter,gemini

GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b

OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openrouter/free

GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### 5. Launch the Application
```bash
python run.py
```
*Note: The server defaults to port `5001` (avoiding the macOS AirPlay Receiver port `5000` collision). Open your browser to:*
**http://127.0.0.1:5001/**

---

## Testing & Quality Assurance

### Run Unit & Integration Tests
```bash
python -m pytest
```
*All 55 tests pass in ~1.5 seconds:*
* `test_matching.py`: Boundary precision, canonical normalization, SQL dialect handling.
* `test_scoring.py`: Section weighting, repetition bonuses, unsectioned fallbacks.
* `test_roadmap.py`: Topological sorting, DAG prerequisite resolution, hours estimates.
* `test_projects.py`: Curated library matching, fallback generation, schema fixing.
* `test_llm_chain.py`: Cooldown handling, daily budget limits, sequential provider fallback.
* `test_llm_validation.py`: Pydantic validation, sentence count tolerances, verb checks.
* `test_uploads.py`: File size caps, corrupted files, scanned PDF detection, DOCX tables.

### Run Benchmark Evaluation
```bash
python eval/run_eval.py
```

---

## REST API Reference

| Endpoint | Method | Rate Limit | Description |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | None | Renders the primary user interface. |
| `/analyze` | `POST` | 10 per min | Ingests resume (`resume` file or `resume_text`) and `job_desc`. Returns ATS match score, extracted skills, topological roadmap, and recommended projects. |
| `/health` | `GET` | None | Returns server health status, loaded NLP engine state, and active LLM providers. |
| `/api/cache/clear` | `POST`, `GET` | None | Flushes the 7-day SHA-256 project recommendation cache. |

---

## Security & Privacy Guarantees

* **In-Memory Streaming:** Resumes are processed strictly in RAM buffers via `io.BytesIO` and are never written to disk.
* **Prompt Privacy:** Candidate personal details (names, phone numbers, addresses, emails) are stripped prior to external LLM calls; only sanitized technical tokens are sent.
* **Injection Defense:** User-supplied job descriptions are isolated inside `<job_description>` XML boundaries to prevent prompt hijacking.
* **Upload Safety:** Strict 5MB limit (`MAX_CONTENT_LENGTH`), magic-byte inspection, and a 10-page extraction ceiling protect against decompression bombs.
