# Resume Optimizer & Skill-Gap Career Coach

An intelligent full-stack web application that compares candidate resumes against job descriptions, identifies critical skill gaps, computes weighted match scores, and generates tailored portfolio project recommendations using Google Gemini.

---

## 🌟 Key Features

* **Deterministic NLP Skill Extraction:** Utilizes data-driven canonical dictionaries (`data/skills.json`) and regex lookarounds `(?<![a-zA-Z0-9+#&])(...) (?![a-zA-Z0-9+#&]|(?:\.[a-zA-Z]))` to eradicate false positives (e.g. `Java` is never falsely credited inside `JavaScript`, and `C++` / `C#` / `.NET` are parsed with exact boundary precision).
* **Scoring Engine v2:** Intelligently parses job descriptions for *Requirements* vs. *Nice to Have* sections:
  * Required skills receive a `1.0` weight; Preferred skills receive a `0.5` weight.
  * Emphasized / repeated skills in the JD receive up to a `+0.5` weighting bonus.
  * Exact matches earn full credit (`1.0`); related/partial matches earn half credit (`0.5`).
  * No artificial score ceilings (candidates with 100% match score a true 100%).
* **Empirical Evaluation Benchmark:** Built with a ground-truth labeled benchmark across 10 diverse technical domains, achieving **97.0% Micro-F1** and **98.7% Recall** on skill extraction.
* **Resilient AI Project Synthesis:** Leverages Google's `gemini-2.5-flash` with strict JSON schemas to design 2 challenging portfolio projects targeting missing skills, backed by an offline deterministic fallback generator if API rate limits hit.
* **Lightning-Fast & Lightweight:** Zero runtime model downloads (`spacy.blank('en')` + compiled PhraseMatcher) initializing in **sub-1 millisecond**.

---

## 🛠️ Tech Stack

* **Backend:** Python 3.12, Flask 3.1
* **NLP & Matching:** spaCy (Tokenizer & PhraseMatcher), scikit-learn (TF-IDF Vectorization & Cosine Semantic Similarity)
* **Generative AI:** Google GenAI SDK (`gemini-2.5-flash`, fallback to `gemini-flash-latest`)
* **PDF Extraction:** PyPDF2 with 6-stage defensive validation (10MB limit, corruption boundary, scanned image detection)
* **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript, dynamic SVG progress ring, responsive dark/light mode
* **Testing:** Pytest (14 unit tests covering matching boundaries and weighted scoring formulas)

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/vaghela-meetraj/resume-optimizer.git
cd resume-optimizer
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the example environment configuration:
```bash
cp .env.example .env     # On Windows: copy .env.example .env
```
Edit `.env` and paste your Google Gemini API key:
```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### 5. Run the Application
```bash
python run.py
```
Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## 🧪 Running Tests & Evaluation Benchmarks

### Run Unit Tests
```bash
pytest tests/ -v
```
*(All 10 unit tests covering whole-token matching, lookarounds, and Scoring v2 formulas pass in ~1.7s).*

### Run the Evaluation Benchmark
```bash
python eval/run_eval.py
```
*(Evaluates skill extraction precision, recall, and F1 across 10 job archetypes and 3 resumes).*

---

## 🔒 Security & Privacy

* **In-Memory Processing:** Resumes are processed strictly in RAM and never saved to disk.
* **Sanitized AI Prompts:** Candidate personal details (names, phone numbers, addresses) are never sent to external LLMs; only sanitized technical skill tokens and short JD excerpts are passed.
* **Input Boundary Protection:** Enforces file size ceilings (10MB) and rejects corrupted or unreadable image-only PDFs with clear user feedback.
