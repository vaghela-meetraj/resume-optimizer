# Resume Optimizer & Career Coach

A resume analyzer and project recommender tool. This app compares your resume to a job description, highlights missing skills, and suggests projects to help you learn what you're missing.

## Features

- **Resume Parsing**: Extracts text from PDF resumes using PyPDF2.
- **Skill Extraction**: Uses spaCy and NLTK to find technical skills, tools, and frameworks in the text.
- **Skill Matching**: Maps variations of skills (like `React.js` and `React`) so you get accurate match scores.
- **Project Recommendations**: Uses the Google Gemini API to suggest realistic projects to help you learn your missing skills.
- **Fallback Mode**: If the Gemini API is down or rate-limited, it will still generate project ideas using a basic NLP fallback.
- **Rate Limiting & Caching**: Uses Flask-Limiter and Flask-Caching to keep things running smoothly.

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, TailwindCSS, JavaScript
- **NLP**: spaCy, NLTK, scikit-learn
- **AI**: Google Gemini API
- **PDF**: PyPDF2

## Setup Instructions

### Prerequisites
- Python 3.8+
- A Google Gemini API Key

### 1. Clone the repo
```bash
git clone https://github.com/vaghela-meetraj/resume-optimizer
cd resume-optimizer
```

### 2. Set up a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

### 5. Set up env variables
Create a `.env` file in the root directory and add your Gemini API Key:
```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### 6. Run it
```bash
python app.py
```
The app will be running at `http://127.0.0.1:5000/`.

## How to use it

1. Open `http://127.0.0.1:5000/` in your browser.
2. Upload your Resume as a PDF (max 10MB).
3. Paste the Job Description you're applying for.
4. Click Analyze.
5. See your match score, which skills you hit/missed, and get project ideas to improve your profile.

## How it works under the hood

- **Concurrency**: Text parsing and NLP extraction run in parallel using ThreadPoolExecutor to speed things up.
- **NLP Singleton**: The NLP models (like spaCy) are loaded once and reused so we don't waste time reloading them for every request.
- **Retries**: The app handles Gemini API rate limits gracefully with built-in retries and exponential backoff.
