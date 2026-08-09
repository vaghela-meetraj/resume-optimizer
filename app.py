import os
import json
import re
import threading
from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from google import genai
from google.genai import types
import PyPDF2 as pdf
from dotenv import load_dotenv
import concurrent.futures
import time
import io
import random

# nlp libraries
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# app configuration
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is not set in your .env file.")

app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# skill normalization and matching
def normalize_skill(skill: str) -> str:
    skill_map = {
        'react.js': 'react', 'reactjs': 'react', 'react js': 'react',
        'vue.js': 'vue', 'vuejs': 'vue', 'vue js': 'vue',
        'angular.js': 'angular', 'angularjs': 'angular',
        'node.js': 'nodejs', 'node js': 'nodejs',
        'next.js': 'nextjs', 'next js': 'nextjs',
        'express.js': 'express', 'expressjs': 'express',
        'mongodb': 'mongo', 'postgresql': 'postgres',
        'sql server': 'sqlserver', 'ms sql': 'sqlserver',
        'amazon web services': 'aws', 'google cloud platform': 'gcp',
        'google cloud': 'gcp', 'azure cloud': 'azure',
        'machine learning': 'ml', 'deep learning': 'dl',
        'natural language processing': 'nlp', 'computer vision': 'cv',
        'rest api': 'restful', 'restful api': 'restful',
        'ci/cd': 'cicd', 'ci cd': 'cicd',
        'dom': 'javascript', 'ajax': 'javascript', 'fetch': 'javascript',
        'promises': 'javascript', 'async': 'javascript', 'es6': 'javascript',
    }
    normalized = skill.lower().strip()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    return skill_map.get(normalized, normalized)

def fuzzy_skill_match(resume_skills: set, jd_skills: set) -> dict:
    norm_resume = {normalize_skill(s): s for s in resume_skills}
    norm_jd = {normalize_skill(s): s for s in jd_skills}
    
    exact_matches = set()
    partial_matches = set()
    missing = set()
    
    for norm_jd_skill, orig_jd_skill in norm_jd.items():
        matched = False
        if norm_jd_skill in norm_resume:
            exact_matches.add(orig_jd_skill)
            matched = True
        else:
            for norm_resume_skill in norm_resume.keys():
                if (len(norm_jd_skill) > 3 and norm_jd_skill in norm_resume_skill) or \
                   (len(norm_resume_skill) > 3 and norm_resume_skill in norm_jd_skill):
                    partial_matches.add(orig_jd_skill)
                    matched = True
                    break
        if not matched:
            missing.add(orig_jd_skill)
            
    return {
        'exact': sorted(exact_matches),
        'partial': sorted(partial_matches),
        'missing': sorted(missing),
        'total_matched': len(exact_matches) + len(partial_matches)
    }

# nlp processing logic
class NLPProcessor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("Initializing NLP models...")
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        self.nlp.max_length = 2_000_000

        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

        self.tech_skills_patterns = self._load_tech_patterns()
        self.NOISE_TERMS = {
            'experience', 'exposure', 'familiarity', 'hands', 'nice', 'preferred', 
            'qualifications', 'required', 'understanding', 'skills', 'library', 
            'knowledge', 'ability', 'years', 'work', 'working', 'using', 'tool', 
            'tools', 'system', 'systems', 'support', 'software', 'solutions',
            'strong', 'excellent', 'good', 'proficiency', 'proficient', 'expert',
            'development', 'technologies', 'platform', 'framework', 'environment',
            'application', 'applications', 'service', 'services', 'team', 'project',
            'projects', 'demonstrated', 'solid', 'extensive', 'deep', 'proven',
            'building', 'creating', 'developing', 'implementing', 'design', 'include',
            'dom', 'ajax', 'fetch', 'promises', 'async', 'callbacks', 'closures',
            'json', 'xml', 'es6', 'es2015', 'es2020', 'modules', 'hoisting',
            'flexbox', 'grid', 'animations', 'transitions', 'responsive',
            'loops', 'arrays', 'objects', 'functions', 'classes', 'inheritance',
            'api', 'apis'
        }
        print("NLP models loaded successfully.")

    def _load_tech_patterns(self):
        return {
            'languages': r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|PHP|Scala|R|MATLAB|SQL|HTML|CSS|Bash|Shell|Perl|Dart|C)\b',
            'frameworks': r'\b(React|React\.js|Angular|Vue|Django|Flask|FastAPI|Spring|Express|Node\.?js|Next\.?js|Nuxt|Laravel|Rails|ASP\.NET|Flutter|SwiftUI|Svelte|Gatsby|Remix)\b',
            'libraries': r'\b(NumPy|Pandas|TensorFlow|PyTorch|Scikit-learn|Keras|OpenCV|NLTK|spaCy|jQuery|Bootstrap|Tailwind|Material-UI|Redux|RxJS|Matplotlib|Jest|Zustand|Recoil)\b',
            'databases': r'\b(MySQL|PostgreSQL|MongoDB|Redis|Cassandra|DynamoDB|Firebase|Oracle|SQL Server|SQLite|Elasticsearch|Neo4j|MariaDB|Supabase|Prisma)\b',
            'cloud': r'\b(AWS|Azure|GCP|Google Cloud|Kubernetes|Docker|Jenkins|CI/CD|Terraform|Ansible|CloudFormation|ECS|Lambda|EC2|S3|Vercel|Netlify|Heroku)\b',
            'tools': r'\b(Git|GitHub|GitLab|Bitbucket|JIRA|Confluence|Postman|Swagger|Webpack|Babel|npm|yarn|Maven|Gradle|VS Code|IntelliJ|Eclipse|Figma|Vite)\b',
            'concepts': r'\b(REST|GraphQL|Microservices|Serverless|OAuth|JWT|TDD|BDD|Agile|Scrum|DevOps|Machine Learning|Deep Learning|NLP|Computer Vision|OOP|MVC|SOLID)\b'
        }

    def extract_skills_nlp(self, text: str) -> set:
        raw_skills = set()
        for category, pattern in self.tech_skills_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            raw_skills.update([m.strip() for m in matches])

        text_sample = text[:50_000]
        doc = self.nlp(text_sample)
        for token in doc:
            if (token.pos_ in ['PROPN', 'NOUN'] and len(token.text) > 2 and
                token.text.lower() not in self.stop_words and
                token.text.lower() not in self.NOISE_TERMS and not token.text.isdigit()):
                if ('.' in token.text or token.text.isupper() or 
                    (token.text[0].isupper() and any(c.isupper() for c in token.text[1:]))):
                    raw_skills.add(token.text)

        return {s for s in raw_skills if len(s) > 2 and s.lower() not in self.NOISE_TERMS}

    def calculate_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        try:
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3), stop_words='english', lowercase=True)
            vectors = vectorizer.fit_transform([resume_text, jd_text])
            return float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100)
        except Exception as e:
            print(f"⚠️ Similarity calculation failed: {e}")
            return 0.0

nlp_processor = NLPProcessor()

# gemini api setup
try:
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to configure Gemini SDK: {e}")

GEMINI_MODELS = [
    ("gemini-2.5-flash", 60),
    ("gemini-flash-latest", 60),
]

def get_gemini_response_with_retry(prompt: str, max_retries: int = 3) -> str:
    last_error = None
    for model_name, timeout in GEMINI_MODELS:
        for attempt in range(max_retries):
            try:
                print(f"[*] [{model_name}] Attempt {attempt + 1}/{max_retries}")
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=10000,
                        response_mime_type="application/json",
                        response_schema={
                            "type": "object",
                            "properties": {
                                "ProjectIdeas": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "ProjectName": {"type": "string"},
                                            "Description": {"type": "string"},
                                            "SkillsPracticed": {"type": "array", "items": {"type": "string"}}
                                        },
                                        "required": ["ProjectName", "Description", "SkillsPracticed"]
                                    },
                                    "minItems": 2,
                                    "maxItems": 2
                                }
                            },
                            "required": ["ProjectIdeas"]
                        }
                    )
                )
                if not response or not response.text:
                    raise ValueError("Empty response received from Gemini.")
                
                
                json_string = response.text.strip()
                if json_string.startswith("```json"):
                    json_string = json_string[7:]
                elif json_string.startswith("```"):
                    json_string = json_string[3:]
                if json_string.endswith("```"):
                    json_string = json_string[:-3]
                    
                return json_string.strip()
            
            except Exception as e:
                last_error = str(e)
                print(f"[*] [{model_name}] Attempt {attempt + 1} failed: {last_error[:100]}...")
                
                if attempt < max_retries - 1:
                    sleep_time = (2 ** attempt) + random.uniform(0.1, 1.0)
                    print(f"[*] Overloaded. Waiting {sleep_time:.2f}s before retry...")
                    time.sleep(sleep_time)
        
        print(f"[*] Exhausted all retries for {model_name}. Moving to next fallback model...")

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

# validate and format projects
def validate_and_fix_projects(projects: list, missing_skills: list, matched_skills: list) -> list:
    """Ensure exactly 2 projects, 5-6 line descriptions, and accurate skills."""
    valid_projects = []
    all_known_skills = set(missing_skills + matched_skills)
    
    for p in projects[:2]:  
        name = p.get("ProjectName", "Untitled Project")
        desc = p.get("Description", "")
        skills = p.get("SkillsPracticed", [])
        
        if len(desc.split()) < 15:
            desc = (f"{desc} Ensure the app follows modern design patterns, clean code principles, "
                    f"and robust architecture.")
        
        filtered_skills = []
        for s in skills:
            norm_s = normalize_skill(s)
            if any(norm_s in normalize_skill(known) for known in all_known_skills) or len(filtered_skills) < 5:
                filtered_skills.append(s)
       
        if len(filtered_skills) < 5:
            filtered_skills.extend(missing_skills[:5-len(filtered_skills)])
        filtered_skills = list(dict.fromkeys(filtered_skills))[:7]
        
        valid_projects.append({
            "ProjectName": name,
            "Description": desc,
            "SkillsPracticed": filtered_skills
        })
    
    
    fallback_templates = [
        {
            "name": "Full-Stack Task Manager App",
            "desc": "Build a responsive task management application with user authentication, database integration, and a RESTful API backend. Focus on state management and modular components."
        },
        {
            "name": "Interactive E-Commerce Storefront",
            "desc": "Develop a modern e-commerce platform with product catalogs, shopping cart functionality, and mock payment integration. Ensure mobile responsiveness and fast load times."
        },
        {
            "name": "Real-Time Chat Application",
            "desc": "Create a live chat application supporting multiple rooms, direct messaging, and online presence indicators. Implement WebSocket connections for low-latency communication."
        }
    ]
    
    while len(valid_projects) < 2:
        template = fallback_templates[len(valid_projects) % len(fallback_templates)]
        valid_projects.append({
            "ProjectName": template['name'],
            "Description": template['desc'],
            "SkillsPracticed": (missing_skills + matched_skills)[:6]
        })
        
    return valid_projects[:2]

def generate_fallback_response(missing_skills: list, matched_skills: list) -> list:
    print("[*] Using intelligent NLP fallback (Gemini unavailable).")
    
    cloud = [s for s in missing_skills if any(c in s.lower() for c in ['aws','azure','gcp','docker','kubernetes','cloud','lambda'])]
    web = [s for s in missing_skills if any(w in s.lower() for w in ['react','angular','vue','node','next','express'])]
    db = [s for s in missing_skills if any(d in s.lower() for d in ['mongo','postgres','mysql','redis','sql','database'])]
    ml = [s for s in missing_skills if any(m in s.lower() for m in ['tensorflow','pytorch','ml','ai','keras','scikit'])]
    
    projects = []
    
    # Project 1: Cloud/Infrastructure or Web
    if cloud or web:
        focus = cloud[0] if cloud else web[0]
        projects.append({
            "ProjectName": f"Cloud-Native {focus.title()} Monitoring & Analytics Platform",
            "Description": (f"Design and deploy a real-time monitoring dashboard that tracks system health and user metrics. "
                            f"Implement containerized microservices and automated failure recovery mechanisms."),
            "SkillsPracticed": list(dict.fromkeys(cloud + web + ['Docker', 'CI/CD', 'REST API']))[:6]
        })
    
    # Project 2: Data/ML or Full-Stack
    if ml or db or web:
        focus = ml[0] if ml else (db[0] if db else web[0] if web else "Full-Stack")
        projects.append({
            "ProjectName": f"Intelligent {focus.title()} SaaS Application",
            "Description": (f"Develop a multi-tenant SaaS platform with user authentication and real-time data synchronization. "
                            f"Implement secure backend APIs, a normalized database schema, and an interactive frontend dashboard."),
            "SkillsPracticed": list(dict.fromkeys(ml + db + web + ['Database Design', 'Authentication', 'React']))[:6]
        })
    
    return validate_and_fix_projects(projects, missing_skills, matched_skills)

# extract text from pdf
def extract_pdf_text(uploaded_file) -> str:
    MAX_FILE_SIZE = 10 * 1024 * 1024
    try:
        file_content = uploaded_file.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read uploaded file: {e}")
    if len(file_content) == 0:
        raise ValueError("Uploaded file is empty.")
    if len(file_content) > MAX_FILE_SIZE:
        raise ValueError("File too large. Maximum allowed size is 10MB.")
    try:
        reader = pdf.PdfReader(io.BytesIO(file_content))
    except Exception as e:
        raise ValueError(f"Invalid or corrupted PDF file: {e}")
    if len(reader.pages) == 0:
        raise ValueError("PDF file contains no pages.")
    text_parts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text)
        except Exception:
            continue
    if not text_parts:
        raise ValueError("No readable text found. If this is a scanned PDF or image, please convert it to a standard text PDF first.")
    full_text = " ".join(text_parts).strip()
    if len(full_text) < 50:
        raise ValueError("Extracted text is too short. Please check your PDF.")
    return full_text

# generate ai prompt
def build_prompt(missing_skills: list, matched_skills: list, job_description: str) -> str:
    jd_snippet = job_description[:2000]
    missing_str = ", ".join(missing_skills[:20])
    matched_str = ", ".join(matched_skills[:15])
    
    return f"""You are a senior technical hiring manager and career coach.

**Job Description Context:**
{jd_snippet}

**Candidate's Current Skills:** {matched_str}

**YOUR TASK:**
these are the skills that i have to learn give me / suggest me 2 project ideas that can help me understand them properly and i can master them.
Critical Missing Skills to learn: {missing_str}

**STRICT REQUIREMENTS:**
1. Return EXACTLY 2 projects. No more, no less.
2. Each Description MUST be 2-3 detailed lines explaining architecture, features, and deployment strategy.
3. SkillsPracticed MUST ONLY include 5-7 technologies DIRECTLY required for that specific project. DO NOT add unrelated skills.
4. Focus on ONE domain per project (Cloud, Web, Data, or ML).
5. The project name should be like the given below, Real-Time Collaborative Task Board or Personal Finance & Budget Tracker this are examples.
6. Make projects realistic, industry-standard, and genuinely challenging.
7. Project name should not have any skills word in them.

Return valid JSON matching the schema."""

# routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze():
    start_time = time.time()
    print("[*] Resume analysis request received.")

    try:
        if 'resume' not in request.files:
            return jsonify({"error": "No resume file uploaded."}), 400
        file = request.files['resume']
        job_desc = request.form.get('job_desc', '').strip()
        if not file or file.filename == '':
            return jsonify({"error": "No file selected."}), 400
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are supported. Please upload a PDF."}), 400
        if not job_desc or len(job_desc) < 50:
            return jsonify({"error": "Job description is required and must be detailed."}), 400

        print("[*] Extracting PDF text...")
        try:
            resume_text = extract_pdf_text(file)
        except (ValueError, RuntimeError) as e:
            return jsonify({"error": str(e)}), 400

        print("[*] Running NLP analysis...")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                f_resume = executor.submit(nlp_processor.extract_skills_nlp, resume_text)
                f_jd = executor.submit(nlp_processor.extract_skills_nlp, job_desc)
                f_sim = executor.submit(nlp_processor.calculate_semantic_similarity, resume_text, job_desc)
                resume_skills = f_resume.result(timeout=5)
                jd_skills = f_jd.result(timeout=5)
                text_similarity = f_sim.result(timeout=5)
        except Exception as e:
            print(f"[*] NLP error: {e}")
            resume_skills, jd_skills, text_similarity = set(), set(), 0.0

        match_result = fuzzy_skill_match(resume_skills, jd_skills)
        matched_skills_list = match_result['exact'] + match_result['partial']
        missing_skills_list = match_result['missing']
        
        if len(jd_skills) > 0:
            base_score = (len(matched_skills_list) / len(jd_skills)) * 100
        else:
            base_score = 0.0
            
        semantic_boost = min(text_similarity * 0.05, 5)
        final_match_score = min(round(base_score + semantic_boost), 95)
        
        print(f"[*] JD Skills: {len(jd_skills)} | Matched: {len(matched_skills_list)} | Missing: {len(missing_skills_list)}")
        print(f"[*] Base Skill Match: {base_score:.1f}% | Semantic Boost: {semantic_boost:.1f}% | Final: {final_match_score}%")

        print("[*] Building prompt...")
        prompt = build_prompt(missing_skills_list, matched_skills_list, job_desc)

        print("[*] Calling Gemini API...")
        ai_used = True
        try:
            json_string = get_gemini_response_with_retry(prompt, max_retries=2)
            project_data = json.loads(json_string)
            projects = project_data.get("ProjectIdeas", [])
            print("[*] Gemini analysis complete.")
        except Exception as e:
            print(f"[*] Gemini unavailable: {str(e)[:80]}...")
            print("[*] Falling back to intelligent NLP project generation...")
            projects = generate_fallback_response(missing_skills_list, matched_skills_list)
            ai_used = False

        projects = validate_and_fix_projects(projects, missing_skills_list, matched_skills_list)

        data = {
            "JD Match": str(final_match_score),
            "MatchedSkills": matched_skills_list,
            "MissingSkills": missing_skills_list,
            "ExactMatches": match_result['exact'],
            "PartialMatches": match_result['partial'],
            "SkillMatchScore": round(base_score),
            "SemanticScore": round(text_similarity),
            "ProjectIdeas": projects,
            "ProcessingTime": round(time.time() - start_time, 2),
            "NLPEnhanced": True,
            "AIUsed": ai_used
        }

        print(f"[*] Done in {data['ProcessingTime']}s | AI: {ai_used}")
        return jsonify(data)

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        print(f"[*] Unhandled error after {elapsed}s: {e}")
        return jsonify({"error": "Server error. Please try again.", "detail": str(e), "ProcessingTime": elapsed}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "nlp_loaded": nlp_processor is not None,
        "gemini_configured": bool(GOOGLE_API_KEY),
        "primary_model": GEMINI_MODELS[0][0],
    })

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("[*] Resume Optimizer — Production Backend")
    print("=" * 55 + "\n")
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)