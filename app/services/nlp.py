import os
import re
import threading

import nltk
try:
    import spacy
except Exception:
    spacy = None
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
        self.nlp = None
        if spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
                self.nlp.max_length = 2_000_000
            except Exception:
                try:
                    os.system("python -m spacy download en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
                    self.nlp.max_length = 2_000_000
                except Exception:
                    self.nlp = None

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

        if self.nlp:
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
