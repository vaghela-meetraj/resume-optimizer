import json
import os
import re
import threading
import time

try:
    import spacy
    from spacy.matcher import PhraseMatcher
except Exception:
    spacy = None
    PhraseMatcher = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Comprehensive, self-contained stop words list (zero NLTK dependency)
STANDARD_STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as',
    'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot',
    'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each',
    'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d',
    'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i',
    'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s',
    'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
    'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll',
    'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs',
    'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve',
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll',
    'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while',
    'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll',
    'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
}


class NLPProcessor:
    _instance = None
    _lock = threading.Lock()
    SINGLE_CHAR_LANGUAGES = {'C', 'R'}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        t0 = time.perf_counter()
        self.stop_words = STANDARD_STOP_WORDS

        # Initialize lightweight blank English tokenizer (zero model downloads)
        self.nlp = None
        self.matcher = None
        if spacy is not None and PhraseMatcher is not None:
            try:
                self.nlp = spacy.blank("en")
                self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
                if hasattr(self.nlp, "Defaults") and hasattr(self.nlp.Defaults, "stop_words"):
                    self.stop_words = set(self.nlp.Defaults.stop_words)
            except Exception:
                self.nlp = None
                self.matcher = None

        self.alias_to_canonical = self._load_alias_to_canonical()
        self.tech_skills_patterns = self._load_tech_patterns()

        # Populate PhraseMatcher if available
        if self.nlp and self.matcher:
            skills_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "skills.json")
            if not os.path.exists(skills_path):
                skills_path = os.path.join("data", "skills.json")
            if os.path.exists(skills_path):
                try:
                    with open(skills_path, "r", encoding="utf-8") as f:
                        skills_data = json.load(f)
                    for item in skills_data:
                        canonical = item.get("canonical")
                        patterns = [self.nlp.make_doc(a) for a in item.get("aliases", []) if len(a) > 1]
                        if patterns:
                            self.matcher.add(canonical, patterns)
                except Exception:
                    pass

        self.NOISE_TERMS = {
            'experience', 'exposure', 'familiarity', 'hands', 'nice', 'preferred',
            'qualifications', 'required', 'understanding', 'skills', 'library',
            'knowledge', 'ability', 'years', 'work', 'working', 'using', 'tool',
            'tools', 'system', 'systems', 'support', 'software', 'solutions',
            'strong', 'excellent', 'good', 'proficiency', 'proficient', 'expert',
            'development', 'technologies', 'platform', 'framework', 'environment',
            'application', 'applications', 'service', 'services', 'team', 'project',
            'projects', 'demonstrated', 'extensive', 'deep', 'proven',
            'building', 'creating', 'developing', 'implementing', 'design', 'include',
            'dom', 'ajax', 'fetch', 'promises', 'async', 'callbacks', 'closures',
            'json', 'xml', 'es6', 'es2015', 'es2020', 'modules', 'hoisting',
            'flexbox', 'grid', 'animations', 'transitions', 'responsive',
            'loops', 'arrays', 'objects', 'functions', 'classes', 'inheritance',
            'api', 'apis'
        }
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[*] NLP engine initialized in {elapsed_ms:.2f}ms (lightweight, zero model downloads).")

    def _load_alias_to_canonical(self):
        alias_map = {}
        skills_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "skills.json")
        if not os.path.exists(skills_path):
            skills_path = os.path.join("data", "skills.json")
        if os.path.exists(skills_path):
            try:
                with open(skills_path, "r", encoding="utf-8") as f:
                    skills_data = json.load(f)
                for item in skills_data:
                    canon = item.get("canonical", "").strip()
                    if canon:
                        alias_map[canon.lower()] = canon
                        for alias in item.get("aliases", []):
                            alias_map[alias.lower().strip()] = canon
            except Exception as e:
                print(f"[!] Warning: Failed loading alias mapping: {e}")
        return alias_map

    def _load_tech_patterns(self=None):
        skills_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "skills.json")
        if not os.path.exists(skills_path):
            skills_path = os.path.join("data", "skills.json")

        if os.path.exists(skills_path):
            try:
                with open(skills_path, "r", encoding="utf-8") as f:
                    skills_data = json.load(f)

                cat_map = {
                    "language": "languages",
                    "framework": "frameworks",
                    "library": "libraries",
                    "database": "databases",
                    "cloud": "cloud",
                    "tool": "tools",
                    "concept": "concepts"
                }

                categorized = {v: [] for v in cat_map.values()}
                for item in skills_data:
                    cat_key = cat_map.get(item.get("category"), "concepts")
                    is_ambig = item.get("ambiguous", False)
                    for alias in item.get("aliases", []):
                        clean_alias = alias.strip()
                        escaped = re.escape(clean_alias)
                        if is_ambig and len(clean_alias) <= 5:
                            token = clean_alias if not clean_alias.islower() else clean_alias.capitalize()
                            categorized[cat_key].append(f'(?-i:{token})')
                        else:
                            categorized[cat_key].append(f'(?i:{escaped})')

                patterns = {}
                for cat, aliases in categorized.items():
                    if aliases:
                        aliases.sort(key=lambda x: len(re.sub(r'\(\?[-a-z]+:', '', x).rstrip(')')), reverse=True)
                        # Negative lookaround that handles sentence-ending punctuation (e.g. "Docker.", "Redis.")
                        patterns[cat] = r'(?<![a-zA-Z0-9+#&])(' + '|'.join(aliases) + r')(?![a-zA-Z0-9+#&]|(?:\.[a-zA-Z]))'
                return patterns
            except Exception as e:
                print(f"[!] Warning: Failed loading skills.json: {e}")

        # Fallback patterns with lookarounds
        return {
            'languages': r'(?<![a-zA-Z0-9+#&])((?i:JavaScript)|(?i:TypeScript)|(?i:Python)|(?i:Java)|(?i:C\+\+)|(?i:C#)|(?i:Ruby)|(?i:Rust)|(?i:Swift)|(?i:Kotlin)|(?i:PHP)|(?i:Scala)|(?i:MATLAB)|(?i:SQL)|(?i:HTML)|(?i:CSS)|(?i:Bash)|(?i:Shell)|(?i:Perl)|(?i:Dart)|(?-i:Go)|(?-i:R)|(?-i:C))(?![a-zA-Z0-9+#&]|(?:\.[a-zA-Z]))',
            'frameworks': r'(?<![a-zA-Z0-9+#&])((?i:React Native)|(?i:Spring Boot)|(?i:React\.js)|(?i:Angular)|(?i:FastAPI)|(?i:Express)|(?i:Next\.?js)|(?i:Laravel)|(?i:ASP\.NET)|(?i:Flutter)|(?i:SwiftUI)|(?i:Svelte)|(?i:Django)|(?i:Flask)|(?i:React)|(?i:Rails)|(?i:Nuxt)|(?i:Vue)|(?i:Node\.?js)|(?i:Gatsby)|(?i:Remix)|(?-i:Spring))(?![a-zA-Z0-9+#&]|(?:\.[a-zA-Z]))',
            'libraries': r'(?<![a-zA-Z0-9+#&])((?i:Scikit-learn)|(?i:Material-UI)|(?i:TensorFlow)|(?i:Bootstrap)|(?i:Matplotlib)|(?i:Tailwind)|(?i:PyTorch)|(?i:OpenCV)|(?i:Pandas)|(?i:NumPy)|(?i:Seaborn)|(?i:Redux)|(?i:Zustand)|(?i:spaCy)|(?i:jQuery)|(?i:NLTK)|(?i:Jest)|(?i:Keras)|(?i:RxJS)|(?i:Recoil))(?![a-zA-Z0-9+#&]|(?:\.[a-zA-Z]))',
            'databases': r'(?<![a-zA-Z0-9+#&])((?i:PostgreSQL)|(?i:SQL Server)|(?i:Elasticsearch)|(?i:DynamoDB)|(?i:Firebase)|(?i:Cassandra)|(?i:MongoDB)|(?i:MariaDB)|(?i:Supabase)|(?i:SQLite)|(?i:MySQL)|(?i:Redis)|(?i:Prisma)|(?i:Neo4j)|(?-i:Oracle))(?![a-zA-Z0-9+#&]|(?:\.[a-zA-Z]))',
            'cloud': r'(?<![a-zA-Z0-9+#&])((?i:Google Cloud Platform)|(?i:CloudFormation)|(?i:Google Cloud)|(?i:Kubernetes)|(?i:Terraform)|(?i:Ansible)|(?i:Jenkins)|(?i:Netlify)|(?i:Docker)|(?i:CI/CD)|(?i:Heroku)|(?i:Vercel)|(?i:Azure)|(?i:EC2)|(?i:ECS)|(?i:EKS)|(?i:GCP)|(?i:AWS)|(?i:RDS)|(?i:S3)|(?-i:Lambda))(?![a-zA-Z0-9+#&]|(?:\.[a-zA-Z]))',
            'tools': r'(?<![a-zA-Z0-9+#&])((?i:Visual Studio Code)|(?i:GitHub Actions)|(?i:Confluence)|(?i:Bitbucket)|(?i:Postman)|(?i:Swagger)|(?i:Webpack)|(?i:IntelliJ)|(?i:Eclipse)|(?i:VS Code)|(?i:GitLab)|(?i:GitHub)|(?i:Gradle)|(?i:Maven)|(?i:Babel)|(?i:Figma)|(?i:JIRA)|(?i:Vite)|(?i:yarn)|(?i:npm)|(?i:Git))(?![a-zA-Z0-9+#&]|(?:\.[a-zA-Z]))',
            'concepts': r'(?<![a-zA-Z0-9+#&])((?i:Natural Language Processing)|(?i:Computer Vision)|(?i:Machine Learning)|(?i:Deep Learning)|(?i:Microservices)|(?i:Serverless)|(?i:GraphQL)|(?i:RESTful)|(?i:REST API)|(?i:OAuth)|(?i:DevOps)|(?-i:SOLID)|(?i:Agile)|(?i:Scrum)|(?i:REST)|(?i:TDD)|(?i:BDD)|(?i:JWT)|(?i:OOP)|(?i:MVC))(?![a-zA-Z0-9+#&]|(?:\.[a-zA-Z]))'
        }

    def extract_skills_nlp(self, text: str) -> set:
        canonical_skills = set()
        alias_map = getattr(self, "alias_to_canonical", {})

        # Step 1: Pre-compiled lookaround regex patterns (sub-millisecond)
        for category, pattern in self.tech_skills_patterns.items():
            matches = re.findall(pattern, text)
            for m in matches:
                clean_m = m.strip()
                if (len(clean_m) > 1 or clean_m in self.SINGLE_CHAR_LANGUAGES) and clean_m.lower() not in self.NOISE_TERMS:
                    canon = alias_map.get(clean_m.lower(), clean_m)
                    canonical_skills.add(canon)

        # Step 2: PhraseMatcher extraction (if spaCy is available)
        if self.nlp and self.matcher:
            try:
                doc = self.nlp(text[:50_000])
                matches = self.matcher(doc)
                for match_id, start, end in matches:
                    matched_canonical = self.nlp.vocab.strings[match_id]
                    canon = alias_map.get(matched_canonical.lower(), matched_canonical)
                    canonical_skills.add(canon)
            except Exception:
                pass

        # Step 3: Noise filtering
        return {s for s in canonical_skills if (len(s) > 1 or s in self.SINGLE_CHAR_LANGUAGES) and s.lower() not in self.NOISE_TERMS}

    def calculate_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        try:
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3), stop_words='english', lowercase=True)
            vectors = vectorizer.fit_transform([resume_text, jd_text])
            return float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100)
        except Exception as e:
            print(f"⚠️ Similarity calculation failed: {e}")
            return 0.0
