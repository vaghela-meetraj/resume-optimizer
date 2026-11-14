import json
import os
import re


def _load_skills_map():
    skills_map = {
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
        'c plus plus': 'cpp', 'c++': 'c++',
        'c sharp': 'csharp', 'c#': 'c#'
    }
    # Check data/skills.json
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "skills.json"),
        os.path.join("data", "skills.json")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        canonical = item.get("canonical", "").strip().lower()
                        for alias in item.get("aliases", []):
                            alias_clean = re.sub(r'[^\w\s+#.]', '', alias.lower().strip())
                            skills_map[alias_clean] = canonical
                            # also without dots/punctuation for fallback
                            plain = re.sub(r'[^\w\s]', '', alias.lower().strip())
                            if plain:
                                skills_map[plain] = canonical
                break
            except Exception:
                pass
    return skills_map


SKILLS_MAP = _load_skills_map()


def normalize_skill(skill: str) -> str:
    raw = skill.lower().strip()
    # Check direct match first (preserves ++ and #)
    if raw in SKILLS_MAP:
        return SKILLS_MAP[raw]
    
    clean_plus = re.sub(r'[^\w\s+#.]', '', raw)
    if clean_plus in SKILLS_MAP:
        return SKILLS_MAP[clean_plus]

    normalized = re.sub(r'[^\w\s]', '', raw)
    return SKILLS_MAP.get(normalized, normalized)


SQL_DATABASES = {'mysql', 'postgresql', 'postgres', 'sqlite', 'sqlserver', 'mariadb', 'oracle'}


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
        elif norm_jd_skill == 'sql' and any(db in norm_resume for db in SQL_DATABASES):
            # Relational database proficiency implies SQL language capability
            exact_matches.add(orig_jd_skill)
            matched = True
        elif norm_jd_skill in SQL_DATABASES and 'sql' in norm_resume:
            # General SQL proficiency counts as partial credit for specific DB engines
            partial_matches.add(orig_jd_skill)
            matched = True
        else:
            for norm_resume_skill in norm_resume.keys():
                if (len(norm_jd_skill) > 3 and re.search(rf'\b{re.escape(norm_jd_skill)}\b', norm_resume_skill)) or \
                   (len(norm_resume_skill) > 3 and re.search(rf'\b{re.escape(norm_resume_skill)}\b', norm_jd_skill)):
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
