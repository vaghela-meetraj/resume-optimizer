import re


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
