from .skills import normalize_skill


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
