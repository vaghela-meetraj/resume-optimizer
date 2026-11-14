import re
from .skills import normalize_skill


REQUIRED_PATTERNS = [
    r'(?i)\b(requirements?|qualifications?|must[\s_-]?haves?|what\s+you(\'ll)?\s+need|core\s+skills?|minimum\s+qualifications?|what\s+you\s+bring)\b'
]

PREFERRED_PATTERNS = [
    r'(?i)\b(nice[\s_-]?to[\s_-]?haves?|preferred(\s+qualifications?)?|bonus|optional|good\s+to\s+have|plus)\b'
]


def split_jd_sections(jd_text: str) -> dict:
    """Split JD text into required, preferred, and general sections based on headings."""
    lines = jd_text.splitlines()
    required_lines = []
    preferred_lines = []
    general_lines = []
    
    current_section = "general"
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        # Check if line acts as a header (ends with colon or short line)
        is_header = len(stripped) < 60 and (stripped.endswith(':') or stripped.isupper() or len(stripped.split()) <= 4)
        
        if is_header:
            if any(re.search(p, stripped) for p in PREFERRED_PATTERNS):
                current_section = "preferred"
                continue
            elif any(re.search(p, stripped) for p in REQUIRED_PATTERNS):
                current_section = "required"
                continue
                
        if current_section == "required":
            required_lines.append(stripped)
        elif current_section == "preferred":
            preferred_lines.append(stripped)
        else:
            general_lines.append(stripped)
            
    has_sections = bool(required_lines or preferred_lines)
    return {
        "has_sections": has_sections,
        "required_text": " ".join(required_lines),
        "preferred_text": " ".join(preferred_lines),
        "general_text": " ".join(general_lines)
    }


SQL_DATABASES = {'mysql', 'postgresql', 'postgres', 'sqlite', 'sqlserver', 'mariadb', 'oracle'}


def calculate_weighted_score(
    resume_skills: set,
    jd_skills: set,
    jd_text: str,
    skill_categories: dict = None
) -> dict:
    """
    Scoring v2:
    - Required skill weight = 1.0 (or 0.75 if no sections)
    - Preferred skill weight = 0.5 (or 0.75 if no sections)
    - Repeated in JD adds up to +0.5 weight (capped at 1.5 total)
    - Exact match = 1.0 credit, Partial match = 0.5 credit
    - Score = (weighted_matched / weighted_total) * 100, no artificial 95 cap
    """
    if not jd_skills:
        return {
            "score": 0,
            "score_version": 2,
            "weighted_matched": 0.0,
            "weighted_total": 0.0,
            "required": {"matched": 0, "total": 0},
            "preferred": {"matched": 0, "total": 0},
            "categories": {}
        }

    sections = split_jd_sections(jd_text)
    req_text = sections["required_text"].lower()
    pref_text = sections["preferred_text"].lower()
    full_jd_lower = jd_text.lower()
    
    norm_resume = {normalize_skill(s): s for s in resume_skills}
    norm_jd = {normalize_skill(s): s for s in jd_skills}
    
    weighted_total = 0.0
    weighted_matched = 0.0
    
    req_total, req_matched = 0, 0
    pref_total, pref_matched = 0, 0
    
    for norm_skill, orig_skill in norm_jd.items():
        # Determine base weight
        if sections["has_sections"]:
            if norm_skill in req_text or orig_skill.lower() in req_text:
                base_weight = 1.0
                is_req = True
            elif norm_skill in pref_text or orig_skill.lower() in pref_text:
                base_weight = 0.5
                is_req = False
            else:
                base_weight = 1.0
                is_req = True
        else:
            base_weight = 0.75
            is_req = None
            
        # Repetition boost: count occurrences in JD
        occurrences = len(re.findall(rf'\b{re.escape(norm_skill)}\b', full_jd_lower))
        repetition_bonus = min(max(0, (occurrences - 1) * 0.25), 0.5)
        skill_weight = min(base_weight + repetition_bonus, 1.5)
        
        weighted_total += skill_weight
        
        # Track counts
        if is_req is True:
            req_total += 1
        elif is_req is False:
            pref_total += 1
            
        # Match credit: exact = 1.0, partial = 0.5
        credit = 0.0
        if norm_skill in norm_resume:
            credit = 1.0
        elif norm_skill == 'sql' and any(db in norm_resume for db in SQL_DATABASES):
            credit = 1.0
        elif norm_skill in SQL_DATABASES and 'sql' in norm_resume:
            credit = 0.5
        else:
            for resume_norm in norm_resume.keys():
                if (len(norm_skill) > 3 and re.search(rf'\b{re.escape(norm_skill)}\b', resume_norm)) or \
                   (len(resume_norm) > 3 and re.search(rf'\b{re.escape(resume_norm)}\b', norm_skill)):
                    credit = 0.5
                    break
                    
        weighted_matched += credit * skill_weight
        
        if credit > 0:
            if is_req is True:
                req_matched += 1
            elif is_req is False:
                pref_matched += 1

    final_score = round((weighted_matched / weighted_total) * 100) if weighted_total > 0 else 0
    # No artificial cap at 95%
    final_score = min(max(final_score, 0), 100)
    
    # Category breakdown
    categories_breakdown = {}
    if skill_categories:
        for cat_name, cat_skills in skill_categories.items():
            cat_matched = []
            cat_missing = []
            for s in cat_skills:
                norm_s = normalize_skill(s)
                if norm_s in norm_jd:
                    if norm_s in norm_resume:
                        cat_matched.append(orig_skill if norm_s == norm_skill else s)
                    else:
                        cat_missing.append(orig_skill if norm_s == norm_skill else s)
            total_cat = len(cat_matched) + len(cat_missing)
            if total_cat > 0:
                categories_breakdown[cat_name] = {
                    "matched": sorted(list(set(cat_matched))),
                    "missing": sorted(list(set(cat_missing))),
                    "total": total_cat,
                    "match_rate": round((len(cat_matched) / total_cat) * 100)
                }

    return {
        "score": final_score,
        "score_version": 2,
        "weighted_matched": round(weighted_matched, 2),
        "weighted_total": round(weighted_total, 2),
        "required": {"matched": req_matched, "total": req_total},
        "preferred": {"matched": pref_matched, "total": pref_total},
        "categories": categories_breakdown
    }

