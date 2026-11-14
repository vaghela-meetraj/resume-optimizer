import json
import os
import re
from typing import Any, Dict, List, Optional, Set

from app.services.skills import SKILLS_MAP, normalize_skill

ALLOWED_STARTING_VERBS = {
    "Built", "Designed", "Implemented", "Automated", "Containerized",
    "Deployed", "Integrated", "Optimized", "Engineered", "Configured"
}
WEAK_STARTING_VERBS = {"Wrote", "Made", "Did", "Worked"}
OUTCOME_MARKERS = ["reducing", "eliminating", "enabling", "replacing", "cutting", "so that", "without"]


def _load_known_tools() -> Dict[str, str]:
    """Load canonical technical tools and distinctive aliases from skills.json."""
    tools: Dict[str, str] = {}
    json_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "skills.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            generic_stop = {
                "rest", "api", "apis", "web", "server", "data", "app", "ui", "ux", "cloud",
                "test", "tests", "testing", "database", "databases", "service", "services",
                "microservices", "pipeline", "pipelines", "frontend", "backend", "fullstack",
                "developer", "engineer", "lead", "code", "architecture", "security", "devops",
                "software", "programming", "system", "systems", "git", "ci", "cd", "html", "css",
                "etl", "crud", "mvc", "oop", "tdd", "bdd", "orm", "sdk", "json", "xml", "yaml",
                "csv", "http", "https", "tcp", "udp", "ssl", "tls", "ssh", "dns", "ip", "vpn",
                "cli", "gui", "spa", "pwa", "jwt", "oauth", "sso", "iam", "rbac", "sql", "nosql"
            }
            for it in items:
                if it.get("ambiguous", False):
                    continue
                canon = it.get("canonical", "").strip()
                if not canon or canon.lower() in generic_stop or len(canon) < 3:
                    continue
                tools[canon.lower()] = canon
                for alias in it.get("aliases", []):
                    a_clean = alias.strip().lower()
                    if a_clean and a_clean not in generic_stop and len(a_clean) >= 3:
                        tools[a_clean] = canon
        except Exception:
            pass
    return tools


KNOWN_TOOLS = _load_known_tools()


def skill_matches_text(skill: str, text: str) -> bool:
    """Check if skill or any of its known aliases/normalized forms appears in text with word boundaries."""
    if not skill or not text:
        return False
    text_lower = text.lower()
    norm = normalize_skill(skill)

    variants = {skill.lower().strip(), norm}
    for alias, canonical in SKILLS_MAP.items():
        if canonical == norm or alias == norm:
            variants.add(alias)
            variants.add(canonical)

    for v in variants:
        if not v:
            continue
        escaped = re.escape(v)
        pattern = r'(?:\b|(?<=\W))' + escaped + r'(?:\b|(?=\W))'
        if re.search(pattern, text_lower):
            return True
    return False


def count_sentences(text: str) -> int:
    """Split text on sentence-ending punctuation and count non-empty sentences."""
    if not text:
        return 0
    parts = [s.strip() for s in re.split(r'[.!?]+(?:\s+|$)', text.strip()) if s.strip()]
    return len(parts)


def validate_projects(
    projects: List[Dict[str, Any]],
    missing_skills: Optional[List[str]] = None,
    matched_skills: Optional[List[str]] = None,
    jd_keywords: Optional[List[str]] = None
) -> List[str]:
    """
    Validates generated project ideas:
    - Exactly 2 projects
    - ProjectTitle present and non-empty
    - Description present and non-empty (2-4 sentences)
    - KeySkills present with 3-6 skills combining missing and matching skills
    - No more than 50% overlap between Project 1 and Project 2
    """
    problems: List[str] = []

    if not isinstance(projects, list):
        return ["Output is not a list of projects."]

    # Rule 1: Exactly 2 projects
    if len(projects) != 2:
        problems.append(f"Expected exactly 2 projects, but got {len(projects)}.")

    missing_set = {normalize_skill(str(s)) for s in (missing_skills or []) if s}
    matched_set = {normalize_skill(str(s)) for s in (matched_skills or []) if s}
    all_allowed = missing_set | matched_set

    for idx, p in enumerate(projects[:2], start=1):
        if not isinstance(p, dict):
            problems.append(f"Project {idx} is not a valid dictionary object.")
            continue

        # Required key: ProjectTitle
        title = p.get("ProjectTitle") or p.get("ProjectName")
        if not title or not str(title).strip():
            problems.append(f"Project {idx} missing required key: 'ProjectTitle'.")

        # Required key: Description
        desc = str(p.get("Description", "")).strip()
        if not desc:
            problems.append(f"Project {idx} missing required key: 'Description'.")
        else:
            num_sentences = count_sentences(desc)
            if not (2 <= num_sentences <= 4):
                problems.append(
                    f"Project {idx} Description should be 2-3 sentences (tolerated 2-4), but has {num_sentences}."
                )

        # Required key: KeySkills (3-6 items combining missing and matched skills)
        skills = p.get("KeySkills") or p.get("SkillsPracticed")
        if skills is None or not isinstance(skills, list):
            problems.append(f"Project {idx} missing required key: 'KeySkills'.")
        else:
            if not (3 <= len(skills) <= 6):
                problems.append(
                    f"Project {idx} KeySkills must have between 3 and 6 items, got {len(skills)}."
                )

            if all_allowed:
                for skill in skills:
                    s_norm = normalize_skill(str(skill))
                    if s_norm not in all_allowed:
                        problems.append(
                            f"Project {idx} KeySkill '{skill}' is not in target missing or matched skills."
                        )

        # Optional Milestones validation (if present)
        if "Milestones" in p and p["Milestones"] is not None:
            milestones = p["Milestones"]
            if isinstance(milestones, list):
                if not (4 <= len(milestones) <= 6):
                    problems.append(
                        f"Project {idx} Milestones must contain 4-6 items, got {len(milestones)}."
                    )
                if milestones:
                    first_m = milestones[0]
                    first_title = first_m.get("Title", "") if isinstance(first_m, dict) else str(first_m)
                    if "setup" not in first_title.lower():
                        problems.append(
                            f"Project {idx} first milestone must contain 'setup' in title, got: '{first_title}'."
                        )

                    last_m = milestones[-1]
                    last_title = last_m.get("Title", "") if isinstance(last_m, dict) else str(last_m)
                    last_lower = last_title.lower()
                    if "demo" not in last_lower and "document" not in last_lower:
                        problems.append(
                            f"Project {idx} last milestone must contain 'demo' or 'document' in title, got: '{last_title}'."
                        )

    # Project 1 and Project 2 KeySkills overlap check
    if len(projects) >= 2 and isinstance(projects[0], dict) and isinstance(projects[1], dict):
        s1 = {normalize_skill(str(s)) for s in (projects[0].get("KeySkills") or projects[0].get("SkillsPracticed") or []) if s}
        s2 = {normalize_skill(str(s)) for s in (projects[1].get("KeySkills") or projects[1].get("SkillsPracticed") or []) if s}
        if s1 and s2:
            overlap = s1 & s2
            max_size = max(len(s1), len(s2))
            overlap_ratio = len(overlap) / max_size
            if overlap_ratio > 0.5:
                problems.append(
                    f"Project 1 and Project 2 have {overlap_ratio:.0%} KeySkills overlap (maximum 50% allowed)."
                )

    return problems


def check_candidate_four_keys(project: Dict[str, Any]) -> bool:
    """Helper to check if a project passes the core key requirement."""
    if not isinstance(project, dict):
        return False
    title = project.get("ProjectTitle") or project.get("ProjectName")
    desc = project.get("Description")
    skills = project.get("KeySkills") or project.get("SkillsPracticed")
    return bool(title and desc and skills)

