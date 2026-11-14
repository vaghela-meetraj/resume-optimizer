import json
import logging
import re
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fence blocks if present."""
    if not text:
        return ""
    text = text.strip()
    # Remove leading ```json or ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    # Remove trailing ```
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_substring(text: str) -> str:
    """Find the outermost JSON array or object substring."""
    first_bracket = text.find("[")
    first_brace = text.find("{")

    if first_bracket == -1 and first_brace == -1:
        return text

    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        # Starts with array
        last_bracket = text.rfind("]")
        if last_bracket != -1 and last_bracket > first_bracket:
            return text[first_bracket : last_bracket + 1]
    elif first_brace != -1:
        # Starts with object
        last_brace = text.rfind("}")
        if last_brace != -1 and last_brace > first_brace:
            return text[first_brace : last_brace + 1]

    return text


def parse_llm_projects(raw_text: str) -> List[Dict[str, Any]]:
    """
    Robust JSON parser for LLM project generation responses:
    - Strips markdown fences (```json ... ```)
    - Locates JSON objects/arrays amidst conversational text
    - Accepts {"Projects": [...]}, {"ProjectIdeas": [...]}, or bare list [...]
    - Normalizes project fields (ProjectTitle <-> ProjectName, KeySkills <-> SkillsPracticed, ResumeBullets)
    - Returns normalized List[Dict[str, Any]]
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty LLM response received.")

    cleaned = strip_markdown_fences(raw_text)
    candidate_str = extract_json_substring(cleaned)

    try:
        parsed = json.loads(candidate_str)
    except Exception:
        # Attempt minimal repair: remove trailing commas before } or ]
        repaired = re.sub(r",\s*([\]}])", r"\1", candidate_str)
        try:
            parsed = json.loads(repaired)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM JSON response: {e}")

    # Extract project list from various envelope shapes
    projects_list: List[Dict[str, Any]] = []

    if isinstance(parsed, list):
        projects_list = parsed
    elif isinstance(parsed, dict):
        if "Projects" in parsed and isinstance(parsed["Projects"], list):
            projects_list = parsed["Projects"]
        elif "ProjectIdeas" in parsed and isinstance(parsed["ProjectIdeas"], list):
            projects_list = parsed["ProjectIdeas"]
        elif "projects" in parsed and isinstance(parsed["projects"], list):
            projects_list = parsed["projects"]
        elif "project_ideas" in parsed and isinstance(parsed["project_ideas"], list):
            projects_list = parsed["project_ideas"]
        elif "data" in parsed and isinstance(parsed["data"], list):
            projects_list = parsed["data"]
        elif "ProjectTitle" in parsed or "ProjectName" in parsed:
            # Single project object returned
            projects_list = [parsed]
        else:
            # Check if any value in the dict is a list of dicts
            for v in parsed.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    projects_list = v
                    break

    # Normalize each project dictionary
    normalized: List[Dict[str, Any]] = []
    for item in projects_list:
        if not isinstance(item, dict):
            continue
        p = dict(item)

        # Title normalization
        title = p.get("ProjectTitle") or p.get("ProjectName") or "Production Portfolio Project"
        p["ProjectTitle"] = str(title).strip()
        p["ProjectName"] = p["ProjectTitle"]

        # Description
        p["Description"] = str(p.get("Description", "")).strip()

        # KeySkills normalization
        skills = p.get("KeySkills") or p.get("SkillsPracticed") or []
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]
        elif not isinstance(skills, list):
            skills = []
        p["KeySkills"] = [str(s).strip() for s in skills]
        p["SkillsPracticed"] = p["KeySkills"]

        # ResumeBullets normalization
        bullets = p.get("ResumeBullets") or []
        if not bullets and p.get("ResumeBullet"):
            bullets = [p["ResumeBullet"]]
        elif isinstance(bullets, str):
            bullets = [b.strip() for b in bullets.split("\n") if b.strip()]
        elif not isinstance(bullets, list):
            bullets = []
        p["ResumeBullets"] = [str(b).strip() for b in bullets]

        normalized.append(p)

    return normalized

