import json
import logging
import os
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_resources() -> Dict[str, Any]:
    """Load curated skill resources, prerequisites, and verified links."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "resources.json"),
        os.path.join("data", "resources.json")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed loading resources.json from {p}: {e}")
    return {}


RESOURCES_DATA = load_resources()


def topological_sort_skills(skills: List[str], resources: Dict[str, Any]) -> List[str]:
    """
    Topologically sort a list of missing skills based on prerequisite dependencies.
    If Skill A is a prerequisite for Skill B (e.g., Docker for Kubernetes),
    Skill A will strictly appear before Skill B in the sorted output.
    """
    skill_set = set(skills)
    skill_case_map = {s.lower(): s for s in skills}

    # Build adjacency list and in-degree counts restricted to current missing skills
    graph = defaultdict(set)
    in_degree = {s: 0 for s in skills}

    for skill in skills:
        meta = resources.get(skill) or resources.get(skill_case_map.get(skill.lower(), skill), {})
        prereqs = meta.get("prerequisites", [])
        for p in prereqs:
            # Check if this prerequisite is also missing in our target list
            canon_p = skill_case_map.get(p.lower())
            if canon_p and canon_p in skill_set and canon_p != skill:
                # canon_p -> skill
                if skill not in graph[canon_p]:
                    graph[canon_p].add(skill)
                    in_degree[skill] += 1

    # Kahn's algorithm for topological sorting
    queue = deque([s for s in skills if in_degree[s] == 0])
    sorted_skills = []

    while queue:
        node = queue.popleft()
        sorted_skills.append(node)
        for neighbor in sorted(graph[node]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # In case of cycles, append any remaining nodes
    for s in skills:
        if s not in sorted_skills:
            sorted_skills.append(s)

    return sorted_skills


def build_learning_roadmap(
    missing_skills: List[str],
    skill_weights: Optional[Dict[str, float]] = None,
    max_skills: int = 6
) -> List[Dict[str, Any]]:
    """
    Constructs a personalized, prioritized learning roadmap:
    1. Computes priority: JD Weight * (1 + number of dependent missing skills).
    2. Topologically sorts skills so foundational tools (Docker, SQL, Python) precede advanced tools (Kubernetes, Kafka).
    3. Returns up to max_skills (default 6) with verified documentation and estimated hours.
    """
    if not missing_skills:
        return []

    resources = RESOURCES_DATA or load_resources()
    weights = skill_weights or {}

    skill_set = set(missing_skills)
    skill_case_map = {s.lower(): s for s in missing_skills}

    # Calculate dependency count: how many other missing skills depend on this skill?
    dependency_counts = {s: 0 for s in missing_skills}
    for s in missing_skills:
        meta = resources.get(s) or resources.get(skill_case_map.get(s.lower(), s), {})
        for prereq in meta.get("prerequisites", []):
            canon_prereq = skill_case_map.get(prereq.lower())
            if canon_prereq and canon_prereq in dependency_counts:
                dependency_counts[canon_prereq] += 1

    # Calculate raw priority scores
    raw_priorities = {}
    for s in missing_skills:
        base_weight = weights.get(s, 1.0)
        dep_multiplier = 1 + dependency_counts.get(s, 0)
        raw_priorities[s] = base_weight * dep_multiplier

    # Step 1: Select top skills by raw priority
    top_candidates = sorted(missing_skills, key=lambda s: raw_priorities.get(s, 1.0), reverse=True)[:max_skills]

    # Step 2: Topologically sort selected skills so prerequisites are completed first
    ordered_skills = topological_sort_skills(top_candidates, resources)

    roadmap = []
    for rank, s in enumerate(ordered_skills, 1):
        meta = resources.get(s) or resources.get(skill_case_map.get(s.lower(), s))
        if meta:
            roadmap.append({
                "rank": rank,
                "skill": s,
                "why": meta.get("why", "Key skill required by the target job description."),
                "est_hours": meta.get("est_hours", 8),
                "prerequisites": [p for p in meta.get("prerequisites", []) if p.lower() in skill_case_map],
                "priority_score": round(raw_priorities.get(s, 1.0), 2),
                "links": meta.get("links", [])
            })
        else:
            roadmap.append({
                "rank": rank,
                "skill": s,
                "why": "Directly requested skill in target job requirements.",
                "est_hours": 8,
                "prerequisites": [],
                "priority_score": round(raw_priorities.get(s, 1.0), 2),
                "links": [
                    {"title": f"{s} Official Documentation", "url": f"https://www.google.com/search?q={s}+official+documentation"}
                ]
            })

    return roadmap

