import pytest
from app.services.roadmap import (
    load_resources,
    topological_sort_skills,
    build_learning_roadmap
)


def test_load_resources():
    res = load_resources()
    assert isinstance(res, dict)
    assert len(res) >= 20
    assert "Docker" in res
    assert "Kubernetes" in res
    assert "SQL" in res

    docker_meta = res["Docker"]
    assert "why" in docker_meta
    assert "est_hours" in docker_meta
    assert "prerequisites" in docker_meta
    assert "links" in docker_meta
    assert len(docker_meta["links"]) >= 1


def test_topological_sort_skills():
    resources = load_resources()

    # Case 1: Docker is prerequisite for Kubernetes
    skills_in = ["Kubernetes", "Docker"]
    sorted_skills = topological_sort_skills(skills_in, resources)
    assert sorted_skills.index("Docker") < sorted_skills.index("Kubernetes")

    # Case 2: SQL is prerequisite for PostgreSQL
    skills_in = ["PostgreSQL", "SQL"]
    sorted_skills = topological_sort_skills(skills_in, resources)
    assert sorted_skills.index("SQL") < sorted_skills.index("PostgreSQL")

    # Case 3: Python is prerequisite for Celery
    skills_in = ["Celery", "Python", "Docker"]
    sorted_skills = topological_sort_skills(skills_in, resources)
    assert sorted_skills.index("Python") < sorted_skills.index("Celery")


def test_dependency_multiplier_priority():
    # When Kubernetes depends on Docker, Docker should have higher priority multiplier
    resources = load_resources()
    missing = ["Kubernetes", "Docker"]
    weights = {"Kubernetes": 1.0, "Docker": 1.0}

    roadmap = build_learning_roadmap(missing, skill_weights=weights)
    assert len(roadmap) == 2

    docker_entry = next(r for r in roadmap if r["skill"] == "Docker")
    k8s_entry = next(r for r in roadmap if r["skill"] == "Kubernetes")

    # Docker has 1 dependent ("Kubernetes"), multiplier is (1 + 1) = 2.0
    assert docker_entry["priority_score"] == 2.0
    # Kubernetes has 0 dependents, multiplier is 1.0
    assert k8s_entry["priority_score"] == 1.0

    # In final roadmap, Docker should be rank 1 because of prerequisite & priority
    assert roadmap[0]["skill"] == "Docker"
    assert roadmap[1]["skill"] == "Kubernetes"


def test_build_learning_roadmap_structure():
    missing = ["React", "TypeScript", "Node.js", "Docker", "Kubernetes", "PostgreSQL", "Redis"]
    roadmap = build_learning_roadmap(missing, max_skills=5)

    assert len(roadmap) == 5
    for idx, item in enumerate(roadmap, start=1):
        assert item["rank"] == idx
        assert "skill" in item
        assert "why" in item
        assert item["est_hours"] > 0
        assert "prerequisites" in item
        assert "priority_score" in item
        assert isinstance(item["links"], list)
        assert len(item["links"]) >= 1


def test_unknown_skill_fallback():
    # Skill not present in curated resources.json
    unknown_skill = "UnicornScriptQuantum"
    roadmap = build_learning_roadmap([unknown_skill])

    assert len(roadmap) == 1
    item = roadmap[0]
    assert item["skill"] == unknown_skill
    assert item["est_hours"] == 8
    assert len(item["links"]) >= 1
    assert "UnicornScriptQuantum" in item["links"][0]["title"]

