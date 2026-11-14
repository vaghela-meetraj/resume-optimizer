import json
import pytest
from typing import Any, Dict, List

from app.llm.base import LLMClient, LLMError, ProjectIdeas, RateLimited
from app.llm.chain import LLMChain
from app.llm.parser import parse_llm_projects
from app.llm.validator import check_candidate_four_keys, count_sentences, validate_projects


def make_valid_project(
    title: str = "Distributed Task Queue",
    desc: str = "Build a scalable asynchronous job worker for clinic receptionist intake. It processes background queues efficiently and reliably. The demo shows job status.",
    skills: List[str] = None,
    bullets: List[str] = None,
    tech_stack: List[str] = None,
    built_on: List[str] = None
) -> Dict[str, Any]:
    skills = skills or ["docker", "fastapi", "redis"]
    p = {
        "Domain": "healthcare/clinic ops",
        "TargetUser": "a 3-person clinic receptionist",
        "Problem": "Patient intake forms are slow and manual.",
        "ProjectTitle": title,
        "Description": desc,
        "KeySkills": skills
    }
    if bullets is not None:
        p["ResumeBullets"] = bullets
    return p


class MockLLMClient(LLMClient):
    """Configurable mock client for testing provider interactions."""

    def __init__(self, name: str, responses: List[Any] = None):
        self.name = name
        self.responses = responses or []
        self.call_count = 0
        self.received_prompts = []

    def generate_json(self, prompt: str, schema_class: type = ProjectIdeas, max_tokens: int = 4000, timeout: float = 15.0):
        self.call_count += 1
        self.received_prompts.append(prompt)

        if not self.responses:
            raise LLMError(f"{self.name} out of configured mock responses.")

        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp

        projects = resp if isinstance(resp, list) else resp.get("Projects", resp.get("ProjectIdeas", []))
        return {
            "data": {
                "Projects": projects,
                "ProjectIdeas": projects
            },
            "raw_projects": projects,
            "tokens_used": 100,
            "latency_ms": 30.0,
            "provider": self.name,
            "model": f"{self.name}-test"
        }


# =========================================================================
# STEP 5: TESTS
# =========================================================================

def test_groq_valid_gemini_never_called():
    """Requirement 1: Groq valid, so Gemini is never called."""
    p1 = make_valid_project(
        title="Event Pipeline Worker",
        skills=["docker", "fastapi", "redis"]
    )
    p2 = make_valid_project(
        title="Streaming Analytics Store",
        skills=["kafka", "postgresql", "kubernetes"]
    )
    mock_groq = MockLLMClient("groq", responses=[[p1, p2]])
    mock_gemini = MockLLMClient("gemini", responses=[])

    chain = LLMChain(providers=["groq", "gemini"])
    chain._adapters["groq"] = mock_groq
    chain._adapters["gemini"] = mock_gemini

    res = chain.execute(
        "Generate projects",
        missing_skills=["docker", "fastapi", "redis", "kafka", "postgresql", "kubernetes"],
        matched_skills=["python"]
    )

    assert res["provider"] == "groq"
    assert mock_groq.call_count == 1
    assert mock_gemini.call_count == 0


def test_groq_invalid_once_then_valid_on_retry():
    """Requirement 2: Groq invalid once, then valid on retry."""
    # Attempt 1: Invalid project (alien skill not in missing or matched)
    invalid_p1 = make_valid_project(
        title="Event Pipeline Worker",
        skills=["docker", "quantum-computing", "redis"]
    )
    invalid_p2 = make_valid_project(
        title="Streaming Analytics Store",
        skills=["kafka", "postgresql", "kubernetes"]
    )

    # Attempt 2: Fixed valid project
    valid_p1 = make_valid_project(
        title="Event Pipeline Worker",
        skills=["docker", "fastapi", "redis"]
    )
    valid_p2 = make_valid_project(
        title="Streaming Analytics Store",
        skills=["kafka", "postgresql", "kubernetes"]
    )

    mock_groq = MockLLMClient("groq", responses=[[invalid_p1, invalid_p2], [valid_p1, valid_p2]])
    mock_gemini = MockLLMClient("gemini", responses=[])

    chain = LLMChain(providers=["groq", "gemini"])
    chain._adapters["groq"] = mock_groq
    chain._adapters["gemini"] = mock_gemini

    res = chain.execute(
        "Generate projects",
        missing_skills=["docker", "fastapi", "redis", "kafka", "postgresql", "kubernetes"],
        matched_skills=["python"]
    )

    assert res["provider"] == "groq"
    assert mock_groq.call_count == 2
    assert mock_gemini.call_count == 0
    # Check that the retry prompt contained the problem description
    assert "Your previous output had these problems:" in mock_groq.received_prompts[1]
    assert "quantum-computing" in mock_groq.received_prompts[1]


def test_groq_raises_429_gemini_is_used():
    """Requirement 3: Groq raises or returns a 429, so Gemini is used."""
    mock_groq = MockLLMClient("groq", responses=[RateLimited("Rate limit 429", retry_after=60, provider="groq")])
    valid_p1 = make_valid_project(skills=["docker", "fastapi", "redis"])
    valid_p2 = make_valid_project(title="P2", skills=["kafka", "postgresql", "kubernetes"])
    mock_gemini = MockLLMClient("gemini", responses=[[valid_p1, valid_p2]])

    chain = LLMChain(providers=["groq", "gemini"])
    chain._adapters["groq"] = mock_groq
    chain._adapters["gemini"] = mock_gemini

    res = chain.execute(
        "Generate projects",
        missing_skills=["docker", "fastapi", "redis", "kafka", "postgresql", "kubernetes"]
    )

    assert res["provider"] == "gemini"
    assert mock_groq.call_count == 1
    assert mock_gemini.call_count == 1
    assert chain.is_in_cooldown("groq") is True


def test_both_invalid_best_candidate_used():
    """Requirement 4a: Both return invalid output, so best 4-key candidate is used."""
    # Groq returns output with invalid sentence count (e.g. 10 sentences), but has all 4 keys
    long_desc = "Sentence. " * 8
    p1 = make_valid_project(title="Candidate P1", desc=long_desc, skills=["docker", "fastapi", "redis"])
    p2 = make_valid_project(title="Candidate P2", desc=long_desc, skills=["kafka", "postgresql", "kubernetes"])

    mock_groq = MockLLMClient("groq", responses=[[p1, p2], [p1, p2]])
    # Gemini throws an error
    mock_gemini = MockLLMClient("gemini", responses=[LLMError("Gemini server failure")])

    chain = LLMChain(providers=["groq", "gemini"])
    chain._adapters["groq"] = mock_groq
    chain._adapters["gemini"] = mock_gemini

    res = chain.execute(
        "Generate projects",
        missing_skills=["docker", "fastapi", "redis", "kafka", "postgresql", "kubernetes"]
    )

    # Groq candidate was saved because it passed the 4-key check
    assert res["provider"] == "groq"
    assert len(res["data"]["Projects"]) == 2
    assert res["data"]["Projects"][0]["ProjectTitle"] == "Candidate P1"


def test_both_invalid_no_candidate_static_fallback_used():
    """Requirement 4b: Both providers fail completely, so static fallback is used."""
    mock_groq = MockLLMClient("groq", responses=[LLMError("Groq 500"), LLMError("Groq 500 retry")])
    mock_gemini = MockLLMClient("gemini", responses=[LLMError("Gemini 500")])

    chain = LLMChain(providers=["groq", "gemini"])
    chain._adapters["groq"] = mock_groq
    chain._adapters["gemini"] = mock_gemini

    res = chain.execute("Generate projects")

    assert res["provider"] == "template"
    assert len(res["data"]["Projects"]) == 2
    assert "Clinic Patient Intake" in res["data"]["Projects"][0]["ProjectTitle"]


def test_parser_accepts_all_shapes():
    """Requirement 5: Parser accepts {"Projects": [...]}, a bare list, and old 4-key projects."""
    p1 = {
        "ProjectTitle": "Telemetry Engine",
        "Description": "Build a telemetry collector. It handles high rate telemetry.",
        "KeySkills": ["Docker", "Go", "Redis"],
        "ResumeBullets": ["Bullet 1 [X].", "Bullet 2 [X].", "Bullet 3 [X]."]
    }

    # Shape 1: {"Projects": [...]}
    json_projects = json.dumps({"Projects": [p1, p1]})
    parsed1 = parse_llm_projects(json_projects)
    assert len(parsed1) == 2
    assert parsed1[0]["ProjectTitle"] == "Telemetry Engine"
    assert parsed1[0]["ProjectName"] == "Telemetry Engine"
    assert len(parsed1[0]["ResumeBullets"]) == 3

    # Shape 2: Bare list [...]
    json_bare = json.dumps([p1, p1])
    parsed2 = parse_llm_projects(json_bare)
    assert len(parsed2) == 2
    assert parsed2[0]["ProjectTitle"] == "Telemetry Engine"

    # Shape 3: Markdown fence + old 4-key keys (ProjectName, SkillsPracticed, ResumeBullet)
    old_format_raw = """```json
    {
        "ProjectIdeas": [
            {
                "ProjectName": "Legacy Dashboard",
                "Description": "Old service system. Handles analytics.",
                "SkillsPracticed": ["Python", "Flask", "SQL"],
                "ResumeBullet": "Built [X] systems with [X]% latency drop."
            },
            {
                "ProjectName": "Pipeline Service",
                "Description": "Data processing. Integrates databases.",
                "SkillsPracticed": ["Kafka", "Docker", "Go"],
                "ResumeBullet": "Processed [X] streams."
            }
        ]
    }
    ```"""
    parsed3 = parse_llm_projects(old_format_raw)
    assert len(parsed3) == 2
    assert parsed3[0]["ProjectTitle"] == "Legacy Dashboard"
    assert parsed3[0]["KeySkills"] == ["Python", "Flask", "SQL"]
    assert "[X]" in parsed3[0]["ResumeBullets"][0]


def test_validator_catches_all_rules():
    """Validator catches: missing required keys, KeySkill not in missing/matched skills, wrong project count."""
    missing = ["docker", "fastapi", "redis"]
    matched = ["python"]

    # 1. Missing ProjectTitle
    p_no_title = [
        {"Description": "Some valid description text here for testing.", "KeySkills": ["docker", "fastapi", "redis"]},
        make_valid_project(title="P2", skills=["docker", "fastapi", "redis"])
    ]
    prob1 = validate_projects(p_no_title, missing_skills=missing, matched_skills=matched)
    assert any("missing required key: 'ProjectTitle'" in p for p in prob1)

    # 2. KeySkill not in missing or matched skills
    p_alien_skill = [
        make_valid_project(skills=["docker", "quantum-computing", "redis"]),
        make_valid_project(title="P2", skills=["docker", "fastapi", "redis"])
    ]
    prob2 = validate_projects(p_alien_skill, missing_skills=missing, matched_skills=matched)
    assert any("quantum-computing" in p and "not in target missing or matched skills" in p for p in prob2)

    # 3. Wrong project count (e.g. 1 project)
    prob3 = validate_projects([make_valid_project()], missing_skills=missing, matched_skills=matched)
    assert any("Expected exactly 2 projects" in p for p in prob3)

    # 4. Overlap > 50% between Project 1 and Project 2
    p_high_overlap = [
        make_valid_project(title="P1", skills=["docker", "fastapi", "redis"]),
        make_valid_project(title="P2", skills=["docker", "fastapi", "redis"])
    ]
    prob4 = validate_projects(p_high_overlap, missing_skills=missing, matched_skills=matched)
    assert any("overlap" in p.lower() for p in prob4)


def test_project_selection_rules_in_prompt_and_schema():
    """Verify that build_prompt includes PROJECT SELECTION RULES and schema fields Domain, TargetUser, Problem."""
    from app.services.projects import build_prompt

    prompt = build_prompt(
        missing_skills=["FastAPI", "Redis"],
        matched_skills=["Python"],
        job_description="Junior Backend Engineer"
    )

    # Selection rules check
    assert "PROJECT SELECTION RULES (follow in this order)" in prompt
    assert '"TargetUser"' in prompt
    assert '"Problem"' in prompt
    assert '"Domain"' in prompt
    assert "personal finance" in prompt
    assert "healthcare/clinic ops" in prompt
    assert "Do NOT use these words in titles:" in prompt
    assert "The Description's first sentence must restate the Problem for the TargetUser." in prompt

    # Verify project with Domain, TargetUser, Problem parses and validates cleanly
    sample_json = json.dumps({
        "Projects": [
            {
                "Domain": "healthcare/clinic ops",
                "TargetUser": "a 3-person clinic receptionist",
                "Problem": "Patient intake forms are manually tracked on paper.",
                "ProjectTitle": "Clinic Intake Assistant",
                "Description": "A 3-person clinic receptionist struggles with manual paper intake forms that cause lost records. The architecture links FastAPI with Redis and PostgreSQL. The demo shows digital intake validation.",
                "KeySkills": ["FastAPI", "Redis", "PostgreSQL"],
                "TechStack": ["FastAPI", "Redis", "PostgreSQL", "Python"],
                "BuiltOn": ["Python"],
                "ResumeBullets": [
                    "Engineered asynchronous intake system handling [X] requests/sec using FastAPI, cutting patient lookup delays by [X]%.",
                    "Implemented Redis caching layer storing patient records, reducing database query retrieval latency by [X]% without dropped packets.",
                    "Automated PostgreSQL indexed database backups and validation workflows, eliminating data entry errors for [X] receptionists so that clinic workflows succeed."
                ]
            },
            {
                "Domain": "personal finance",
                "TargetUser": "a freelance designer",
                "Problem": "Invoice tracking across spreadsheets leads to missed payments.",
                "ProjectTitle": "Freelance Invoice Tracker",
                "Description": "A freelance designer frequently misses payment deadlines due to spreadsheet tracking. The architecture streams webhook invoice updates into Python workers. The demo displays automatic invoice status alerts.",
                "KeySkills": ["Docker", "Python", "Kafka"],
                "TechStack": ["Docker", "Python", "Kafka"],
                "BuiltOn": ["Python"],
                "ResumeBullets": [
                    "Engineered invoice tracker containerized with Docker, reducing invoice dispatch latency by [X]% for [X] freelance clients.",
                    "Automated reminders using Python background tasks, improving client payment collection rates by [X]% without manual tracking.",
                    "Integrated Kafka event streaming pipeline handling [X] webhook events, cutting reconciliation errors by [X]% so that freelancers stay paid."
                ]
            }
        ]
    })
    parsed = parse_llm_projects(sample_json)
    assert len(parsed) == 2
    assert parsed[0]["Domain"] == "healthcare/clinic ops"
    assert parsed[0]["TargetUser"] == "a 3-person clinic receptionist"
    assert parsed[0]["Problem"] == "Patient intake forms are manually tracked on paper."
    problems = validate_projects(parsed)
    assert problems == []


# =========================================================================
# VALIDATOR CORE RULES TESTS (Bullet-free & Skill combination)
# =========================================================================

def test_validator_accepts_combination_of_missing_and_matching_skills():
    """KeySkills allows a combination of missing and matching skills."""
    missing = ["fastapi", "redis", "docker"]
    matched = ["python", "postgresql"]

    # Project with 2 missing and 1 matched skill
    p1 = make_valid_project(title="P1", skills=["fastapi", "redis", "python"])
    p2 = make_valid_project(title="P2", skills=["docker", "postgresql", "fastapi"])
    probs = validate_projects([p1, p2], missing_skills=missing, matched_skills=matched)
    assert not any("not in target missing or matched skills" in p for p in probs)


def test_validator_rejects_missing_title():
    """Project missing ProjectTitle is flagged."""
    p1 = {"Description": "Build a scalable asynchronous job worker. Processes background queues.", "KeySkills": ["docker", "fastapi", "redis"]}
    p2 = make_valid_project(title="P2", skills=["kafka", "postgresql", "kubernetes"])
    probs = validate_projects([p1, p2])
    assert any("missing required key: 'ProjectTitle'" in p for p in probs)


def test_validator_rejects_missing_description():
    """Project missing Description is flagged."""
    p1 = {"ProjectTitle": "Task Queue", "KeySkills": ["docker", "fastapi", "redis"]}
    p2 = make_valid_project(title="P2", skills=["kafka", "postgresql", "kubernetes"])
    probs = validate_projects([p1, p2])
    assert any("missing required key: 'Description'" in p for p in probs)


def test_validator_rejects_keyskills_count_out_of_bounds():
    """KeySkills must have between 3 and 6 items."""
    # Too few (2 items)
    p1 = make_valid_project(skills=["docker", "fastapi"])
    p2 = make_valid_project(title="P2", skills=["kafka", "postgresql", "kubernetes"])
    probs = validate_projects([p1, p2])
    assert any("between 3 and 6 items" in p for p in probs)

    # Too many (7 items)
    p1_too_many = make_valid_project(skills=["docker", "fastapi", "redis", "python", "sql", "git", "linux"])
    probs_too_many = validate_projects([p1_too_many, p2])
    assert any("between 3 and 6 items" in p for p in probs_too_many)


def test_validator_bullets_completely_optional():
    """Projects validate cleanly without any ResumeBullets."""
    p1 = {
        "ProjectTitle": "Clinic Intake Assistant",
        "Description": "A dental clinic receptionist struggles with paper forms. The architecture links FastAPI with PostgreSQL. The demo shows patient records.",
        "KeySkills": ["FastAPI", "PostgreSQL", "Docker"]
    }
    p2 = {
        "ProjectTitle": "Freelance Invoice Tracker",
        "Description": "A freelance designer misses payment deadlines. The architecture streams webhook invoice updates into Python. The demo displays invoice alerts.",
        "KeySkills": ["Kafka", "Python", "Redis"]
    }
    probs = validate_projects([p1, p2])
    assert probs == []


def test_validator_rejects_excessive_keyskills_overlap():
    """Projects sharing > 50% KeySkills are flagged."""
    p1 = make_valid_project(title="P1", skills=["docker", "fastapi", "redis"])
    p2 = make_valid_project(title="P2", skills=["docker", "fastapi", "redis"])
    probs = validate_projects([p1, p2])
    assert any("KeySkills overlap" in p for p in probs)


def test_validator_accepts_two_distinct_projects():
    """Two valid projects with distinct skills pass cleanly with 0 problems."""
    p1 = make_valid_project(title="P1", skills=["docker", "fastapi", "redis"])
    p2 = make_valid_project(title="P2", skills=["kafka", "postgresql", "kubernetes"])
    probs = validate_projects([p1, p2])
    assert probs == []


def test_project_generation_uses_groq_then_openrouter_then_gemini():
    """Verify that project generation attempts groq first, then openrouter, then gemini."""
    p1 = make_valid_project(skills=["docker", "fastapi", "redis"])
    p2 = make_valid_project(title="P2", skills=["kafka", "postgresql", "kubernetes"])

    mock_groq = MockLLMClient("groq", responses=[[p1, p2]])
    mock_openrouter = MockLLMClient("openrouter", responses=[[p1, p2]])
    mock_gemini = MockLLMClient("gemini", responses=[[p1, p2]])

    chain = LLMChain(providers=["groq", "openrouter", "gemini"])
    chain._adapters["groq"] = mock_groq
    chain._adapters["openrouter"] = mock_openrouter
    chain._adapters["gemini"] = mock_gemini

    # Groq is primary: succeeds on first call, openrouter and gemini never called
    res = chain.execute(
        "Generate projects",
        missing_skills=["docker", "fastapi", "redis", "kafka", "postgresql", "kubernetes"],
        providers=["groq", "openrouter", "gemini"]
    )

    assert mock_groq.call_count == 1
    assert mock_openrouter.call_count == 0
    assert mock_gemini.call_count == 0
    assert res["provider"] == "groq"


def test_project_generation_falls_to_openrouter_if_groq_fails():
    """If groq fails (e.g. 429), it falls through to openrouter."""
    p1 = make_valid_project(skills=["docker", "fastapi", "redis"])
    p2 = make_valid_project(title="P2", skills=["kafka", "postgresql", "kubernetes"])

    mock_groq = MockLLMClient("groq", responses=[RateLimited("429 rate limit", retry_after=60, provider="groq")])
    mock_openrouter = MockLLMClient("openrouter", responses=[[p1, p2]])
    mock_gemini = MockLLMClient("gemini", responses=[[p1, p2]])

    chain = LLMChain(providers=["groq", "openrouter", "gemini"])
    chain._adapters["groq"] = mock_groq
    chain._adapters["openrouter"] = mock_openrouter
    chain._adapters["gemini"] = mock_gemini

    res = chain.execute(
        "Generate projects",
        missing_skills=["docker", "fastapi", "redis", "kafka", "postgresql", "kubernetes"],
        providers=["groq", "openrouter", "gemini"]
    )

    assert mock_groq.call_count == 1
    assert mock_openrouter.call_count == 1
    assert mock_gemini.call_count == 0
    assert res["provider"] == "openrouter"


def test_project_generation_falls_to_gemini_if_groq_and_openrouter_fail():
    """If groq and openrouter fail, it falls through to gemini."""
    p1 = make_valid_project(skills=["docker", "fastapi", "redis"])
    p2 = make_valid_project(title="P2", skills=["kafka", "postgresql", "kubernetes"])

    mock_groq = MockLLMClient("groq", responses=[RateLimited("429 rate limit", retry_after=60, provider="groq")])
    mock_openrouter = MockLLMClient("openrouter", responses=[RateLimited("429 rate limit", retry_after=60, provider="openrouter")])
    mock_gemini = MockLLMClient("gemini", responses=[[p1, p2]])

    chain = LLMChain(providers=["groq", "openrouter", "gemini"])
    chain._adapters["groq"] = mock_groq
    chain._adapters["openrouter"] = mock_openrouter
    chain._adapters["gemini"] = mock_gemini

    res = chain.execute(
        "Generate projects",
        missing_skills=["docker", "fastapi", "redis", "kafka", "postgresql", "kubernetes"],
        providers=["groq", "openrouter", "gemini"]
    )

    assert mock_groq.call_count == 1
    assert mock_openrouter.call_count == 1
    assert mock_gemini.call_count == 1
    assert res["provider"] == "gemini"



