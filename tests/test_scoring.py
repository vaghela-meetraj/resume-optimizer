import pytest
from app.services.scoring import calculate_weighted_score, split_jd_sections


def test_split_jd_sections_detects_headings():
    jd = """
    Job Description: Backend Engineer
    Requirements:
    - 3+ years experience with Python and Django.
    - Strong database skills in PostgreSQL.
    
    Nice to have:
    - Experience with Docker and Redis.
    """
    sections = split_jd_sections(jd)
    assert sections["has_sections"] is True
    assert "Python" in sections["required_text"]
    assert "PostgreSQL" in sections["required_text"]
    assert "Docker" in sections["preferred_text"]
    assert "Redis" in sections["preferred_text"]


def test_required_vs_preferred_weighting():
    jd = """
    Requirements:
    Python
    PostgreSQL
    
    Nice to have:
    Docker
    Redis
    """
    resume_skills = {"Python", "PostgreSQL", "Docker"}
    jd_skills = {"Python", "PostgreSQL", "Docker", "Redis"}
    
    result = calculate_weighted_score(resume_skills, jd_skills, jd)
    
    # Hand-computed:
    # Required: Python (1.0), PostgreSQL (1.0) -> 2.0
    # Preferred: Docker (0.5), Redis (0.5) -> 1.0
    # Total = 3.0
    # Matched = Python (1.0) + PostgreSQL (1.0) + Docker (0.5) = 2.5
    # Score = (2.5 / 3.0) * 100 = 83%
    assert result["score"] == 83
    assert result["required"]["matched"] == 2
    assert result["required"]["total"] == 2
    assert result["preferred"]["matched"] == 1
    assert result["preferred"]["total"] == 2


def test_unsectioned_jd_default_weight():
    jd = "We are seeking a developer with hands-on skills in Python, SQL, and Git."
    resume_skills = {"Python", "SQL"}
    jd_skills = {"Python", "SQL", "Git"}
    
    result = calculate_weighted_score(resume_skills, jd_skills, jd)
    
    # Hand-computed:
    # 3 skills * 0.75 weight = 2.25 total
    # 2 matched * 0.75 weight = 1.50 matched
    # Score = (1.50 / 2.25) * 100 = 67%
    assert result["score"] == 67
    assert result["weighted_total"] == 2.25
    assert result["weighted_matched"] == 1.5


def test_repetition_boost_for_emphasized_skills():
    jd = """
    Requirements:
    Python is our core language. You will write Python code and optimize Python services.
    Flask
    """
    # Python repeated 3 times -> base 1.0 + 0.5 (2 extra * 0.25) = 1.5 weight
    # Flask appears 1 time -> base 1.0 weight
    # Total = 2.5
    # Candidate with only Python gets 1.5 / 2.5 = 60%
    resume_skills = {"Python"}
    jd_skills = {"Python", "Flask"}
    
    result = calculate_weighted_score(resume_skills, jd_skills, jd)
    assert result["score"] == 60
    assert result["weighted_total"] == 2.5
    assert result["weighted_matched"] == 1.5


def test_partial_match_credit():
    jd = """
    Requirements:
    React
    Docker
    """
    # Candidate has React Native (partial match for React) and lacks Docker
    # React weight = 1.0, Docker weight = 1.0 -> Total = 2.0
    # React Native gets 0.5 * 1.0 = 0.5 credit
    # Score = (0.5 / 2.0) * 100 = 25%
    resume_skills = {"React Native"}
    jd_skills = {"React", "Docker"}
    
    result = calculate_weighted_score(resume_skills, jd_skills, jd)
    assert result["score"] == 25
    assert result["weighted_matched"] == 0.5


def test_perfect_match_reaches_100_percent():
    jd = """
    Requirements:
    Python
    PostgreSQL
    Docker
    """
    resume_skills = {"Python", "PostgreSQL", "Docker"}
    jd_skills = {"Python", "PostgreSQL", "Docker"}
    
    result = calculate_weighted_score(resume_skills, jd_skills, jd)
    assert result["score"] == 100

