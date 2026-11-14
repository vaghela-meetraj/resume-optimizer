import re

import pytest

from app.services.nlp import NLPProcessor
from app.services.skills import fuzzy_skill_match


def _language_matches(text):
    language_pattern = NLPProcessor._load_tech_patterns(None)["languages"]
    return {match.lower() for match in re.findall(language_pattern, text, re.IGNORECASE)}


def test_javascript_does_not_match_java():
    result = fuzzy_skill_match({"JavaScript"}, {"Java"})

    assert result["missing"] == ["Java"]
    assert result["exact"] == []
    assert result["partial"] == []

    react_result = fuzzy_skill_match({"React Native"}, {"React"})
    assert react_result["partial"] == ["React"]


def test_cpp_is_extracted_as_cpp_not_c():
    matches = _language_matches("C++")

    assert "c++" in matches
    assert "c" not in matches


def test_go_in_sentence_is_not_extracted_as_skill():
    assert "go" not in _language_matches("We go above and beyond")


def test_r_and_d_is_not_extracted_as_r():
    assert "r" not in _language_matches("R&D")


def test_mysql_satisfies_sql_requirement():
    """Candidates with MySQL should satisfy a general SQL requirement."""
    result = fuzzy_skill_match({"MySQL"}, {"SQL"})
    assert "SQL" in result["exact"]
    assert "SQL" not in result["missing"]


def test_sentence_ending_period_does_not_block_skill_extraction():
    """Skills immediately followed by a sentence-ending period must be extracted."""
    nlp = NLPProcessor()
    text = "Experience with containerization technologies like Docker. Proficient with Git and GitHub."
    extracted = nlp.extract_skills_nlp(text)
    assert "Docker" in extracted
    assert "Git" in extracted
    assert "GitHub" in extracted


def test_single_char_languages_c_and_r_extracted():
    """Languages C and R must not be filtered out by noise/length check."""
    nlp = NLPProcessor()
    text = "Strong proficiency in C, C++, Python, and R."
    extracted = nlp.extract_skills_nlp(text)
    assert "C" in extracted
    assert "C++" in extracted
    assert "Python" in extracted
    assert "R" in extracted


def test_solid_adjective_does_not_extract_solid_principles():
    """English adjective 'solid' should not trigger the engineering concept SOLID."""
    nlp = NLPProcessor()
    adjective_text = "Solid understanding of relational database systems."
    extracted_adj = nlp.extract_skills_nlp(adjective_text)
    assert "SOLID" not in extracted_adj

    concept_text = "Experience applying OOP and SOLID principles."
    extracted_concept = nlp.extract_skills_nlp(concept_text)
    assert "SOLID" in extracted_concept

