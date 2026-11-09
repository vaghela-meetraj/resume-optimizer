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


@pytest.mark.xfail(strict=True)
def test_cpp_is_extracted_as_cpp_not_c():
    matches = _language_matches("C++")

    assert "c++" in matches
    assert "c" not in matches


@pytest.mark.xfail(strict=True)
def test_go_in_sentence_is_not_extracted_as_skill():
    assert "go" not in _language_matches("We go above and beyond")


@pytest.mark.xfail(strict=True)
def test_r_and_d_is_not_extracted_as_r():
    assert "r" not in _language_matches("R&D")
