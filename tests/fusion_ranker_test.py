import pytest
from intelligent_autocompleter.core.fusion_ranker import FusionRanker


def test_fusion_basic():
    r = FusionRanker("strict")
    markov = {"hello": 0.9, "hi": 0.8}
    ranked = r.rank(markov)
    assert ranked[0][0] == "hello"


def test_fusion_personal():
    r = FusionRanker("personal")
    markov = {"hey": 0.5, "hello": 0.4}
    personal = {"hello": 1.0}
    ranked = r.rank(markov, personal)
    assert ranked[0][0] == "hello"


def test_fusion_empty():
    r = FusionRanker()
    ranked = r.rank({}, {}, {})
    assert ranked == []
