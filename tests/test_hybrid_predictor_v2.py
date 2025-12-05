# tests/test_hybrid_predictor.py

import pytest
from unittest.mock import MagicMock
from hybrid_predictor import HybridPredictor


@pytest.fixture
def mocks():
    return {
        "markov": MagicMock(),
        "semantic": MagicMock(),
        "bk": MagicMock(),
        "ranker": MagicMock(),
        "personal": MagicMock(),
        "rl": MagicMock(),
        "logger": MagicMock(),
    }


@pytest.fixture
def predictor(mocks):
    return HybridPredictor(
        markov=mocks["markov"],
        semantic_engine=mocks["semantic"],
        bk_tree=mocks["bk"],
        ranker=mocks["ranker"],
        personalizer=mocks["personal"],
        rl_agent=mocks["rl"],
        logger=mocks["logger"],
    )


# -----------------------------------
# Test: empty fragment
# -----------------------------------
def test_empty_fragment_returns_empty(predictor):
    assert predictor.suggest("") == []


# -----------------------------------
# Test: basic happy path
# -----------------------------------
def test_suggest_normal_flow(predictor, mocks):

    mocks["markov"].predict_next.return_value = {"hello": 0.9}
    mocks["semantic"].similar_words.return_value = {"hello": 0.4}
    mocks["bk"].search.return_value = {"hello": 0.1}

    # personalization just returns what it receives
    mocks["personal"].adjust_map.side_effect = lambda m, source: m
    mocks["rl"].adjust.side_effect = lambda name, frag, m: m

    mocks["ranker"].rank_normalized.return_value = [("hello", 1.0)]

    suggestions = predictor.suggest("he", top_k=5)
    assert suggestions == [("hello", 1.0)]

    mocks["ranker"].rank_normalized.assert_called_once()


# -----------------------------------
# Test: long fragment
# -----------------------------------
def test_long_fragment(predictor, mocks):
    long = "x" * 500
    mocks["markov"].predict_next.return_value = {}
    mocks["semantic"].similar_words.return_value = {}
    mocks["bk"].search.return_value = {}

    mocks["personal"].adjust_map.side_effect = lambda m, source: m
    mocks["rl"].adjust.side_effect = lambda name, frag, m: m

    mocks["ranker"].rank_normalized.return_value = []

    assert predictor.suggest(long) == []


# -----------------------------------
# Test: upstream component failure
# -----------------------------------
def test_graceful_failure(predictor, mocks):
    mocks["markov"].predict_next.side_effect = Exception("oops")

    res = predictor.suggest("he")
    assert res == []  # graceful fallback
    mocks["logger"].error.assert_called()
