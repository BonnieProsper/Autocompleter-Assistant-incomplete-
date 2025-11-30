# tests/test_fusion_ranker.py
# unit tests for rank_noramlized in FusionRanker

import pytest
from fusion_ranker import FusionRanker


@pytest.fixture
def ranker():
    return FusionRanker(
        w_markov=0.6,
        w_semantic=0.3,
        w_fuzzy=0.1,
    )


def test_rank_normalized_basic(ranker):
    out = ranker.rank_normalized(
        markov={"apple": 1.0},
        semantic={"apple": 0.5},
        fuzzy={"apple": 0.0},
        top_k=5,
    )

    # score = .6*1.0 + .3*0.5 + .1*0.0 = 0.75
    assert out == [("apple", pytest.approx(0.75, rel=1e-6))]

# tests/test_fusion_ranker.py
import pytest
from intelligent_autocompleter.core.fusion_ranker import FusionRanker

def test_rank_normalized_basic_ordering():
    fr = FusionRanker(weights={"markov": 0.6, "embed": 0.4, "fuzzy": 0.0, "freq": 0.0, "recency": 0.0})
    # normalized features for two words
    markov = {"apple": 1.0, "apply": 0.0}
    embed  = {"apple": 0.0, "apply": 1.0}
    fuzzy  = {"apple": 0.0, "apply": 0.0}
    freq   = {"apple": 0.0, "apply": 0.0}
    recency= {"apple": 0.0, "apply": 0.0}

    ranked = fr.rank_normalized(markov, embed, fuzzy, freq, recency, topn=2)
    assert ranked[0][0] == "apple" and ranked[1][0] == "apply"
    # scores: apple = 0.6, apply = 0.4
    assert ranked[0][1] > ranked[1][1]

def test_rank_normalized_tie_breaking_lexicographic():
    # equal weights produce equal scores -> tie-break lexicographically
    fr = FusionRanker(weights={"markov": 0.5, "embed": 0.5, "fuzzy": 0.0, "freq": 0.0, "recency": 0.0})
    # both words have identical normalized features -> same score
    markov = {"a": 1.0, "b": 1.0}
    embed  = {"a": 0.0, "b": 0.0}
    fuzzy  = {"a": 0.0, "b": 0.0}
    freq   = {"a": 0.0, "b": 0.0}
    recency= {"a": 0.0, "b": 0.0}

    ranked = fr.rank_normalized(markov, embed, fuzzy, freq, recency, topn=2)
    # tie-breaker uses lexicographic ascending
    assert ranked[0][0] == "a"
    assert ranked[1][0] == "b"

def test_rank_normalized_merges_keys(ranker):
    out = ranker.rank_normalized(
        markov={"cat": 1.0},
        semantic={"dog": 1.0},
        fuzzy={"rat": 1.0},
        top_k=3,
    )

    # deterministic order using weighted score
    scores = dict(out)
    assert scores["cat"] == pytest.approx(0.6)
    assert scores["dog"] == pytest.approx(0.3)
    assert scores["rat"] == pytest.approx(0.1)


def test_rank_normalized_top_k(ranker):
    out = ranker.rank_normalized(
        markov={"a": 1, "b": 0.9, "c": 0.8},
        semantic={},
        fuzzy={},
        top_k=2,
    )
    assert len(out) == 2
    assert out[0][0] == "a"
    assert out[1][0] == "b"
