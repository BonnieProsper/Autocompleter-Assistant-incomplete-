# tests/test_fusion.py
import pytest
from intelligent_autocompleter.core.fusion_ranker import FusionRanker


def mk_cands():
    # markov: apple strongest, apricot second, banana weak
    return [("apple", 10), ("apricot", 6), ("banana", 2), ("avocado", 4)]


def emb_cands():
    # embeddings: avocado is semantically close, banana also okay
    return [("avocado", 0.9), ("banana", 0.6), ("apricot", 0.3)]


def fuzzy_cands():
    # fuzzy returns distance, smaller is better
    return [("apricot", 1), ("apple", 2)]


def test_fusion_basic_strict():
    r = FusionRanker(preset="strict")
    out = r.rank(markov=mk_cands(), embeddings=[], fuzzy=[], base_freq={}, topn=3)
    words = [w for w, _ in out]
    assert words[0] == "apple"
    assert "apricot" in words


def test_fusion_personal_boost():
    class DummyPersonal:
        def bias_words(self, ranked):
            # push banana to top artificially
            out = [(w, sc + (5.0 if w == "banana" else 0.0)) for w, sc in ranked]
            out.sort(key=lambda x: -x[1])
            return out

    r = FusionRanker(preset="personal", personalizer=DummyPersonal())
    out = r.rank(
        markov=mk_cands(),
        embeddings=emb_cands(),
        fuzzy=fuzzy_cands(),
        base_freq={},
        topn=3,
    )
    words = [w for w, _ in out]
    assert words[0] == "banana"


def test_fusion_empty():
    r = FusionRanker()
    assert r.rank([], [], [], {}) == []
    assert r.rank(None, None, None, None) == []
