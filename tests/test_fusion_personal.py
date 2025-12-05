# tests/test_fusion_personal.py
from intelligent_autocompleter.core.fusion_ranker import FusionRanker


def test_fusion_personal():
    r = FusionRanker("personal")
    mk = {"alpha": 1.0, "beta": 1.0}
    eb = {"alpha": 0.5, "beta": 0.5}
    # user strongly prefers beta
    per = {"alpha": 0.0, "beta": 10.0}
    freq = {"alpha": 1.0, "beta": 1.0}
    out = r.rank(mk, eb, per, freq, topn=2)
    assert out[0][0] == "beta"
