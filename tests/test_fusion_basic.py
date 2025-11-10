# tests/test_fusion_basic.py
from intelligent_autocompleter.core.fusion_ranker import FusionRanker

def test_fusion_basic():
    r = FusionRanker("strict")
    # markov strongly favors 'run' over others
    mk = {"run": 10.0, "jog": 2.0}
    eb = {"run": 0.1, "jog": 0.4}
    per = {"run": 0.0, "jog": 0.0}
    freq = {"run": 5.0, "jog": 1.0}
    out = r.rank(mk, eb, per, freq, topn=2)
    assert out[0][0] == "run"
