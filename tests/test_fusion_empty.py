# tests/test_fusion_empty.py
from intelligent_autocompleter.core.fusion_ranker import FusionRanker

def test_fusion_empty():
    r = FusionRanker()
    out = r.rank({}, {}, {}, {}, topn=5)
    assert out == []

