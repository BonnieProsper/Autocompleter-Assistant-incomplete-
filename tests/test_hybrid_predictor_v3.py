"""
To run test locally:

# from repo root
python -m pip install -U pytest
pytest -q
"""

# tests/test_hybrid_predictor_v3.py
import types
from intelligent_autocompleter.core.hybrid_predictor import HybridPredictor

class DummyMarkov:
    def top_next(self, prev, topn=10):
        return [("apple", 2), ("apply", 1)]

class DummySemantic:
    def similar(self, token, topn=10):
        return [("apply", 0.9)]

class DummyBK:
    def query(self, token, max_dist=1):
        return []

class DummyReinforcement:
    def get_weights(self):
        return {"markov": 0.6, "semantic": 0.4, "personal": 0.0, "plugin": 0.0}
    def record_accept(self, *a, **k):
        pass
    def record_reject(self, *a, **k):
        pass

def test_suggest_uses_rank_normalized(monkeypatch):
    hp = HybridPredictor()
    # inject simple mocks
    hp.markov = DummyMarkov()
    hp.semantic = DummySemantic()
    hp.bk = DummyBK()
    hp.reinforcement = DummyReinforcement()

    # patch the preprocessor to return deterministic normalized map
    def fake_normalize_all(markov=None, embed=None, fuzzy=None, freq=None, recency=None):
        return {
            "apple": {"markov": 1.0, "embed": 0.0, "fuzzy": 0.0, "freq": 0.0, "recency": 0.0},
            "apply": {"markov": 0.0, "embed": 1.0, "fuzzy": 0.0, "freq": 0.0, "recency": 0.0},
        }
    monkeypatch.setattr(hp.pre, "normalize_all", fake_normalize_all)

    out = hp.suggest("I want to ap", topn=2, fuzzy=False)
    assert len(out) >= 2
    # weights put weight 0.6 on markov and 0.4 on embed so 'apple' should be ranked first
    assert out[0][0] == "apple"
    assert out[1][0] == "apply"
