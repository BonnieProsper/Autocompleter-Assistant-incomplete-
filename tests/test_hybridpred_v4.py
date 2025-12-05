# tests/test_hybrid_predictor.py
import pytest
from unittest.mock import MagicMock

from intelligent_autocompleter.core.hybrid_predictor import HybridPredictor


class StubMarkov:
    def top_next(self, prev, topn=5):
        return [("world", 3), ("there", 1)] if prev == "hello" else []


class StubSemantic:
    def similar(self, token, topn=5):
        return [("world", 0.9)] if token == "hello" else []


class StubBK:
    def __init__(self):
        pass

    def query(self, q, max_dist=1):
        return [("word", 1)]


class StubCtx:
    def __init__(self):
        self.freq = {"world": 2}
        self.recent = ["world"]

    def learn(self, txt):
        pass

    def bias_words(self, ranked):
        return ranked

    def save(self):
        pass


class StubRL:
    def get_weights(self):
        return {"semantic": 0.5, "markov": 0.4, "personal": 0.1}


@pytest.fixture
def hybrid(tmp_path, monkeypatch):
    hp = HybridPredictor()
    # inject stubs
    hp.markov = StubMarkov()
    hp.semantic = StubSemantic()
    hp.bk = StubBK()
    hp.ctx = StubCtx()
    hp.reinforcement = StubRL()
    return hp


def test_suggest_basic(hybrid):
    out = hybrid.suggest("hello", topn=3)
    assert isinstance(out, list)
    # Expect 'world' to appear (from markov or semantic)
    assert any(w for w, _ in out if w in ("world", "there"))


def test_accept_records(hybrid, monkeypatch):
    # monkeypatch reinforcement.record_accept to inspect calls
    called = {}

    def record_accept(ctx, sug, src=None):
        called["ok"] = (ctx, sug, src)

    hybrid.reinforcement.record_accept = record_accept
    hybrid._last_sources = {"world": "markov"}
    hybrid.accept("world", context="hello world")
    assert "ok" in called and called["ok"][1] == "world"
