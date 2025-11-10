# test_fusion.py
"""
Unit tests for FusionRanker behavior and integration sanity checks, to test:
 - Basic fusion (Markov-only) ranking under strict preset
 - Personalization effect (user context boosts top word)
 - Empty input ( is there a proper return)
"""

import pytest
from intelligent_autocompleter.core.fusion_ranker import FusionRanker

# helpers -------------------------------------------------------------------
def fake_candidates() -> list:
    """Create candidate list with example component scores."""
    # format of each pretend tuple is (word, {'markov':score1, 'personal':score2, 'semantic':score3})
    return [
        ("apple",    {"markov": 0.9, "personal": 0.2, "semantic": 0.1}),
        ("apricot",  {"markov": 0.6, "personal": 0.4, "semantic": 0.3}),
        ("banana",   {"markov": 0.3, "personal": 0.7, "semantic": 0.2}),
        ("avocado",  {"markov": 0.5, "personal": 0.6, "semantic": 0.8}),
    ]

def extract_words(ranked):
    """Extractor for ranked word list."""
    return [w for w, _ in ranked]

# tests ---------------------------------------------------------------------
def test_fusion_basic():
    """With 'strict' preset markov is used."""
    ranker = FusionRanker(preset="strict")
    ranked = ranker.rank(fake_candidates())
    words = extract_words(ranked)
    # apple has highest markov so it should be first
    assert words[0] == "apple"
    # ensure order follows descending markov
    assert words.index("apple") < words.index("banana")
    assert words.index("apple") < words.index("apricot")


def test_fusion_personalization():
    """
    Pretend user who strongly prefers 'banana'
    check if the personalizer weighting makes it outrank apple under 'personal' preset.
    """
    class DummyCtx:
        def top_words(self, n=5):  # emulate ctx.personalizer-like API
            return {"banana": 10.0, "apple": 1.0}

    ctx = DummyCtx()
    ranker = FusionRanker(preset="personal", personalizer=ctx)
    ranked = ranker.rank(fake_candidates())
    words = extract_words(ranked)
    # banana should now rank higher due to context bias
    assert words[0] == "banana"
    assert "banana" in words
    assert words.index("banana") < words.index("apple")


def test_fusion_empty_input():
    """Empty or None input should return an empty list, not raise."""
    ranker = FusionRanker()
    assert ranker.rank([]) == []
    assert ranker.rank(None) == []

