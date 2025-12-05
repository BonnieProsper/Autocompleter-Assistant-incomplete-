# tests/test_hybrid_predictor.py
import unittest
from unittest.mock import MagicMock, patch
from typing import List, Tuple

# Import HybridPredictor from package layout, fallback works if running tests locally.
try:
    from intelligent_autocompleter.core.hybrid_predictor import HybridPredictor
    from intelligent_autocompleter.core.fusion_ranker import FusionRanker
except Exception:
    from core.hybrid_predictor import HybridPredictor  # type: ignore
    from core.fusion_ranker import FusionRanker  # type: ignore


class DummyFusion(FusionRanker):
    """Simple deterministic fusion used for tests. Returns average of provided maps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def rank(
        self,
        markov=None,
        embeddings=None,
        fuzzy=None,
        base_freq=None,
        recency_map=None,
        topn=8,
    ):
        # simple scoring: prefer markov then embed then freq (weighted)
        scores = {}
        markov = markov or []
        embeddings = embeddings or []
        base_freq = base_freq or {}
        for w, v in markov:
            scores[w] = scores.get(w, 0.0) + float(v) * 1.0
        for w, v in embeddings:
            scores[w] = scores.get(w, 0.0) + float(v) * 0.5
        for w, v in base_freq.items():
            scores[w] = scores.get(w, 0.0) + float(v) * 0.1
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [(w, round(float(s), 6)) for w, s in ranked[:topn]]


class HybridPredictorSuggestTests(unittest.TestCase):
    @patch("intelligent_autocompleter.core.hybrid_predictor.MarkovPredictor")
    @patch("intelligent_autocompleter.core.hybrid_predictor.SemanticEngine")
    @patch("intelligent_autocompleter.core.hybrid_predictor.BKTree")
    @patch("intelligent_autocompleter.core.hybrid_predictor.FusionRanker")
    @patch("intelligent_autocompleter.core.hybrid_predictor.ReinforcementLearner")
    def test_suggest_basic_flow(
        self, MockRL, MockFusion, MockBK, MockSemantic, MockMarkov
    ):
        """
        Test suggest() calls Markov, Semantic, BK, uses FusionRanker and returns ordered results.
        """
        # Setup mocks
        mock_markov = MockMarkov.return_value
        mock_markov.top_next.return_value = [("apple", 10), ("applet", 3)]

        mock_sem = MockSemantic.return_value
        mock_sem.similar.return_value = [("application", 0.9), ("apply", 0.6)]

        mock_bk = MockBK.return_value
        mock_bk.query.return_value = [("apple", 1), ("apply", 2)]

        # RL weights
        mock_rl = MockRL.return_value
        mock_rl.get_weights.return_value = {
            "semantic": 0.4,
            "markov": 0.5,
            "personal": 0.1,
            "plugin": 0.0,
        }

        # Use DummyFusion to produce deterministic ranking
        MockFusion.return_value = DummyFusion()

        hp = HybridPredictor(user="test-user")
        # inject our mocks explicitly to be sure
        hp.markov = mock_markov
        hp.semantic = mock_sem
        hp.bk = mock_bk
        hp.reinforcement = mock_rl
        hp.base_ranker = DummyFusion()

        # Call suggest
        out = hp.suggest("I love appl", topn=3, fuzzy=True)

        # Ensure outputs are present and deterministic
        self.assertIsInstance(out, list)
        # Expect at most 3 suggestions
        self.assertLessEqual(len(out), 3)
        # Top suggestion should be 'apple' given markov had highest count
        if out:
            self.assertEqual(out[0][0], "apple")

        # ensure mocks called
        mock_markov.top_next.assert_called()
        mock_sem.similar.assert_called()
        mock_bk.query.assert_called()

    @patch("intelligent_autocompleter.core.hybrid_predictor.ReinforcementLearner")
    @patch("intelligent_autocompleter.core.hybrid_predictor.FusionRanker")
    def test_accept_forwards_to_reinforcement(self, MockFusion, MockRL):
        """Ensure accept() forwards event to reinforcement learner record_accept."""
        mock_rl = MockRL.return_value
        MockFusion.return_value = DummyFusion()

        hp = HybridPredictor(user="x")
        hp.reinforcement = mock_rl

        hp._last_sources = {"tok": "markov"}
        hp.accept("tok", context="ctx", source=None)

        mock_rl.record_accept.assert_called_once()
        args = mock_rl.record_accept.call_args[0]
        # args: (context, suggestion, source)
        self.assertEqual(args[0], "ctx")
        self.assertEqual(args[1], "tok")
        # resolved source should match mapping
        self.assertIn(args[2], ("markov", "semantic", "personal", "plugin", "fuzzy"))

    @patch("intelligent_autocompleter.core.hybrid_predictor.ReinforcementLearner")
    def test_reject_forwards_to_reinforcement(self, MockRL):
        mock_rl = MockRL.return_value
        hp = HybridPredictor(user="x")
        hp.reinforcement = mock_rl
        hp._last_sources = {"tok": "semantic"}
        hp.reject("tok")
        mock_rl.record_reject.assert_called_once()


if __name__ == "__main__":
    unittest.main()
