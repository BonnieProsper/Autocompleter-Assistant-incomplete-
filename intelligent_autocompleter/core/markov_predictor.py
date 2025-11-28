# markov_predictor.py
# first-order Markov language model for next-token prediction.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple, Optional
from collections import defaultdict, Counter
import math
import re

Word = str
Score = float
NextList = List[Tuple[Word, Score]]

@dataclass(frozen=True)
class MarkovConfig:
    """
    Configurable knobs for smoothing, tokenisation, and top-k behaviour.
    """
    topn: int = 5
    min_token_len: int = 1
    smoothing_k: float = 0.5  # laplace smoothing constant
    lowercase: bool = True

class MarkovPredictor:
    """
    First-order Markov chain predictor with:
      - configurable Laplace smoothing
      - robust tokenisation
      - probability-normalized scoring
      - serialization helpers
      - public API

    MarkovPredictor produces probabilistic scores not raw counts, 
    making it a better resource for FusionRanker.
    """

    _token_re = re.compile(r"[A-Za-z0-9']+")   # improved tokenizer (keeps contractions)

    def __init__(self, config: Optional[MarkovConfig] = None) -> None:
        self.cfg = config or MarkovConfig()
        # prev → Counter(next)
        self._chain: Dict[Word, Counter] = defaultdict(Counter)
        # unigram model
        self._uni: Counter = Counter()
        # total tokens
        self._total: int = 0

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> List[Word]:
        if not text:
            return []
        tokens = self._token_re.findall(text)
        if self.cfg.lowercase:
            tokens = [t.lower() for t in tokens]
        return [t for t in tokens if len(t) >= self.cfg.min_token_len]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_sentence(self, text: str) -> None:
        toks = self._tokenize(text)
        if not toks:
            return

        self._uni.update(toks)
        self._total += len(toks)

        for a, b in zip(toks, toks[1:]):
            self._chain[a][b] += 1

    def train_many(self, sentences: Iterable[str]) -> None:
        for s in sentences:
            self.train_sentence(s)

    # ------------------------------------------------------------------
    # Probability computation
    # ------------------------------------------------------------------
    def _compute_probs(self, counter: Counter) -> Dict[Word, Score]:
        """
        Convert a next-word Counter into a probability distribution
        with Laplace smoothing.
        """
        k = self.cfg.smoothing_k
        vocab_size = len(self._uni)

        total = sum(counter.values())
        denom = total + k * vocab_size

        return {
            w: (c + k) / denom
            for w, c in counter.items()
        }

    # ------------------------------------------------------------------
    # Prediction (public API)
    # ------------------------------------------------------------------
    def top_next(self, prev: Word, topn: Optional[int] = None) -> NextList:
        """
        Returns a list of tuples (word, probability) representing
        the most likely next tokens after 'prev'.
        If prev is unseen it uses unigram probabilities.
        """
        n = topn or self.cfg.topn

        if not prev:
            return self._top_unigrams(n)

        key = prev.lower() if self.cfg.lowercase else prev
        counter = self._chain.get(key)

        if not counter:
            return self._top_unigrams(n)

        probs = self._compute_probs(counter)
        return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:n]

    def _top_unigrams(self, n: int) -> NextList:
        if not self._uni:
            return []
        total = sum(self._uni.values())
        return [
            (w, c / total)
            for w, c in self._uni.most_common(n)
        ]

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def vocabulary_size(self) -> int:
        return len(self._uni)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_state(self) -> dict:
        return {
            "chain": {k: dict(v) for k, v in self._chain.items()},
            "uni": dict(self._uni),
            "total": self._total,
            "config": self.cfg.__dict__,
        }

    def load_state(self, data: dict) -> None:
        try:
            chain = data.get("chain", {})
            self._chain = defaultdict(Counter, {k: Counter(v) for k, v in chain.items()})
            self._uni = Counter(data.get("uni", {}))
            self._total = int(data.get("total", 0))
        except Exception:
            pass
