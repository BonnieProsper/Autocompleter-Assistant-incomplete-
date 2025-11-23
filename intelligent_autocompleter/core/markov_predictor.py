# markov_predictor.py
# small first-order Markov next-word predictor, trained on plain sentences, used by HybridPredictor

from __future__ import annotations
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Iterable

Word = str
Count = int
NextList = List[Tuple[Word, Count]]

class MarkovPredictor:
    """
    Simple first-order Markov chain:
        prev_word -> Counter(next_word)

    Intentionally simple, tends to produce
    "local" predictions (what usually follows X),
    which blend nicely with embeddings + semantic predictors.
    """
    _token_re = re.compile(r"\w+")

    def __init__(self) -> None:
        # prev -> Counter(next)
        self._chain: Dict[Word, Counter] = defaultdict(Counter)
        # unigram fallback (rarely used but useful)
        self._uni: Counter = Counter()
        # keep track of total tokens for potential smoothing/normalization (future)
        self._total = 0

    # training -----------------------------------------------------------------------
    def train_sentence(self, text: str) -> None:
        """Train on a single sentence. Very forgiving, empty/short inputs are ignored."""
        if not text:
            return

        toks = self._token_re.findall(text.lower())
        if not toks:
            return
            
        # update unigrams
        self._uni.update(toks)
        self._total += len(toks)
        # update bigrams
        # (small pitfalls: trailing punctuation, numbers, weird unicode - regex handles 80% fine)
        for a, b in zip(toks, toks[1:]):
            self._chain[a][b] += 1

    def train_many(self, sentences: Iterable[str]) -> None:
        """Simple bulk training."""
        for s in sentences:
            self.train_sentence(s)

    # prediction ---------------------------------------------------------------
    def top_next(self, prev: Word, topn: int = 5) -> NextList:
        """
        Returns a list of (word, count) for the most common next words.
        If prev not in chain, fall back to global unigrams.
        The caller (HybridPredictor) will rescale these counts
        into actual scores and merge with embeddings/semantic sources.
        """
        if not prev:
            # completely empty fragment → unigrams
            return self._uni.most_common(topn)

        key = prev.lower()
        nxt = self._chain.get(key)
        if nxt:
            return nxt.most_common(topn)

        # unseen prev word → unigram fallback
        return self._uni.most_common(topn)

    # optional helpers
    def vocabulary_size(self) -> int:
        """Rough vocabulary measure (unique token count)."""
        return len(self._uni)

    def save_state(self) -> dict:
        """
        Compact serializable snapshot.
        (Used optionally by persistence layer, deliberately lightweight.)
        """
        return {
            "chain": {k: dict(v) for k, v in self._chain.items()},
            "uni": dict(self._uni),
            "total": self._total,
        }

    def load_state(self, data: dict) -> None:
        """Restore from save_state() output. Missing fields ignored."""
        try:
            chain = data.get("chain", {})
            self._chain = defaultdict(Counter, {k: Counter(v) for k, v in chain.items()})
            self._uni = Counter(data.get("uni", {}))
            self._total = int(data.get("total", 0))
        except Exception:
            # if bad file or format mismatch, fail softly
            pass
