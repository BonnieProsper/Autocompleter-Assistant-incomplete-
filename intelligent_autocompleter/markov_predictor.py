# markov_predictor.py - a markov chain next-word predictor that trains on sentences and gives top-n next words.

from collections import defaultdict, Counter
from typing import List, Tuple
import re


class MarkovPredictor:
    """first-order markov predictor (prev -> Counter(next))."""
    def __init__(self):
        self._table = defaultdict(Counter)
        self._unigrams = Counter()

    def train_sentence(self, s: str):
        toks = re.findall(r"\w+", s.lower())
        if not toks:
            return
        for i in range(len(toks) - 1):
            a, b = toks[i], toks[i + 1]
            self._table[a][b] += 1
            self._unigrams[a] += 1
        self._unigrams[toks[-1]] += 1

    def train_many(self, sentences: List[str]):
        for s in sentences:
            self.train_sentence(s)

    def top_next(self, prev: str, topn: int = 5) -> List[Tuple[str, int]]:
        prev = prev.lower()
        if prev in self._table:
            return self._table[prev].most_common(topn)
        # backoff to unigrams
        return self._unigrams.most_common(topn)


