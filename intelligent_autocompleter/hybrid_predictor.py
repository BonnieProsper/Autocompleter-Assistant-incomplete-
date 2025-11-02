# hybrid_predictor.py - combine Markov and embedding signals for smarter next-word predictions

from collections import defaultdict
from typing import List, Tuple
import math

from markov_predictor import MarkovPredictor
from embeddings import Embeddings

class HybridPredictor:
    """fusion of markov freq model + semantic similarity"""
    def __init__(self, embed_path=None):
        self.markov = MarkovPredictor()
        self.embeds = Embeddings(embed_path)
        self.alpha = 0.65   # relative weight of markov vs embeddings

    def train(self, sentences: List[str]):
        self.markov.train_many(sentences)

    def suggest(self, prev_word: str, topn: int = 5) -> List[Tuple[str, float]]:
        """returns combined top-n suggestions"""
        if not prev_word:
            return []

        # Markov signal
        m_out = self.markov.top_next(prev_word, topn)
        m_scores = {w: float(c) for w, c in m_out}

        # Embedding signal
        e_out = self.embeds.similar(prev_word, topn)
        e_scores = {w: float(s) for w, s in e_out}

        # normalize both to [0,1]
        if m_scores:
            mmax = max(m_scores.values())
            for k in m_scores: m_scores[k] /= (mmax or 1)
        if e_scores:
            emax = max(e_scores.values())
            for k in e_scores: e_scores[k] /= (emax or 1)

        # fuse scores
        combo = defaultdict(float)
        for w, s in m_scores.items(): combo[w] += s * self.alpha
        for w, s in e_scores.items(): combo[w] += s * (1 - self.alpha)

        # rank
        out = sorted(combo.items(), key=lambda kv: -kv[1])
        return out[:topn]

    def retrain(self, text: str):
        """incrementally learn from a new sentence"""
        self.markov.train_sentence(text)

    def set_balance(self, val: float):
        """adjust weighting between models (0..1)"""
        self.alpha = max(0.0, min(1.0, val))
