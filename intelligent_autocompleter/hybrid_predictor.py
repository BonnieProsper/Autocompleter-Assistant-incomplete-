# hybrid_predictor.py - combines markov, embeddings, and user context for ranked suggestions

from markov_predictor import MarkovPredictor
from embeddings import Embeddings
from context_personal import CtxPersonal
from collections import Counter

class HybridPredictor:
    def __init__(self, user="default"):
        self.mk = MarkovPredictor()
        self.emb = Embeddings()
        self.ctx = CtxPersonal(user)
        self.alpha = 0.5    # markov/embed weight balance (0=markov only)
        self._trained = False

    # training -------------------------
    def train(self, lines):
        for ln in lines:
            self.mk.train_sentence(ln)
            self.ctx.learn(ln)
        self.ctx.save()
        self._trained = True

    def retrain(self, s):
        self.mk.train_sentence(s)
        self.ctx.learn(s)
        self.ctx.save()
        self._trained = True

    # suggestion core -------------------------------
    def suggest(self, word, topn=5):
        word = word.lower().strip()
        if not word:
            return []
        # markov next-words (contextual)
        m_res = self.mk.top_next(word, topn=topn)
        m_rank = {w: sc for w, sc in m_res}

        # embeddings (semantic similarity)
        e_res = self.emb.similar(word, topn=topn)
        e_rank = {w: sc for w, sc in e_res}

        # merge with weighting
        merged = Counter()
        for w, sc in m_rank.items():
            merged[w] += sc * (1 - self.alpha)
        for w, sc in e_rank.items():
            merged[w] += sc * self.alpha

        suggs = merged.most_common(topn)
        # bias by user prefs
        suggs = self.ctx.bias_words(suggs)
        # normalize and return
        return [(w, round(sc, 3)) for w, sc in suggs]

    def set_balance(self, val: float):
        """set 0–1 balance between markov and embeddings"""
        self.alpha = max(0.0, min(1.0, val))


    def set_balance(self, val: float):
        """adjust weighting between models (0..1)"""
        self.alpha = max(0.0, min(1.0, val))
