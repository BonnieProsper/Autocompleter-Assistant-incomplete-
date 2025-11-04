# hybrid_predictor.py - combines markov, embeddings, and user context for ranked suggestions

import time
from markov_predictor import MarkovPredictor
from embeddings import Embeddings
from context_personal import CtxPersonal
from collections import Counter
from semantic_engine import SemanticEngine
from logger_utils import Log


class HybridPredictor:
    def __init__(self, user="default"):
        self.mk = MarkovPredictor()
        self.emb = Embeddings()
        self.ctx = CtxPersonal(user)
        self.alpha = 0.5    # markov/embed weight balance (0=markov only)
        self._trained = False
        self.sem = SemanticEngine()

    # training -------------------------
    def train(self, lines):
        if not lines:
            return
        with Log.time_block("Hybrid train"):
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

    # manage suggestions -------------------------------
    def suggest(self, word, topn=5):
        word = (word or "").lower().strip()
        if not word:
            return []
        t0 = time.time()
        try:
            # markov next words (contextual), markov predictions
            m_res = self.mk.top_next(word, topn=topn)
            m_rank = {w: sc for w, sc in m_res}

            # embeddings predictions, using semantic similarity
            e_res = self.emb.similar(word, topn=topn)
            e_rank = {w: sc for w, sc in e_res}

            # merge with weighting, combine markov and embeddingd
            merged = Counter()
            for w, sc in m_rank.items():
                merged[w] += sc * (1 - self.alpha)
            for w, sc in e_rank.items():
                merged[w] += sc * self.alpha

            # use semantic expansion if merging doesnt work
            if not merged or len(merged) < topn:
                sem_res = self.sem.similar(word, topn=topn)
                for w, sc in sem_res:
                    merged[w] += sc * 0.4  # smaller influence weight

            # bias by user prefs
            suggs = merged.most_common(topn)
            suggs = self.ctx.bias_words(suggs) 
        
            dur = round(time.time() - t0, 3)
            Log.metric("Suggest latency", dur, "s")
            return [(w, round(sc, 3)) for w, sc in suggs] # normalise and return
    
        except Exception as e:
            Log.write(f"Error in suggest({word}): {e}")
            return []

    # tuning ------------------------
    def set_balance(self, val: float):
        #adjust weighting between models (0-1)
        self.alpha = max(0.0, min(1.0, val))

