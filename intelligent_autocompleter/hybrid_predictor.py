# hybrid_predictor.py - combines markov, embeddings, and user context for ranked suggestions, 
# with adaptive fuzzy distance, BK cache, lightweight reinforcement.

import time
from collections import Counter, deque
from markov_predictor import MarkovPredictor
from embeddings import Embeddings
from context_personal import CtxPersonal
from semantic_engine import SemanticEngine
from logger_utils import Log
from bktree import BKTree

class HybridPredictor:
    """
    hybrid_predictor.py combines three knowledge sources:
      - Markov chain probabilities (syntactic structure)
      - Word embeddings (semantic proximity)
      - User personalization (biasing + context memory)

    It can optionally use multi-word context to predict the next word
    based on the recent sentence fragment.
    """
    def __init__(self, user="default", context_window=2):
        self.mk = MarkovPredictor()
        self.emb = Embeddings()
        self.ctx = CtxPersonal(user)
        self.alpha = 0.5   # markov/embed weight balance (0=markov only)
        self._trained = False
        self.sem = SemanticEngine()
        self.context_window = context_window

        """
        take the following out?
        # BK-tree for fuzzy lookups, synced with the training
        self.bk = BKTree()

        # reinforcement counts of accepted suggestions
        self.accepted = Counter()

        # query cache for BK queries: key -> (result) + timestamps
        self._bk_cache = {}
        self._bk_cache_time = {}
        self._bk_cache_ttl = 60.0  # seconds
        """

    # training -----------------------
    def train(self, lines):
        """Train on file/corpus.Include in BK tree for fuzzy lookup."""
        if not lines:
            return
        with Log.time_block("Hybrid train"):
            for ln in lines:
                self.mk.train_sentence(ln)
                self.ctx.learn(ln)
                """ - take out?
                # update BK with words from the line
                for tok in self._tokens_from_line(ln):
                    try:
                        self.bk.insert(tok)
                    except Exception:
                        # BK insert could fail on stray tokens, if so ignore/continue
                        pass
                """ 
            self.ctx.save()
            self._trained = True

    def retrain(self, sentence):
        """update from interactive input."""
        self.mk.train_sentence(sentence)
        self.ctx.learn(sentence)
        """
        for tok in self._tokens_from_line(s):
            try:
                self.bk.insert(tok)
            except Exception:
                pass
        """
        self.ctx.save()
        self._trained = True

    """ take out
    # helpers -----------------------
    def _tokens_from_line(self, s):
        # keep alpha-only tokens
        return [t.lower() for t in s.split() if t.isalpha()]

    def _adaptive_max_dist(self, q: str) -> int:
        """Heuristic, meaning that longer queries allow more edit distance."""
        L = len(q)
        if L <= 3:
            return 0
        if L <= 6:
            return 1
        if L <= 12:
            return 2
        return 3

    def _bk_query_cached(self, q: str, maxd: int):
        """Cache BK queries for a short TTL to increase efficiency."""
        key = (q, maxd)
        now = time.time()
        if key in self._bk_cache:
            ts = self._bk_cache_time.get(key, 0)
            if now - ts < self._bk_cache_ttl:
                return self._bk_cache[key]
            # expired
            try:
                del self._bk_cache[key]
                del self._bk_cache_time[key]
            except KeyError:
                pass
        # run query and cache
        try:
            res = self.bk.query(q, max_dist=maxd)
        except Exception:
            res = []
        self._bk_cache[key] = res
        self._bk_cache_time[key] = now
        return res
    """

    # manage context aware suggestions ------------------------
    def suggest(self, word, topn=5):
        """
        Return a list of (word,score) suggestions.
        Scoring is a weighted merge of Markov counts and embedding similarity,
        optionally supplemented by fuzzy matches from the BK tree.
        """
        q = (word or "").lower().strip()
        if not q:
            return []

        t0 = time.time()
        try:
            # markov next words (contextual), markov predictions
            m_res = self.mk.top_next(q, topn=topn)
            m_rank = {w: sc for w, sc in m_res}

            # embeddings predictions, using semantic similarity
            e_res = self.emb.similar(q, topn=topn)
            e_rank = {w: sc for w, sc in e_res}

            # merge with weighting, combine markov and embeddings
            merged = Counter()
            for w, sc in m_rank.items():
                merged[w] += sc * (1 - self.alpha)
            for w, sc in e_rank.items():
                merged[w] += sc * self.alpha

            # use semantic expansion as fallback if merging doesnt work
            if not merged or len(merged) < topn:
                try:
                    sem_res = self.sem.similar(q, topn=topn)
                    for w, sc in sem_res:
                        merged[w] += sc * 0.4
                except Exception:
                    pass

            # BK tree-based, controlled by adaptive max distance for fuzzy
            maxd = self._adaptive_max_dist(q)
            if maxd > 0 and getattr(self, "bk", None) is not None:
                fuzzy = self._bk_query_cached(q, maxd)
                # fuzzy entries: (word, dist) tuples
                for w, dist in fuzzy:
                    bonus = max(0.0, (maxd + 1) - dist)
                    merged[w] += bonus * 0.8

            # accepted suggestions get boosted for reinforcement purposes
            for w, c in self.accepted.items():
                merged[w] += c * 1.5

            # bias by user preferences 
            suggs = merged.most_common()
            suggs = self.ctx.bias_words(suggs)

            dur = round(time.time() - t0, 3)
            Log.metric("Suggest latency", dur, "s")

            # normalize and return topn
            out = [(w, round(float(sc), 3)) for w, sc in suggs[:topn]]
            return out

        except Exception as e:
            Log.write(f"Error in suggest({q}): {e}")
            return []

    # reinforcement ----------------------
    def accept(self, word: str):
        """This is called when a user accepts a suggestion, as it increments reinforcement count."""
        if not word:
            return
        w = word.lower().strip()
        self.accepted[w] += 1
        # bound counts
        if self.accepted[w] > 1000:
            self.accepted[w] = 1000

    # tuning -------------------------------
    def set_balance(self, val: float):
        self.alpha = max(0.0, min(1.0, float(val)))

