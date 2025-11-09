# hybrid_predictor.py combines markov + Embeddings + BK-tree fuzzy + personalization


import time
from collections import Counter
from typing import List, Tuple, Iterable

from markov_predictor import MarkovPredictor
from embeddings import Embeddings
from context_personal import CtxPersonal
from semantic_engine import SemanticEngine
from logger_utils import Log
from bktree import BKTree
from fusion_ranker import FusionRanker

# create once (reuse) in constructor
self.rankr = FusionRanker(preset="balanced", personalizer=self.ctx)


class HybridPredictor:
    """
    Combine Markov predictions, embedding similarity and fuzzy matches 
    then bias with per-user preferences.

    Includes (in order):
      - get markov next-word counts
      - get embedding-based similar words
      - merge with weight alpha (0..1), alpha==1 -> embeddings only
      - add fuzzy matches from BK-tree with adaptive max dist (optional)
      - boost accepted words and user-preferred words with CtxPersonal
    """

    def __init__(self, user: str = "default"):
        self.mk = MarkovPredictor()
        self.emb = Embeddings()
        self.ctx = CtxPersonal(user)
        self.sem = SemanticEngine()

        self.alpha = 0.5  # markov/embed weight balance (0=markov only)
        self._trained = False

        # BK-tree for fuzzy queries, synced with training/retrain 
        self.bk = BKTree()

        # counts of accepted suggestions for reinforcement training
        self.accepted = Counter()

        # cache for BK queries: key -> (result list), expire after ttl
        self._bk_cache = {}
        self._bk_cache_time = {}
        self._bk_ttl = 60.0  # seconds

    # training ----------------------------------------------------
    def train(self, lines: Iterable[str]) -> None:
        """Train from an iterable of sentence strings."""
        if not lines:
            return
        with Log.time_block("Hybrid train"):
            for ln in lines:
                self.mk.train_sentence(ln)
                self.ctx.learn(ln)
                # add words to BK for fuzzy matching
                for tok in self._tokens_from_line(ln):
                    try:
                        self.bk.insert(tok)
                    except Exception:
                        # ignore odd tokens
                        pass
            self.ctx.save()
            self._trained = True

    def retrain(self, s: str) -> None:
        """Online update when user types a new line/from interactive input."""
        self.mk.train_sentence(s)
        self.ctx.learn(s)
        for tok in self._tokens_from_line(s):
            try:
                self.bk.insert(tok)
            except Exception:
                pass
        self.ctx.save()
        self._trained = True

    # helpers ---------------------------------------------------
    def _tokens_from_line(self, s: str) -> List[str]:
        """Return alpha-only tokens in lowercase."""
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

    def _bk_query_cached(self, q: str, maxd: int) -> List[Tuple[str, int]]:
        """Cache BK queries for short TTL to avoid repeated work and increase efficiency."""
        key = (q, maxd)
        now = time.time()
        if key in self._bk_cache:
            ts = self._bk_cache_time.get(key, 0)
            if now - ts < self._bk_ttl:
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

    # suggestion core ---------------------------------------------------
    def suggest(self, word: str, topn: int = 6) -> List[Tuple[str, float]]:
        """
        Return list of (word, score) suggestions. Scores are not probabilities but
        relative values useful for ranking.
        """
        q = (word or "").lower().strip()
        if not q:
            return []

        t0 = time.time()
        try:
            # markov next-word suggestions with contextual frequency
            m_res = self.mk.top_next(q, topn=topn)
            m_map = {w: float(sc) for w, sc in m_res}

            # embedding-based preditcions using semantic similarity
            e_res = self.emb.similar(q, topn=topn)
            e_map = {w: float(sc) for w, sc in e_res}

            # merge with weighting: markov*(1-alpha) + emb*alpha
            merged = Counter()
            for w, sc in m_map.items():
                merged[w] += sc * (1.0 - self.alpha)
            for w, sc in e_map.items():
                merged[w] += sc * self.alpha

            # use semantic_engine expansion as fallback if merge doesnt work
            if not merged or len(merged) < topn:
                try:
                    sem = self.sem.similar(q, topn=topn)
                    for w, sc in sem:
                        merged[w] += float(sc) * 0.35
                except Exception:
                    pass

            # Add fuzzy matches from BK-tree using adaptive edit distance
            maxd = self._adaptive_max_dist(q)
            if maxd > 0 and getattr(self, "bk", None) is not None:
                fuzzy = self._bk_query_cached(q, maxd)
                # fuzzy: list of (word, dist)
                for w, dist in fuzzy:
                    # bonus proportional to closeness and penalise larger dist
                    bonus = max(0.0, (maxd + 1) - dist)
                    merged[w] += bonus * 0.8

            # Accepted suggestions are reinforced
            for w, cnt in self.accepted.items():
                merged[w] += cnt * 1.2

            # Convert to list and bias by user preferences
            ranked = merged.most_common()
            ranked = self.ctx.bias_words(ranked)

            # Normalize/truncate, measure time, and return topn
            out = [(w, round(float(score), 3)) for w, score in ranked[:topn]]

            Log.metric("Suggest latency", round(time.time() - t0, 3), "s")
            return out

        except Exception as e:
            Log.write(f"[ERROR] suggest({q}): {e}")
            return []

        m_res = self.mk.top_next(last_word, topn=10)      # [(w,count), ...]
        e_res = self.emb.similar(last_word, topn=10)      # [(w,sim), ...]
        f_res = self._bk_query_cached(last_word, maxd)    # [(w,dist), ...]
        freq_map = self._word_freq_map()                  # {w:count}

final = self.rankr.rank(markov=m_res,
                        embeddings=e_res,
                        fuzzy=f_res,
                        base_freq=freq_map,
                        recency_map=self.ctx.get_recency_map(),  # optional
                        topn=8)

    # reinforcement and tuning -------------------------------------
    def accept(self, word: str) -> None:
        """Record that the user accepted this suggestion (boost future ranking)."""
        if not word:
            return
        w = word.lower().strip()
        self.accepted[w] += 1
        # bound to avoid runaway numbers
        if self.accepted[w] > 1000:
            self.accepted[w] = 1000

    def set_balance(self, val: float) -> None:
        """Set blending weight (0..1) for embeddings vs markov."""
        try:
            v = float(val)
        except Exception:
            return
        self.alpha = max(0.0, min(1.0, v))

    # small utility for debugging/demo purpose
    def vocab_size(self) -> int:
        """Return an estimate of vocab size (BK-tree or markov)."""
        try:
            # BK-tree might have method, if so fallback to markov internals
            if hasattr(self.bk, "size"):
                return self.bk.size()
        except Exception:
            pass
        # fallback: count words seen in markov table if accessible
        try:
            # markov predictor stores counts, this is a best-effort estimate
            return sum(len(v) for v in getattr(self.mk, "_table", {}).values())
        except Exception:
            return 0
