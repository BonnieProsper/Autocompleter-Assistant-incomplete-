# hybrid_predictor.py
#
# Combines:
# - Markov (syntactic/context)
# - Embeddings (semantic)
# - BK-tree (fuzzy)
# - User personalization (bias + recent context)
# - FusionRanker for flexible weighting/presets
#
# Exposes train/retrain/suggest/accept and simple state save/load hooks.

import time
import pickle
import math
from collections import Counter, defaultdict, deque
from typing import List, Tuple, Dict, Optional

from markov_predictor import MarkovPredictor
from embeddings import Embeddings
from context_personal import CtxPersonal
from semantic_engine import SemanticEngine
from bktree import BKTree
from logger_utils import Log

# small local helper types
Scores = Dict[str, float]


class FusionRanker:
    """Weighted fusion of multiple signal sources into a single ranked list."""
    PRESETS = {
        "balanced": {"markov": 0.4, "embed": 0.4, "personal": 0.15, "freq": 0.05},
        "strict":   {"markov": 0.7, "embed": 0.15, "personal": 0.1,  "freq": 0.05},
        "personal": {"markov": 0.2, "embed": 0.2,  "personal": 0.55, "freq": 0.05},
        "semantic": {"markov": 0.1, "embed": 0.75, "personal": 0.1,  "freq": 0.05},
    }

    def __init__(self, preset: str = "balanced"):
        self.update_preset(preset)

    def update_preset(self, preset: str):
        if preset not in self.PRESETS:
            raise ValueError(f"bad preset '{preset}'")
        self.preset = preset
        self.weights = self.PRESETS[preset]

    def fuse(self, markov: Scores, embed: Scores,
             personal: Scores, freq: Scores) -> Scores:
        out: Scores = {}
        all_words = set(markov) | set(embed) | set(personal) | set(freq)
        for w in all_words:
            m = markov.get(w, 0.0)
            e = embed.get(w, 0.0)
            p = personal.get(w, 0.0)
            f = freq.get(w, 0.0)
            score = (self.weights["markov"] * m +
                     self.weights["embed"] * e +
                     self.weights["personal"] * p +
                     self.weights["freq"] * f)
            out[w] = score
        return out

    def rank(self, markov: Scores, embed: Scores,
             personal: Scores, freq: Scores, topn: int = 5) -> List[Tuple[str, float]]:
        combined = self.fuse(markov, embed, personal, freq)
        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:topn]


class HybridPredictor:
    """
    Top-level hybrid predictor.

    Usage pattern:
      hp = HybridPredictor(user="bonnie")
      hp.train(lines)            # train from corpus
      hp.retrain(sentence)       # incremental learning during CLI
      suggestions = hp.suggest("hello", topn=6)
      hp.accept("hello")         # user accepted suggestion -> reinforcement
      hp.save_state(path)        # optional persistence
    """

    def __init__(self, user: str = "default", context_window: int = 2,
                 bk_cache_ttl: float = 60.0):
        self.mk = MarkovPredictor()
        self.emb = Embeddings()
        self.ctx = CtxPersonal(user)
        self.sem = SemanticEngine()
        self.bk = BKTree()

        self.rank = FusionRanker("balanced")
        self.alpha = 0.5               # legacy balance (kept for compatibility)
        self.context_window = max(1, int(context_window))

        self._trained = False
        self.accepted = Counter()      # reinforcement counts
        self.freq_stats = Counter()    # surface frequency (from training)

        # BK cache for fuzzy queries: (q, maxd) -> [(word, dist), ...]
        self._bk_cache: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
        self._bk_cache_time: Dict[Tuple[str, int], float] = {}
        self._bk_cache_ttl = float(bk_cache_ttl)

    # ---------------------
    # Training / updating
    # ---------------------
    def train(self, lines: List[str]):
        if not lines:
            return
        with Log.time_block("Hybrid train"):
            for ln in lines:
                self._train_line(ln)
            self.ctx.save()
            self._trained = True

    def retrain(self, sentence: str):
        self._train_line(sentence)
        # save small increments; safe and quick
        try:
            self.ctx.save()
        except Exception:
            pass
        self._trained = True

    def _train_line(self, ln: str):
        self.mk.train_sentence(ln)
        self.ctx.learn(ln)
        for tok in self._tokens(ln):
            self.freq_stats[tok] += 1
            try:
                self.bk.insert(tok)
            except Exception:
                # ignore malformed token inserts; BK can be finicky
                pass

    # ---------------------
    # Helpers
    # ---------------------
    @staticmethod
    def _tokens(s: str) -> List[str]:
        return [t.lower() for t in s.strip().split() if t.isalpha()]

    def _adaptive_maxd(self, q: str) -> int:
        L = len(q)
        if L <= 3:
            return 0
        if L <= 6:
            return 1
        if L <= 12:
            return 2
        return 3

    def _bk_query_cached(self, q: str, maxd: int):
        key = (q, maxd)
        now = time.time()
        if key in self._bk_cache:
            if now - self._bk_cache_time.get(key, 0.0) < self._bk_cache_ttl:
                return self._bk_cache[key]
            # expired
            self._bk_cache.pop(key, None)
            self._bk_cache_time.pop(key, None)

        try:
            res = self.bk.query(q, max_dist=maxd)
        except Exception:
            res = []
        self._bk_cache[key] = res
        self._bk_cache_time[key] = now
        return res

    # ---------------------
    # Main suggestion API
    # ---------------------
    def suggest(self, text_fragment: str, topn: int = 6,
                fuzzy: bool = True) -> List[Tuple[str, float]]:
        """
        Provide ranked suggestions for the last token of text_fragment.
        Returns list of (word, score).
        """
        if not text_fragment:
            return []

        start = time.time()
        try:
            toks = [t.lower() for t in text_fragment.strip().split() if t]
            if not toks:
                return []

            # use recent token(s)
            context = toks[-self.context_window:]
            last = context[-1]

            # --- Markov suggestions (counts)
            mk_list = self.mk.top_next(last, topn=topn * 2)
            markov_scores: Scores = {w: float(s) for w, s in mk_list}

            # --- Embedding suggestions (similar words)
            emb_list = self.emb.similar(last, topn=topn * 2) if hasattr(self.emb, "similar") else []
            embed_scores: Scores = {w: float(s) for w, s in emb_list}

            # --- Personal scores (from CtxPersonal.freq)
            personal_scores: Scores = {w: float(self.ctx.freq.get(w, 0)) for w in set(list(markov_scores) + list(embed_scores))}
            # make sure we include some personal-only candidates (recent)
            for w in list(self.ctx.recent):
                personal_scores.setdefault(w, float(self.ctx.freq.get(w, 0)))

            # --- Frequency stats (global)
            freq_scores: Scores = {w: float(self.freq_stats.get(w, 0)) for w in set(list(markov_scores) + list(embed_scores) + list(personal_scores))}

            # --- BK fuzzy fallback (if needed)
            if fuzzy:
                maxd = self._adaptive_maxd(last)
                if maxd > 0:
                    fuzzy_res = self._bk_query_cached(last, maxd)
                    # fuzzy_res: (word, dist)
                    for w, dist in fuzzy_res:
                        bonus = max(0.0, (maxd + 1) - dist)
                        # attach to embed scores (small weight)
                        embed_scores[w] = max(embed_scores.get(w, 0.0), bonus * 0.5)

            # --- Reinforcement: bump accepted words
            for w, c in self.accepted.items():
                personal_scores[w] = personal_scores.get(w, 0.0) + math.log1p(c) * 0.4

            # --- Fusion + ranking
            ranked = self.rank.rank(markov_scores, embed_scores, personal_scores, freq_scores, topn=topn)

            # --- Bias again by CtxPersonal heuristics (keeps instincts)
            ranked = self.ctx.bias_words(ranked)

            # --- final trimming/normalisation
            out = [(w, round(float(sc), 3)) for w, sc in ranked][:topn]

            Log.metric("Suggest latency", round(time.time() - start, 3), "s")
            return out

        except Exception as e:
            Log.write(f"[HybridPredictor] suggest error: {e}")
            return []

    # ---------------------
    # Reinforcement & tuning
    # ---------------------
    def accept(self, word: str):
        if not word:
            return
        w = word.lower().strip()
        self.accepted[w] += 1
        # keep counts bounded
        if self.accepted[w] > 10_000:
            self.accepted[w] = 10_000

    def set_balance(self, val: float):
        # kept for backwards compatibility — maps roughly to embed vs markov
        self.alpha = max(0.0, min(1.0, float(val)))
        # convert to simple preset change if extremes
        if self.alpha < 0.25:
            self.rank.update_preset("strict")
        elif self.alpha > 0.75:
            self.rank.update_preset("semantic")
        else:
            self.rank.update_preset("balanced")

    def set_mode(self, preset: str):
        try:
            self.rank.update_preset(preset)
        except ValueError:
            raise

    def set_context_window(self, n: int):
        self.context_window = max(1, min(8, int(n)))

    # ---------------------
    # Persistence (optional helpers)
    # ---------------------
    def save_state(self, path: str):
        try:
            with open(path, "wb") as fh:
                pickle.dump({
                    "markov": self.mk,       # these objects are usually pickleable
                    "ctx": self.ctx,
                    "bk": self.bk,
                    "freq": self.freq_stats,
                    "accepted": self.accepted,
                    "rank_preset": self.rank.preset,
                }, fh)
            Log.write(f"[HybridPredictor] state saved -> {path}")
        except Exception as e:
            Log.write(f"[HybridPredictor] save_state failed: {e}")

    def load_state(self, path: str):
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            # replace local components if present in the file
            self.mk = data.get("markov", self.mk)
            self.ctx = data.get("ctx", self.ctx)
            self.bk = data.get("bk", self.bk)
            self.freq_stats = data.get("freq", self.freq_stats)
            self.accepted = data.get("accepted", self.accepted)
            preset = data.get("rank_preset")
            if preset:
                try:
                    self.rank.update_preset(preset)
                except Exception:
                    pass
            Log.write(f"[HybridPredictor] state loaded from {path}")
            self._trained = True
        except Exception as e:
            Log.write(f"[HybridPredictor] load_state failed: {e}")

    # ---------------------
    # Simple inspect/debug helpers
    # ---------------------
    def top_freq(self, n: int = 20) -> List[Tuple[str, int]]:
        return self.freq_stats.most_common(n)

    def stats_summary(self) -> Dict[str, int]:
        return {
            "trained": int(bool(self._trained)),
            "vocab": len(self.freq_stats),
            "accepted": sum(self.accepted.values()),
        }

        # fallback: count words seen in markov table if accessible
        try:
            # markov predictor stores counts, this is a best-effort estimate
            return sum(len(v) for v in getattr(self.mk, "_table", {}).values())
        except Exception:
            return 0
