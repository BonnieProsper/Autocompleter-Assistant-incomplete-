# intelligent_autocompleter/core/hybrid_predictor.py
"""
HybridPredictor - manages signals and ranking for the Intelligent Autocompleter.
Purpose:
 - Uses FeaturePreprocessor to normalize features maps (single, central place)
 - Collect signals from Markov, Semantic/Embeddings, BK-tree (fuzzy), per-user Personal context.
 - Cache BK queries for speed.
 - Produces ranked suggestions via FusionRanker. Uses ReinforcementLearner for
   feedback-driven weight adaptation and FeaturePreprocessor for consistent normalization.
 - Record user feedback (accepted/rejected) into ReinforcementLearner and nudge weights based on this feedback.
 - Offer explain() that returns component contributions for debugging and /explain CLI.
 - Support optional PluginRegistry for extension points.
 - Keeps an internal LRU (OrderedDict) cache for normalized feature maps to avoid repeated scaling.
 - Calls FusionRanker.rank_normalized(...) hot-path for speed and deterministic ordering.
 - Integrates ReinforcementLearner for live adaptive weights and feedback recording.
 - Deterministic-mode, debug_features() and explain() helpers for inspectability.
 
 e.g takes a fragment and returns complete ranked predictions
"""

from __future__ import annotations

import os
import pickle
import time
import math
from typing import Any, Dict, List, Optional, Tuple, Iterable
from collections import OrderedDict

# imports for packaged vs local dev
try:
    from intelligent_autocompleter.core.markov_predictor import MarkovPredictor
    from intelligent_autocompleter.core.bktree import BKTree
    from intelligent_autocompleter.core.semantic_engine import SemanticEngine
    from intelligent_autocompleter.core.fusion_ranker import FusionRanker
    from intelligent_autocompleter.core.reinforcement_learner import ReinforcementLearner
    from intelligent_autocompleter.core.feature_preprocessor import FeaturePreprocessor
    from intelligent_autocompleter.context_personal import CtxPersonal
    from intelligent_autocompleter.utils.logger_utils import Log
except Exception:
    # local fallback for unit tests or simpler layout
    from core.markov_predictor import MarkovPredictor
    from core.bktree import BKTree
    from core.semantic_engine import SemanticEngine
    from core.fusion_ranker import FusionRanker
    from core.reinforcement_learner import ReinforcementLearner
    from core.feature_preprocessor import FeaturePreprocessor
    from context_personal import CtxPersonal
    from utils.logger_utils import Log  # logger_utils (ensure it exposes write/metric)

Candidate = Tuple[str, float]


def _canonical_key_from_maps(markov: Dict[str, float],
                             embed: Dict[str, float],
                             fuzzy: Dict[str, int],
                             freq: Dict[str, float],
                             recency: Dict[str, float]) -> Tuple:
    """
    Canonicalize the maps into a deterministic key suitable for OrderedDict based LRU caching.
    Uses sorted tuples (word, value) to ensure deterministic ordering.
    """
    return (
        tuple(sorted(markov.items())),
        tuple(sorted(embed.items())),
        tuple(sorted(fuzzy.items())),
        tuple(sorted(freq.items())),
        tuple(sorted(recency.items())),
    )


class HybridPredictor:
    """
    Manage multiple signals (markov, semantic, fuzzy, personal) and return ranked suggestions.

    Public API:
      - train(lines)
      - retrain(sentence)
      - suggest(fragment, topn=6, fuzzy=True)
      - accept(word, context=None, source=None)
      - reject(word, context=None, source=None)
      - explain(fragment)
      - debug_features(fragment)
      - save_state(path)/load_state(path)
    """

    def __init__(self,
                 user: str = "default",
                 context_window: int = 2,
                 registry: Optional[Any] = None,
                 feedback_verbose: bool = False,
                 ranker_preset: str = "balanced",
                 normalized_cache_size: int = 4096):
        # Core components
        self.markov = MarkovPredictor()
        self.semantic = SemanticEngine()
        self.bk = BKTree()
        self.ctx = CtxPersonal(user)

        # Utilities
        self.pre = FeaturePreprocessor()
        self.reinforcement = ReinforcementLearner(verbose=feedback_verbose)
        self.base_ranker = FusionRanker(preset=ranker_preset)

        # Plugin registry
        self.registry = registry

        # State
        self.context_window = max(1, int(context_window))
        self.freq_stats: Dict[str, int] = {}
        self.accepted: Dict[str, int] = {}
        self._trained = False

        # BK caching
        self._bk_cache: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
        self._bk_cache_time: Dict[Tuple[str, int], float] = {}
        self._bk_cache_ttl = 60.0

        # Normalized-feature LRU cache (OrderedDict used as simple LRU)
        self._normalized_cache: "OrderedDict[Tuple, Dict[str, Dict[str, float]]]" = OrderedDict()
        self._normalized_cache_max = max(512, int(normalized_cache_size))

        # Last suggestion sources (for feedback resolution)
        self._last_sources: Dict[str, str] = {}
        self.last_suggestions: List[Candidate] = []

        # Deterministic/test mode toggles
        self._deterministic = False

    # -------------------------
    # Training/incremental
    # -------------------------
    def train(self, lines: Iterable[str]) -> None:
        lines = list(lines) if lines is not None else []
        if not lines:
            return
        with Log.time_block("Hybrid.train"):
            for ln in lines:
                self._train_line(ln)
            if self.registry:
                try:
                    self.registry.call_train(lines)
                except Exception as e:
                    Log.write(f"[Hybrid] plugin train hook failed: {e}")
            try:
                self.ctx.save()
            except Exception:
                pass
            self._trained = True
            Log.write("[Hybrid] training complete")

    def retrain(self, sentence: str) -> None:
        self._train_line(sentence)
        if self.registry:
            try:
                self.registry.call_retrain(sentence)
            except Exception as e:
                Log.write(f"[Hybrid] plugin retrain hook failed: {e}")
        try:
            self.ctx.save()
        except Exception:
            pass
        self._trained = True

    def _train_line(self, ln: str) -> None:
        if not ln or not isinstance(ln, str):
            return
        self.markov.train_sentence(ln)
        self.ctx.learn(ln)
        for t in self._tokens(ln):
            self.freq_stats[t] = self.freq_stats.get(t, 0) + 1
            try:
                self.bk.insert(t)
            except Exception:
                # ignore invalid tokens for BK tree
                pass

    @staticmethod
    def _tokens(s: str) -> List[str]:
        return [t.lower() for t in s.strip().split() if t.isalpha()]

    # -------------------------
    # BK caching
    # -------------------------
    def _adaptive_maxd(self, q: str) -> int:
        L = len(q)
        if L <= 3:
            return 0
        if L <= 6:
            return 1
        if L <= 12:
            return 2
        return 3

    def _bk_query_cached(self, q: str, maxd: int) -> List[Tuple[str, int]]:
        key = (q, maxd)
        now = time.time()
        if key in self._bk_cache and (now - self._bk_cache_time.get(key, 0.0)) < self._bk_cache_ttl:
            return self._bk_cache[key]
        try:
            res = self.bk.query(q, max_dist=maxd)
        except Exception:
            res = []
        self._bk_cache[key] = res
        self._bk_cache_time[key] = now
        return res

    # ------------------------------------
    # Normalized feature cache (simple LRU)
    # --------------------------------------
    def _get_normalized_from_cache(self, key: Tuple) -> Optional[Dict[str, Dict[str, float]]]:
        if key in self._normalized_cache:
            # move to end -> mark as recently used
            self._normalized_cache.move_to_end(key)
            return self._normalized_cache[key]
        return None

    def _put_normalized_to_cache(self, key: Tuple, value: Dict[str, Dict[str, float]]) -> None:
        self._normalized_cache[key] = value
        self._normalized_cache.move_to_end(key)
        # evict oldest if over capacity
        while len(self._normalized_cache) > self._normalized_cache_max:
            self._normalized_cache.popitem(last=False)

    # -------------------------
    # Suggest API (hot-path)
    # -------------------------
    def suggest(self, fragment: str, topn: int = 6, fuzzy: bool = True) -> List[Candidate]:
        """
        Suggest completions for the last token in 'fragment'.

        Steps:
         - derive last token
         - gather Markov and semantic candidates
         - collect fuzzy (BK) candidates using adaptive maxd
         - build raw feature maps
         - lookup cached normalized feature maps or compute + cache
         - call FusionRanker.rank_normalized for efficient ranking
         - apply plugin pipeline and personal bias
         - populate last_sources mapping and return topn results
        """
        if not fragment or not fragment.strip():
            return []

        start = time.time()
        try:
            toks = [t for t in fragment.strip().split() if t]
            if not toks:
                return []
            context = toks[-self.context_window:]
            last = context[-1].lower()

            # Markov
            mk_list = self.markov.top_next(last, topn=topn * 3)
            markov_map = {w: float(c) for w, c in mk_list}

            # Semantic embeddings (try .similar first)
            embed_list = []
            try:
                emb_fn = getattr(self.semantic, "similar", None)
                if callable(emb_fn):
                    embed_list = emb_fn(last, topn=topn * 3)
                else:
                    hits = self.semantic.search(last, k=topn * 3)
                    embed_list = [(h, s) for h, s in hits]
            except Exception:
                embed_list = []
            embed_map = {w: float(s) for w, s in embed_list}

            # Fuzzy via BKTree
            fuzzy_pairs: List[Tuple[str, int]] = []
            if fuzzy:
                maxd = self._adaptive_maxd(last)
                if maxd > 0:
                    fuzzy_pairs = self._bk_query_cached(last, maxd)

            # incorporate fuzzy as small semantic boost (so fuzzy candidates get represented in embed_map)
            for w, dist in fuzzy_pairs:
                bonus = max(0.0, (self._adaptive_maxd(last) + 1) - dist) * 0.4
                embed_map[w] = max(embed_map.get(w, 0.0), float(bonus))

            # personal and freq
            personal_candidates = set(list(markov_map.keys()) + list(embed_map.keys())) | set(self.ctx.recent)
            personal_map = {w: float(self.ctx.freq.get(w, 0.0)) for w in personal_candidates}
            freq_map = {w: float(self.freq_stats.get(w, 0)) for w in set(list(markov_map.keys()) + list(embed_map.keys()) + list(personal_map.keys()))}

            # Normalization: try LRU cache first
            recency_map = {}  # placeholder: if you track timestamps, pass them here
            cache_key = _canonical_key_from_maps(markov_map, embed_map, {w: d for w, d in fuzzy_pairs}, freq_map, recency_map)

            normalized = None
            if not self._deterministic:
                normalized = self._get_normalized_from_cache(cache_key)

            if normalized is None:
                normalized = self.pre.normalize_all(markov=markov_map,
                                                    embed=embed_map,
                                                    fuzzy={w: d for w, d in fuzzy_pairs},
                                                    freq=freq_map,
                                                    recency=recency_map)
                # If deterministic mode, do not cache (keeps tests reproducible if needed)
                if not self._deterministic:
                    try:
                        self._put_normalized_to_cache(cache_key, normalized)
                    except Exception:
                        # cache is best-effort
                        pass

            if not normalized:
                return []

            # Fetch live weights and map to ranker names
            try:
                rl_weights = self.reinforcement.get_weights()
            except Exception:
                rl_weights = {}

            ranker_weights = {
                "markov": float(rl_weights.get("markov", rl_weights.get("markov", 0.35))),
                "embed": float(rl_weights.get("semantic", rl_weights.get("embed", 0.45))),
                "personal": float(rl_weights.get("personal", 0.15)),
                "freq": float(0.05),
                "fuzzy": float(0.05),
                "recency": float(0.0),
            }

            # Rank using FusionRanker hot-path
            ranker = FusionRanker(weights=ranker_weights)
            ranked = ranker.rank_normalized(normalized, weights=ranker_weights, personalizer=self.ctx, topn=topn * 2)

            # Plugin pipeline
            if self.registry:
                try:
                    bundle = {"context": context, "user": getattr(self.ctx, "user", None), "last": last}
                    # registry may return list[(word,score)]
                    ranked = self.registry.run_suggest_pipeline(last, ranked, bundle)
                except Exception as e:
                    Log.write(f"[Hybrid] plugin suggest pipeline error: {e}")

            # ensure personal bias applied (CtxPersonal.bias_words expects list[(w,score)])
            try:
                ranked = self.ctx.bias_words(ranked)
            except Exception:
                pass

            # Build last_sources mapping for feedback and produce final list
            self._last_sources.clear()
            final: List[Candidate] = []
            seen = set()
            fuzzy_set = {w for w, _ in fuzzy_pairs}
            for w, score in ranked:
                if w in seen:
                    continue
                seen.add(w)
                if w in markov_map:
                    src = "markov"
                elif w in embed_map:
                    src = "semantic"
                elif w in personal_map and personal_map.get(w, 0.0) > 0.0:
                    src = "personal"
                elif w in fuzzy_set:
                    src = "fuzzy"
                else:
                    src = "plugin"
                self._last_sources[w] = src
                final.append((w, float(score)))

            out = [(w, round(float(sc), 4)) for w, sc in final][:topn]
            self.last_suggestions = out

            Log.metric("hybrid.suggest_latency", round(time.time() - start, 6), "s")
            return out

        except Exception as e:
            Log.write(f"[HybridPredictor] suggest error: {e}")
            return []

    # -------------------------
    # Feedback API
    # -------------------------
    def accept(self, word: str, context: Optional[str] = None, source: Optional[str] = None) -> None:
        if not word:
            return
        w = word.lower().strip()
        self.accepted[w] = self.accepted.get(w, 0) + 1
        if self.accepted[w] > 100000:
            self.accepted[w] = 100000

        resolved = source or self._last_sources.get(w, "semantic")
        try:
            self.reinforcement.record_accept(context or "", w, resolved)
        except Exception as e:
            Log.write(f"[Hybrid] reinforcement record_accept failed: {e}")

        if self.registry:
            try:
                self.registry.call_accept(w, {"user": getattr(self.ctx, "user", None), "source": resolved})
            except Exception as e:
                Log.write(f"[Hybrid] plugin accept hook failed: {e}")

    def reject(self, word: str, context: Optional[str] = None, source: Optional[str] = None) -> None:
        if not word:
            return
        w = word.lower().strip()
        resolved = source or self._last_sources.get(w, "semantic")
        try:
            self.reinforcement.record_reject(context or "", w, resolved)
        except Exception:
            pass
        if self.registry:
            try:
                if hasattr(self.registry, "call_reject"):
                    self.registry.call_reject(w, {"user": getattr(self.ctx, "user", None), "source": resolved})
            except Exception as e:
                Log.write(f"[Hybrid] plugin reject hook failed: {e}")

    # -------------------------
    # Explain & debug helpers
    # -------------------------
    def explain(self, fragment: str, topn: int = 6) -> Dict[str, Any]:
        toks = [t for t in fragment.strip().split() if t]
        if not toks:
            return {}
        last = toks[-1].lower()
        try:
            mk = dict(self.markov.top_next(last, topn=topn))
        except Exception:
            mk = {}
        try:
            emb = getattr(self.semantic, "similar", lambda *_: [])(last, topn=topn)
            emb = dict(emb)
        except Exception:
            emb = {}
        per = {w: self.ctx.freq.get(w, 0) for w in set(list(mk.keys()) + list(emb.keys()))}
        freq = {w: self.freq_stats.get(w, 0) for w in set(list(mk.keys()) + list(emb.keys()) + list(per.keys()))}
        try:
            fused = self.base_ranker.rank(markov=list(mk.items()), embeddings=list(emb.items()), fuzzy=[], base_freq=freq, recency_map={}, topn=topn)
        except Exception:
            fused = []
        return {"markov": mk, "embed": emb, "personal": per, "freq": freq, "fused": fused}

    def debug_features(self, fragment: str, topn: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Return normalized feature map and contribution debug info for the top candidates for a fragment.
        For unit tests and explainability in the UI.
        """
        toks = [t for t in fragment.strip().split() if t]
        if not toks:
            return {}
        # Use same internal pipeline but request a larger topn and return normalized + contributions
        raw = self.suggest(fragment, topn=topn, fuzzy=True)
        words = [w for w, _ in raw]
        # Reconstruct normalized cache entry if available (best-effort)
        mk_list = self.markov.top_next(toks[-1].lower(), topn=topn * 3)
        markov_map = {w: float(c) for w, c in mk_list}
        embed_list = []
        try:
            emb_fn = getattr(self.semantic, "similar", None)
            if callable(emb_fn):
                embed_list = emb_fn(toks[-1].lower(), topn=topn * 3)
            else:
                embed_list = [(h, s) for h, s in self.semantic.search(toks[-1].lower(), k=topn * 3)]
        except Exception:
            embed_list = []
        embed_map = {w: float(s) for w, s in embed_list}
        fuzzy_pairs = []
        maxd = self._adaptive_maxd(toks[-1].lower())
        if maxd > 0:
            fuzzy_pairs = self._bk_query_cached(toks[-1].lower(), maxd)
        freq_map = {w: float(self.freq_stats.get(w, 0)) for w in set(list(markov_map.keys()) + list(embed_map.keys()))}
        normalized = self.pre.normalize_all(markov=markov_map, embed=embed_map, fuzzy={w: d for w, d in fuzzy_pairs}, freq=freq_map, recency={})
        contributions = {}
        weights = self.reinforcement.get_weights()
        ranker_weights = {
            "markov": float(weights.get("markov", 0.35)),
            "embed": float(weights.get("semantic", weights.get("embed", 0.45))),
            "personal": float(weights.get("personal", 0.15)),
            "freq": 0.05,
            "fuzzy": 0.05,
            "recency": 0.0,
        }
        from intelligent_autocompleter.core.fusion_ranker import FusionRanker
        fr = FusionRanker(weights=ranker_weights)
        for w in words:
            contributions[w] = fr.debug_contributions(w, normalized, ranker_weights)
        return contributions

    # -------------------------
    # Persistence
    # -------------------------
    def save_state(self, path: str) -> None:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump({
                    "markov": self.markov,
                    "ctx": self.ctx,
                    "bk": self.bk,
                    "freq": self.freq_stats,
                    "accepted": self.accepted,
                }, fh)
            Log.write(f"[Hybrid] state saved -> {path}")
        except Exception as e:
            Log.write(f"[Hybrid] save_state failed: {e}")

    def load_state(self, path: str) -> None:
        if not os.path.exists(path):
            Log.write(f"[Hybrid] load_state: path not found {path}")
            return
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            self.markov = data.get("markov", self.markov)
            self.ctx = data.get("ctx", self.ctx)
            self.bk = data.get("bk", self.bk)
            self.freq_stats = data.get("freq", self.freq_stats)
            self.accepted = data.get("accepted", self.accepted)
            self._trained = True
            Log.write(f"[Hybrid] state loaded from {path}")
        except Exception as e:
            Log.write(f"[Hybrid] load_state failed: {e}")

    # -------------------------
    # Utility: deterministic mode
    # -------------------------
    def enable_deterministic_mode(self) -> None:
        """
        Use deterministic mode for testing:
         - disable caching
         - disable learning side-effects (notably RL write-through)
        """
        self._deterministic = True
        self._bk_cache_ttl = 0.0
        # best-effort: if reinforcement learner supports a disable/readonly mode, call it
        try:
            if hasattr(self.reinforcement, "disable_learning"):
                self.reinforcement.disable_learning()
        except Exception:
            pass
