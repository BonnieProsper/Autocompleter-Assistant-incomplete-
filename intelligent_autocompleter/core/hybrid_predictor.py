# intelligent_autocompleter/core/hybrid_predictor.py
"""
HybridPredictor - manages signals and ranking for the Intelligent Autocompleter.
Purpose:
 - Uses FeaturePreprocessor to normalize features
 - Collect signals from Markov, Semantic/Embeddings, BK-tree (fuzzy), per-user Personal context.
 - Cache BK queries for speed.
 - Produces ranked suggestions via FusionRanker. Uses ReinforcementLearner for
   feedback-driven weight adaptation and FeaturePreprocessor for consistent normalization.
 - Record user feedback (accepted/rejected) into ReinforcementLearner and nudge weights based on this feedback.
 - Offer explain() that returns component contributions for debugging and /explain CLI.
 - Support optional PluginRegistry for extension points.
 - Calls FusionRanker.rank_normalized(...) on the hot path (fast, allocation-minimised).
 - Deterministic ordering and robust plugin/feedback hooks.
 e.g take a fragment and return complete ranked predictions
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

# imports (packaged or local dev)
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
    # local dev fallback, all type: ignore
    from core.markov_predictor import MarkovPredictor  
    from core.bktree import BKTree  
    from core.semantic_engine import SemanticEngine  
    from core.fusion_ranker import FusionRanker  
    from core.reinforcement_learner import ReinforcementLearner
    from core.feature_preprocessor import FeaturePreprocessor  
    from context_personal import CtxPersonal  
    from utils.logger_utils import Log  

Candidate = Tuple[str, float]


class HybridPredictor:
    """
    Predictor that fuses signals from markov, semantic, fuzzy, and personal context.
   
    Public methods:
      - train(lines: Iterable[str]) -> None
      - retrain(sentence: str) -> None
      - suggest(fragment: str, topn: int = 6, fuzzy: bool = True) -> List[(word, score)]
      - accept(word: str, context: Optional[str] = None, source: Optional[str] = None) -> None
      - reject(word: str, context: Optional[str] = None, source: Optional[str] = None) -> None
      - explain(fragment: str, topn: int = 6) -> Dict[str, Any]
      - save_state(path: str)/load_state(path: str)

    Usage:
      - Pipeline is deterministic: gather signals -> preprocess -> rank -> postprocess.
      - FeaturePreprocessor centralises normalization, FusionRanker does the weighted fusion.
      - ReinforcementLearner provides live weights and persists feedback.
      - Plugins are supported through an optional registry (best-effort and sandboxed).
    """

    def __init__(self,
                 user: str = "default",
                 context_window: int = 2,
                 registry: Optional[Any] = None,
                 feedback_verbose: bool = False,
                 ranker_preset: str = "balanced"):
        # core models
        self.markov = MarkovPredictor()
        self.semantic = SemanticEngine()
        self.bk = BKTree()
        self.ctx = CtxPersonal(user)

        # utilities
        self.pre = FeaturePreprocessor()
        self.reinforcement = ReinforcementLearner(verbose=feedback_verbose)
        self.base_ranker = FusionRanker(preset=ranker_preset, personalizer=self.ctx)

        # optional plugin registry
        self.registry = registry

        # state
        self.context_window = max(1, int(context_window))
        self.freq_stats: Dict[str, float] = {}
        self.accepted: Dict[str, int] = {}
        self._trained = False

        # BK cache
        self._bk_cache: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
        self._bk_cache_time: Dict[Tuple[str, int], float] = {}
        self._bk_cache_ttl = 60.0

        # last suggestion -> source (for feedback resolution)
        self._last_sources: Dict[str, str] = {}
        self.last_ranked: List[Candidate] = []

    # Training ----------------------------------------------
    def train(self, lines: List[str]) -> None:
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
                pass

    @staticmethod
    def _tokens(s: str) -> List[str]:
        return [t.lower() for t in s.strip().split() if t.isalpha()]

    # BK caching -------------------------------------------
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

    # suggest API (hot path: rank_normalized) -----------------------------
    def suggest(self, fragment: str, topn: int = 6, fuzzy: bool = True) -> List[Candidate]:
        """
        Suggest completions for last token in fragment.

        Fast path: build normalized per-feature maps via FeaturePreprocessor and call
        FusionRanker.rank_normalized(...) using live RL weights.
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

            # gather raw signals
            mk_list = self.markov.top_next(last, topn=topn * 3)
            markov_map_raw = {w: float(c) for w, c in mk_list}

            # semantic/embedding candidates
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
            embed_map_raw = {w: float(s) for w, s in embed_list}

            # fuzzy BK
            fuzzy_pairs: List[Tuple[str, int]] = []
            if fuzzy:
                maxd = self._adaptive_maxd(last)
                if maxd > 0:
                    fuzzy_pairs = self._bk_query_cached(last, maxd)

            # incorporate fuzzy as small semantic boost in raw embed map (so fuzzy words are present)
            for w, dist in fuzzy_pairs:
                bonus = max(0.0, (self._adaptive_maxd(last) + 1) - dist) * 0.4
                embed_map_raw[w] = max(embed_map_raw.get(w, 0.0), float(bonus))

            # personal/freq
            personal_candidates = set(list(markov_map_raw.keys()) + list(embed_map_raw.keys())) | set(self.ctx.recent)
            personal_map_raw = {w: float(self.ctx.freq.get(w, 0.0)) for w in personal_candidates}
            freq_map_raw = {w: float(self.freq_stats.get(w, 0)) for w in set(list(markov_map_raw.keys()) + list(embed_map_raw.keys()) + list(personal_map_raw.keys()))}

            # Preprocess -> normalized per-word feature dicts
            normalized = self.pre.normalize_all(
                markov=markov_map_raw,
                embed=embed_map_raw,
                fuzzy={w: dist for w, dist in fuzzy_pairs},
                freq=freq_map_raw,
                recency={},  # not tracking per-word timestamps here
            )

            if not normalized:
                return []

            # Unpack normalized maps expected by FusionRanker.rank_normalized
            words = sorted(normalized.keys())
            markov_norm = {w: normalized[w].get("markov", 0.0) for w in words}
            embed_norm = {w: normalized[w].get("embed", 0.0) for w in words}
            fuzzy_norm = {w: normalized[w].get("fuzzy", 0.0) for w in words}
            freq_norm = {w: normalized[w].get("freq", 0.0) for w in words}
            recency_norm = {w: normalized[w].get("recency", 0.0) for w in words}

            # get live RL weights and map to ranker names
            rl_weights = self.reinforcement.get_weights()
            ranker_weights = {
                "markov": float(rl_weights.get("markov", 0.35)),
                "embed": float(rl_weights.get("semantic", rl_weights.get("embed", 0.45))),
                "personal": float(rl_weights.get("personal", 0.15)),
                "freq": 0.05,
                "fuzzy": 0.05,
                "recency": 0.0,
            }

            # construct FusionRanker with live weights (fast path)
            ranker = FusionRanker(weights=ranker_weights, personalizer=self.ctx)

            # call the fast normalized ranking API
            ranked = ranker.rank_normalized(markov_norm, embed_norm, fuzzy_norm, freq_norm, recency_norm, topn=topn * 2)

            # plugin pipeline (best-effort)
            if self.registry:
                try:
                    bundle = {"context": context, "user": getattr(self.ctx, "user", None), "last": last}
                    ranked = self.registry.run_suggest_pipeline(last, ranked, bundle)
                except Exception as e:
                    Log.write(f"[Hybrid] plugin suggest pipeline error: {e}")

            # personal bias (ctx.bias_words expects list[(w,score)])
            try:
                ranked = self.ctx.bias_words(ranked)
            except Exception:
                # ignore if ctx uses different API
                pass

            # build last source mapping (prefer markov > embed > personal > fuzzy > plugin)
            self._last_sources.clear()
            final: List[Candidate] = []
            seen = set()
            # original raw maps for source inference
            for w, score in ranked:
                if w in seen:
                    continue
                seen.add(w)
                if w in markov_map_raw:
                    src = "markov"
                elif w in embed_map_raw:
                    src = "semantic"
                elif w in personal_map_raw and personal_map_raw.get(w, 0.0) > 0.0:
                    src = "personal"
                elif any(w == fw for fw, _ in fuzzy_pairs):
                    src = "fuzzy"
                else:
                    src = "plugin"
                self._last_sources[w] = src
                final.append((w, float(score)))

            out = [(w, round(float(sc), 4)) for w, sc in final][:topn]
            self.last_ranked = out

            Log.metric("hybrid.suggest_latency", round(time.time() - start, 4), "s")
            return out

        except Exception as e:
            Log.write(f"[HybridPredictor] suggest error: {e}")
            return []

    # Feedback -----------------------------------------------------------
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

    # Persistence & explain -------------------------------------------
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

    def get_weights(self) -> Dict[str, float]:
        try:
            return self.reinforcement.get_weights()
        except Exception:
            return {"semantic": 0.45, "markov": 0.35, "personal": 0.15, "plugin": 0.05}
