# intelligent_autocompleter/core/hybrid_predictor.py
"""
HybridPredictor - manages signals and ranking for the Intelligent Autocompleter.
Purpose:
 - Collect signals from Markov, Semantic/Embeddings, BK-tree (fuzzy), per-user Personal context.
 - Cache BK queries for speed.
 - Produces ranked suggestions via FusionRanker. Uses ReinforcementLearner for
   feedback-driven weight adaptation and FeaturePreprocessor for consistent normalization.
 - Record user feedback (accepted/rejected) into ReinforcementLearner and nudge weights based on this feedback.
 - Offer explain() that returns component contributions for debugging and /explain CLI.
 - Support optional PluginRegistry for extension points.
"""

from __future__ import annotations

import os
import pickle
import time
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# check: remove local fallback/include it?
# imports: support package layout and local file-layout for tests/dev.
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
    # Local fallback
    # all types: ignore
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
    Orchestration for the autocompleter.

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
    _BK_CACHE_TTL_DEFAULT = 60.0  # seconds

    def __init__(
        self,
        user: str = "default",
        context_window: int = 2,
        registry: Optional[Any] = None,
        feedback_verbose: bool = False,
        ranker_preset: str = "balanced",
    ) -> None:
        # Core signal sources
        self.markov = MarkovPredictor()
        self.semantic = SemanticEngine()
        self.bk = BKTree()
        self.ctx = CtxPersonal(user)

        # Normalizer + feedback learner + base ranker
        self.pre = FeaturePreprocessor()
        self.reinforcement = ReinforcementLearner(verbose=feedback_verbose)
        # Base ranker: used when RL weights are unavailable or as fallback
        self.base_ranker = FusionRanker(preset=ranker_preset, personalizer=None)

        # Plugin registry (hooks: call_train, call_retrain, run_suggest_pipeline, call_accept, call_reject)
        self.registry = registry

        # Mutable state
        self.context_window = max(1, int(context_window))
        self.freq_stats: Dict[str, int] = {}
        self.accepted: Dict[str, int] = {}  #in memory counters for session reinforcement
        self._trained = False

        # BK caching to avoid repeated expensive queries
        self._bk_cache: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
        self._bk_cache_time: Dict[Tuple[str, int], float] = {}
        self._bk_cache_ttl = float(self._BK_CACHE_TTL_DEFAULT)

        # last suggestion -> inferred source mapping (used by feedback API)
        self._last_sources: Dict[str, str] = {}
        self.last_ranked: List[Candidate] = []

    # Training/incremental -------------------------------------------------
    def train(self, lines: Iterable[str]) -> None:
        """
        Bulk-train the internal components from an iterable of lines.
        Updates Markov, per user context, BK-tree and corpus frequency stats.
        """
        lines = list(lines or [])
        if not lines:
            return

        with Log.time_block("Hybrid.train"):
            for ln in lines:
                self._train_line(ln)

            # plugin hook 
            if self.registry:
                try:
                    self.registry.call_train(lines)
                except Exception as e:
                    Log.write(f"[Hybrid] plugin.train hook error: {e}")

            # try personal context persist
            try:
                self.ctx.save()
            except Exception:
                # no repercussions
                pass

            self._trained = True
            Log.write("[Hybrid] training complete")

    def retrain(self, sentence: str) -> None:
        """Incrementally train on a single sentence (e.g. accepted/recent user input)."""
        if not sentence:
            return
        self._train_line(sentence)
        if self.registry:
            try:
                self.registry.call_retrain(sentence)
            except Exception as e:
                Log.write(f"[Hybrid] plugin.retrain hook error: {e}")
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

        for tok in self._tokenize(ln):
            self.freq_stats[tok] = self.freq_stats.get(tok, 0) + 1
            try:
                # BK insertion may reject non-strings or raise. Ignore on failure
                self.bk.insert(tok)
            except Exception:
                pass

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        """
        Lightweight tokenizer used for Markov/BK update. Keeps it cheap and robust.
        """
        return [t.lower() for t in s.strip().split() if t.isalpha()]

    # ----------------------------
    # BK caching helpers
    # ----------------------------
    def _adaptive_maxd(self, token: str) -> int:
        """Heuristic controlling fuzzy search radius based on token length."""
        L = len(token or "")
        if L <= 3:
            return 0
        if L <= 6:
            return 1
        if L <= 12:
            return 2
        return 3

    def _bk_query_cached(self, q: str, maxd: int) -> List[Tuple[str, int]]:
        """
        Query BK tree with caching. Returns list of (word, distance).
        """
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

    # -------------------------
    # Core suggest pipeline
    # -------------------------
    def suggest(self, fragment: str, topn: int = 6, fuzzy: bool = True) -> List[Candidate]:
        """
        Produce top suggestions for the last token in 'fragment'.
        Pipeline:
         - Determine last token and short context.
         - Collect Markov next tokens and semantic similar tokens.
         - Query fuzzy BKtree (optionally).
         - Build per-feature raw maps and call FeaturePreprocessor.normalize_all.
         - Fetch live RL weights and instantiate FusionRanker (translating keys).
         - Rank via FusionRanker.
         - Optional plugin re-rank and personal bias via CtxPersonal.
         - Record last-sources mapping and return topn.
        """
        if not fragment or not fragment.strip():
            return []

        start_ts = time.time()

        try:
            toks = [t for t in fragment.strip().split() if t]
            if not toks:
                return []
            context = toks[-self.context_window:]
            last = context[-1].lower()

            # gather signals ----------------------------
            # Markov: top next words (counts)
            mk_list = self.markov.top_next(last, topn=topn * 3)
            markov_map: Dict[str, float] = {w: float(c) for w, c in mk_list}

            # Semantic: try to call '.similar(word, topn=...)' if implemented, fallback to '.search(...)'
            embed_list: List[Tuple[str, float]] = []
            try:
                emb_fn = getattr(self.semantic, "similar", None)
                if callable(emb_fn):
                    embed_list = emb_fn(last, topn=topn * 3)
                else:
                    hits = getattr(self.semantic, "search")(last, k=topn * 3)
                    embed_list = [(h, s) for h, s in hits]
            except Exception:
                embed_list = []
            embed_map: Dict[str, float] = {w: float(s) for w, s in embed_list}

            # Fuzzy BK candidates (distance)
            fuzzy_pairs: List[Tuple[str, int]] = []
            if fuzzy:
                maxd = self._adaptive_maxd(last)
                if maxd > 0:
                    fuzzy_pairs = self._bk_query_cached(last, maxd)

            # Add fuzzy-induced small semantic boosts (keeps fuzzy candidates competitive)
            if fuzzy_pairs:
                # compute single adaptive_maxd once for bonus calc
                am = self._adaptive_maxd(last) + 1
                for w, dist in fuzzy_pairs:
                    bonus = max(0.0, am - dist) * 0.4
                    # ensure we don't override a higher semantic score
                    embed_map[w] = max(embed_map.get(w, 0.0), float(bonus))

            # Personal & freq maps
            personal_candidates = set(markov_map.keys()) | set(embed_map.keys()) | set(self.ctx.recent)
            personal_map = {w: float(self.ctx.freq.get(w, 0.0)) for w in personal_candidates}
            freq_map = {w: float(self.freq_stats.get(w, 0)) for w in set(list(markov_map.keys()) + list(embed_map.keys()) + list(personal_map.keys()))}

            # normalize via FeaturePreprocessor (single source of truth)-----------------
            normalized = self.pre.normalize_all(
                markov=markov_map,
                embed=embed_map,
                fuzzy={w: d for w, d in fuzzy_pairs},
                freq=freq_map,
                recency={},  # recency timestamps could be passed here when available
            )

            if not normalized:
                return []

            # map RL weights to FusionRanker expected keys (safe mapping) -----------------
            rl_weights = {}
            try:
                rl_weights = self.reinforcement.get_weights()
            except Exception:
                rl_weights = {}

            ranker_weights = {
                "markov": float(rl_weights.get("markov", rl_weights.get("markov", 0.35))),
                "embed": float(rl_weights.get("semantic", rl_weights.get("embed", 0.45))),
                "personal": float(rl_weights.get("personal", 0.15)),
                # small defaults for other signals
                "freq": 0.05,
                "fuzzy": 0.05,
                "recency": 0.0,
            }

            # Build per-feature lists/maps required by FusionRanker
            words = sorted(normalized.keys())  # deterministic ordering
            markov_norm = {w: normalized[w].get("markov", 0.0) for w in words}
            embed_norm = {w: normalized[w].get("embed", 0.0) for w in words}
            fuzzy_map_for_ranker = {w: int(d) for w, d in fuzzy_pairs}
            freq_norm = {w: normalized[w].get("freq", 0.0) for w in words}
            recency_norm = {w: normalized[w].get("recency", 0.0) for w in words}

            # Use FusionRanker with the computed weights
            ranker = FusionRanker(weights=ranker_weights, personalizer=None)

            # convert to ranker expected inputs (lists or maps)
            mk_list_norm = list(markov_norm.items())
            emb_list_norm = list(embed_norm.items())
            fuzzy_list_norm = list((w, fuzzy_map_for_ranker.get(w, 999)) for w in words)

            ranked = ranker.rank(
                markov=mk_list_norm,
                embeddings=emb_list_norm,
                fuzzy=fuzzy_list_norm,
                base_freq=freq_norm,
                recency_map=recency_norm,
                topn=topn * 2,
            )

            # plugin pipeline ---------------------------
            if self.registry:
                try:
                    bundle = {"context": context, "user": getattr(self.ctx, "user", None), "last": last}
                    ranked = self.registry.run_suggest_pipeline(last, ranked, bundle)
                except Exception as e:
                    Log.write(f"[Hybrid] plugin suggest pipeline error: {e}")

            # personalization bias (CtxPersonal) if available ------------------------
            try:
                ranked = self.ctx.bias_words(ranked)
            except Exception:
                # ctx may implement different shapes so ignore gracefully
                pass

            # produce final list and source mapping ------------------------------------
            self._last_sources.clear()
            final: List[Candidate] = []
            seen = set()
            # iterate ranked suggestions and infer origin for feedback bookkeeping
            for w, score in ranked:
                if w in seen:
                    continue
                seen.add(w)
                if w in markov_map:
                    src = "markov"
                elif w in embed_map:
                    src = "semantic"
                elif personal_map.get(w, 0.0) > 0.0:
                    src = "personal"
                elif any(w == fw for fw, _ in fuzzy_pairs):
                    src = "fuzzy"
                else:
                    src = "plugin"
                self._last_sources[w] = src
                final.append((w, float(score)))

            out = [(w, round(float(sc), 6)) for w, sc in final][:topn]
            self.last_ranked = out

            Log.metric("hybrid.suggest_latency", round(time.time() - start_ts, 6), "s")
            return out

        except Exception as exc:
            Log.write(f"[HybridPredictor] suggest error: {exc}")
            return []

    # -------------------------
    # Feedback API
    # -------------------------
    def accept(self, word: str, context: Optional[str] = None, source: Optional[str] = None) -> None:
        """
        Record that a suggestion was accepted. Update in-memory counters and forwards
        the event to ReinforcementLearner for persistent tracking and weight updates.
        """
        if not word:
            return
        w = word.lower().strip()
        self.accepted[w] = self.accepted.get(w, 0) + 1
        if self.accepted[w] > 100_000:
            self.accepted[w] = 100_000

        resolved = source or self._last_sources.get(w, "semantic")
        try:
            self.reinforcement.record_accept(context or "", w, resolved)
        except Exception as e:
            Log.write(f"[Hybrid] reinforcement.record_accept failed: {e}")

        # plugin hook
        if self.registry:
            try:
                self.registry.call_accept(w, {"user": getattr(self.ctx, "user", None), "source": resolved})
            except Exception as e:
                Log.write(f"[Hybrid] plugin accept hook failed: {e}")

    def reject(self, word: str, context: Optional[str] = None, source: Optional[str] = None) -> None:
        """Record that a suggestion was rejected, forwards to ReinforcementLearner."""
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

    # ---------------------------------
    # Persistence & inspectability
    # -----------------------------------
    def save_state(self, path: str) -> None:
        """Pickle a compact snapshot of core models."""
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump(
                    {
                        "markov": self.markov,
                        "ctx": self.ctx,
                        "bk": self.bk,
                        "freq": self.freq_stats,
                        "accepted": self.accepted,
                    },
                    fh,
                )
            Log.write(f"[Hybrid] state saved -> {path}")
        except Exception as e:
            Log.write(f"[Hybrid] save_state failed: {e}")

    def load_state(self, path: str) -> None:
        """Load snapshot written by save_state."""
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
        """
        Return a structured explanation of signal contributions for 'fragment'.
        Useful for debugging and the '/explain' CLI.
        """
        toks = [t for t in fragment.strip().split() if t]
        if not toks:
            return {}
        last = toks[-1].lower()

        try:
            mk = dict(self.markov.top_next(last, topn=topn))
        except Exception:
            mk = {}
        try:
            emb_raw = getattr(self.semantic, "similar", lambda *_: [])(last, topn=topn)
            emb = dict(emb_raw)
        except Exception:
            emb = {}

        per = {w: self.ctx.freq.get(w, 0) for w in set(list(mk.keys()) + list(emb.keys()))}
        freq = {w: self.freq_stats.get(w, 0) for w in set(list(mk.keys()) + list(emb.keys()) + list(per.keys()))}

        try:
            fused = self.base_ranker.rank(
                markov=list(mk.items()),
                embeddings=list(emb.items()),
                fuzzy=[],
                base_freq=freq,
                recency_map={},
                topn=topn,
            )
        except Exception:
            fused = []

        return {"markov": mk, "embed": emb, "personal": per, "freq": freq, "fused": fused}

    def get_weights(self) -> Dict[str, float]:
        """Convenience to expose current reinforcement learner weights."""
        try:
            return self.reinforcement.get_weights()
        except Exception:
            return {"semantic": 0.45, "markov": 0.35, "personal": 0.15, "plugin": 0.05}
