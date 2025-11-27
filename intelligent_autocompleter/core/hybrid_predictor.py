# intelligent_autocompleter/core/hybrid_predictor.py
"""
HybridPredictor - manages signals and ranking for the Intelligent Autocompleter.
Purpose:
 - Collect signals from Markov, Semantic/Embeddings, BK-tree (fuzzy), Personal context.
 - Cache BK queries for speed.
 - Ask AdaptiveLearner (change to reinforcementlearner) for live weights if available, pass them to FusionRanker.
 - Record user feedback (accepted/rejected) into FeedbackTracker (change to reinforcementlearner?) and nudge AdaptiveLearner.
 - Offer explain() that returns component contributions for debugging and /explain CLI.
 - Support optional PluginRegistry for extension points.
"""

from __future__ import annotations
import time
import math
import pickle
import os
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any

# best-effort imports (package layout or local dev)
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
    # local-run fallback
    from core.markov_predictor import MarkovPredictor
    from core.bktree import BKTree
    from core.semantic_engine import SemanticEngine
    from core.fusion_ranker import FusionRanker, FusionStrategy
    from core.feature_preprocessor import FeaturePreprocessor
    from core.reinforcement_learner import ReinforcementLearner
    from context_personal import CtxPersonal
    from utils.logger_utils import Log

# plugin registry is optional
try:
    from intelligent_autocompleter.plugins.registry import PluginRegistry
except Exception:
    PluginRegistry = None  # optional


Candidate = Tuple[str, float]


class HybridPredictor:
    def __init__(self,
                 user: str = "default",
                 context_window: int = 2,
                 registry: Optional[Any] = None,
                 feedback_verbose: bool = False,
                 ranker_preset: str = "balanced"):
        # core signal models
        self._markov = MarkovPredictor()
        self._semantic = SemanticEngine()
        self._bk = BKTree()
        self._ctx = CtxPersonal(user)

        # ranker and learner
        self._base_ranker = FusionRanker(preset=ranker_preset, personalizer=self._ctx)
        self._learner = AdaptiveLearner()
        # feedback tracker writes to disk and notifies learner; quiet by default
        self._feedback = FeedbackTracker(learner=self._learner, verbose=feedback_verbose)

        # plugin registry (optional)
        self.registry = registry

        # state
        self.context_window = max(1, int(context_window))
        self.accepted = Counter()     # local accept counts for quick reinforcement
        self.freq_stats = Counter()   # global frequency (corpus)
        self._trained = False

        # BK cache (q,maxd) => list[(word,dist)]
        self._bk_cache: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
        self._bk_cache_time: Dict[Tuple[str, int], float] = {}
        self._bk_cache_ttl = 60.0

        # source mapping for last suggestions (word -> inferred source)
        self._last_sources: Dict[str, str] = {}
        self._last_suggestions: List[Candidate] = []

    # training/incremental learning -------------------------
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
                    Log.write(f"[Hybrid] plugin train hook error: {e}")
            try:
                self._ctx.save()
            except Exception:
                pass
            self._trained = True
            Log.write("[Hybrid] training complete")

    def retrain(self, sentence: str) -> None:
        # incremental update (user typed or accepted a custom token)
        self._train_line(sentence)
        if self.registry:
            try:
                self.registry.call_retrain(sentence)
            except Exception as e:
                Log.write(f"[Hybrid] plugin retrain hook error: {e}")
        try:
            self._ctx.save()
        except Exception:
            pass
        self._trained = True

    def _train_line(self, ln: str) -> None:
        if not ln or not isinstance(ln, str):
            return
        self._markov.train_sentence(ln)
        self._ctx.learn(ln)
        for tok in self._tokenize(ln):
            self.freq_stats[tok] += 1
            try:
                self._bk.insert(tok)
            except Exception:
                # ignore bad tokens that bk-tree can't handle
                pass

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        return [t.lower() for t in s.strip().split() if t.isalpha()]

    def _adaptive_maxd(self, token: str) -> int:
        L = len(token)
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
            res = self._bk.query(q, max_dist=maxd)
        except Exception:
            res = []
        self._bk_cache[key] = res
        self._bk_cache_time[key] = now
        return res

    # main suggestion API -----------------------------------
    def suggest(self, text_fragment: str, topn: int = 6, fuzzy: bool = True) -> List[Candidate]:
        """
        Produce suggestions for the last token in text_fragment.
        Returns list[(word, score)] ordered by score desc.
        Also fills internal _last_sources map for feedback resolution.
        """
        if not text_fragment:
            return []
        start = time.time()

        try:
            toks = [t.lower() for t in text_fragment.strip().split() if t]
            if not toks:
                return []
            context = toks[-self.context_window:]
            last = context[-1]

            # Markov candidates: (word, count)
            mk_list = self._markov.top_next(last, topn=topn * 3)
            # embeddings/semantic similar words (word,score)
            emb_list = []
            try:
                # semantic engine exposes a 'search similar words' method; if not, skip
                emb_list = getattr(self._semantic, "similar", lambda *_: [])(last, topn=topn * 3)
            except Exception:
                try:
                    # fallback to SemanticEngine.search (sentence-level) mapped to words
                    emb_hits = self._semantic.search(last, k=topn * 3)
                    emb_list = [(h, s) for h, s in emb_hits]
                except Exception:
                    emb_list = []

            # personal candidates & scores from user context
            personal_candidates = set([w for w, _ in mk_list] + [w for w, _ in emb_list]) | set(self._ctx.recent)
            personal_scores = {w: float(self._ctx.freq.get(w, 0.0)) for w in personal_candidates}

            # global freq
            freq_scores = {w: float(self.freq_stats.get(w, 0)) for w in set(list(dict(mk_list)) | set(dict(emb_list)) | set(personal_scores))}

            # fuzzy results -> incorporate as small embedding boost
            fuzzy_pairs: List[Tuple[str, int]] = []
            if fuzzy:
                maxd = self._adaptive_maxd(last)
                if maxd > 0:
                    fuzzy_pairs = self._bk_query_cached(last, maxd)
                    for w, dist in fuzzy_pairs:
                        bonus = max(0.0, (maxd + 1) - dist)
                        # boost embeddings slightly so fuzzy shows up in ranker
                        emb_map_val = dict(emb_list).get(w, 0.0)
                        emb_map_val = max(emb_map_val, bonus * 0.4)
                        # update or append into emb_list structure (rebuild list)
                        # we'll reconstruct embeddings list from a combined map below

            # build maps for ranker
            markov_map = {w: float(c) for w, c in mk_list}
            embed_map = {w: float(s) for w, s in emb_list}
            # add fuzzy-induced boosts into embed_map
            for w, dist in fuzzy_pairs:
                bonus = max(0.0, (self._adaptive_maxd(last) + 1) - dist) * 0.4
                embed_map[w] = max(embed_map.get(w, 0.0), float(bonus))

            # reinforcement from in-session accepted words
            for w, c in self.accepted.items():
                personal_scores[w] = personal_scores.get(w, 0.0) + math.log1p(c) * 0.3

            # choose weights from Reinforcementlearner if available
            weights_override = None
            try:
                weights_override = self._learner.get_weights()
            except Exception:
                weights_override = None

            # prepare FusionRanker - if learner exists, pass weights
            try:
                if weights_override:
                    # convert learner mapping (semantic/markov/personal/plugin) to ranker weights
                    ranker_weights = {
                        "markov": weights_override.get("markov", 0.0),
                        "embed": weights_override.get("semantic", weights_override.get("embed", 0.0)),
                        "personal": weights_override.get("personal", 0.0),
                        "freq": 0.05,
                        "fuzzy": 0.05,
                        "recency": 0.0,
                    }
                    ranker = FusionRanker(weights=ranker_weights, personalizer=self._ctx)
                else:
                    ranker = self._base_ranker
            except Exception:
                ranker = self._base_ranker

            # call ranker; it expects lists for markov/embeddings and fuzzy as list[(w,dist)]
            ranked = ranker.rank(
                markov=list(markov_map.items()),
                embeddings=list(embed_map.items()),
                fuzzy=list(fuzzy_pairs),
                base_freq=freq_scores,
                recency_map={},  # not tracking recency timestamps here (could be added)
                topn=topn * 2
            )

            # plugin pipeline: allow plugins to add or re-score candidates
            if self.registry:
                try:
                    bundle = {"context": context, "user": getattr(self._ctx, "user", None), "last": last}
                    # registry may return list[(w,score)] or list[(w,score,src)]
                    ranked = self.registry.run_suggest_pipeline(last, ranked, bundle)
                except Exception as e:
                    Log.write(f"[Hybrid] plugin suggest pipeline error: {e}")

            # allow personal bias (CtxPersonal) if ranker didn't already call it
            try:
                ranked = self._ctx.bias_words(ranked)
            except Exception:
                # ignore if ctx has different expectations
                pass

            # infer sources for feedback mapping: prefer markov > embed > personal > fuzzy > plugin
            self._last_sources.clear()
            final_list: List[Candidate] = []
            seen = set()
            for w, sc in ranked:
                if w in seen:
                    continue
                seen.add(w)
                if w in markov_map:
                    src = "markov"
                elif w in embed_map:
                    src = "semantic"
                elif w in personal_scores and personal_scores.get(w, 0) > 0:
                    src = "personal"
                elif any(w == fw for fw, _ in fuzzy_pairs):
                    src = "fuzzy"
                else:
                    src = "plugin"
                self._last_sources[w] = src
                final_list.append((w, float(sc)))

            # trim and format
            out = [(w, round(float(sc), 4)) for w, sc in final_list][:topn]
            self._last_suggestions = out

            Log.metric("hybrid.suggest_latency", round(time.time() - start, 4), "s")
            return out

        except Exception as e:
            Log.write(f"[HybridPredictor] suggest error: {e}")
            return []

    # feedback/reinforcement -----------------------------------------
    def accept(self, word: str, context: Optional[str] = None, source: Optional[str] = None) -> None:
        if not word:
            return
        w = word.lower().strip()
        self.accepted[w] += 1
        # cap to avoid runaway numbers
        if self.accepted[w] > 100000:
            self.accepted[w] = 100000

        resolved_source = source or self._last_sources.get(w, "unknown")

        # record into feedback tracker (best-effort on API shape)
        try:
            if hasattr(self._feedback, "record_accept"):
                self._feedback.record_accept(context or "", w, resolved_source)
            elif hasattr(self._feedback, "record"):
                self._feedback.record(context or "", w, True, resolved_source)
            else:
                # fallback generic
                try:
                    self._feedback.push("accepted", {"word": w, "src": resolved_source, "context": context or ""})
                except Exception:
                    pass
        except Exception as e:
            Log.write(f"[Hybrid] feedback write failed: {e}")

        # reward learner
        try:
            # normalize source keys to learner namespace
            src_key = "semantic" if resolved_source in ("semantic", "embed") else ("markov" if resolved_source == "markov" else "personal")
            if hasattr(self._learner, "reward"):
                self._learner.reward(src_key)
        except Exception:
            pass

        # plugin hook
        if self.registry:
            try:
                self.registry.call_accept(w, {"user": getattr(self._ctx, "user", None), "source": resolved_source})
            except Exception as e:
                Log.write(f"[Hybrid] plugin accept hook error: {e}")

    def reject(self, word: str, context: Optional[str] = None, source: Optional[str] = None) -> None:
        if not word:
            return
        w = word.lower().strip()
        resolved_source = source or self._last_sources.get(w, "unknown")

        # feedback record
        try:
            if hasattr(self._feedback, "record_reject"):
                self._feedback.record_reject(context or "", w, resolved_source)
            elif hasattr(self._feedback, "record"):
                self._feedback.record(context or "", w, False, resolved_source)
            else:
                try:
                    self._feedback.push("rejected", {"word": w, "src": resolved_source, "context": context or ""})
                except Exception:
                    pass
        except Exception:
            pass

        # penalize learner
        try:
            if hasattr(self._learner, "penalize"):
                self._learner.penalize(resolved_source)
        except Exception:
            pass

        # plugin hook (best-effort)
        if self.registry:
            try:
                if hasattr(self.registry, "call_reject"):
                    self.registry.call_reject(w, {"user": getattr(self._ctx, "user", None), "source": resolved_source})
            except Exception as e:
                Log.write(f"[Hybrid] plugin reject hook error: {e}")

    # Persistence --------------------------------------------------
    def save_state(self, path: str) -> None:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump({
                    "markov": self._markov,
                    "ctx": self._ctx,
                    "bk": self._bk,
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
            self._markov = data.get("markov", self._markov)
            self._ctx = data.get("ctx", self._ctx)
            self._bk = data.get("bk", self._bk)
            self.freq_stats = data.get("freq", self.freq_stats)
            self.accepted = data.get("accepted", self.accepted)
            self._trained = True
            Log.write(f"[Hybrid] state loaded from {path}")
        except Exception as e:
            Log.write(f"[Hybrid] load_state failed: {e}")

    # Explain/debug helpers ----------------------------------------------------
    def explain(self, fragment: str, topn: int = 6) -> Dict[str, Any]:
        toks = [t.lower() for t in fragment.strip().split() if t]
        if not toks:
            return {}
        last = toks[-1]
        try:
            mk = dict(self._markov.top_next(last, topn=topn))
        except Exception:
            mk = {}
        try:
            emb = getattr(self._semantic, "similar", lambda *_: [])(last, topn=topn)
            emb = dict(emb)
        except Exception:
            emb = {}
        per = {w: self._ctx.freq.get(w, 0) for w in set(list(mk) + list(emb))}
        freq = {w: self.freq_stats.get(w, 0) for w in set(list(mk) + list(emb) + list(per))}
        # ask base ranker for fused values (best-effort)
        try:
            fused = self._base_ranker.rank(markov=list(mk.items()),
                                           embeddings=list(emb.items()),
                                           fuzzy=[],
                                           base_freq=freq,
                                           recency_map={},
                                           topn=topn)
        except Exception:
            fused = []
        return {"markov": mk, "embed": emb, "personal": per, "freq": freq, "fused": fused}

    def get_weights(self) -> Dict[str, float]:
        try:
            return self._learner.get_weights()
        except Exception:
            # fallback values
            return {"semantic": 0.45, "markov": 0.35, "personal": 0.15, "plugin": 0.05}
