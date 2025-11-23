# intelligent_autocompleter/hybrid_predictor.py
"""
HybridPredictor - portfolio-ready, human-style implementation.
Features:
 - Markov (syntactic/contextual)
 - Embeddings (semantic)
 - BK-tree (fuzzy)
 - Personal context (CtxPersonal)
 - FusionRanker for combining signals
 - AdaptiveLearner for online weight tuning
 - FeedbackTracker to persist and compute acceptance ratios
 - Optional PluginRegistry (train/retrain/suggest/accept hooks)
 - Caching, persistence, explainability helpers
"""

import time
import math
import pickle
import os
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any

# model imports (try package import first, fallback to relative)
try:
    from intelligent_autocompleter.core.markov_predictor import MarkovPredictor
    from intelligent_autocompleter.core.embeddings import Embeddings
    from intelligent_autocompleter.core.bktree import BKTree
    from intelligent_autocompleter.core.semantic_engine import SemanticEngine
    from intelligent_autocompleter.core.fusion_ranker import FusionRanker
    from intelligent_autocompleter.core.feedback_tracker import FeedbackTracker
    from intelligent_autocompleter.core.adaptive_learner import AdaptiveLearner
    from intelligent_autocompleter.context_personal import CtxPersonal
    from intelligent_autocompleter.utils.logger_utils import Log
except Exception:
    # local-run fallback for tests/quick run
    from core.markov_predictor import MarkovPredictor
    from core.embeddings import Embeddings
    from core.bktree import BKTree
    from core.semantic_engine import SemanticEngine
    from core.fusion_ranker import FusionRanker
    from core.feedback_tracker import FeedbackTracker
    from core.adaptive_learner import AdaptiveLearner
    from context_personal import CtxPersonal
    from utils.logger_utils import Log

# optional best effort plugin registry import
try:
    from intelligent_autocompleter.plugins.registry import PluginRegistry
except Exception:
    try:
        from plugins.registry import PluginRegistry
    except Exception:
        PluginRegistry = None  # plugins are optional

Scores = Dict[str, float]
Candidate = Tuple[str, float]  # (word, score)

class HybridPredictor:
    """
    Hybrid Predictor. Main public methods:
      - train(lines)
      - retrain(sentence)
      - suggest(fragment, topn=6) -> List[(word, score)]
      - accept(word, context=None, source=None)
      - reject(word, context=None, source=None)
      - save_state(path), load_state(path)
      - explain(fragment)  -> dict breakdown
      - get_weights() -> exposes adaptive learner weights
    """
    def __init__(self,
                 user: str = "default",
                 context_window: int = 2,
                 registry: Optional[Any] = None,
                 feedback_verbose: bool = False):
        """
        Initializes the hybrid predictor with default user and context window size.
        Args:
        - user: Identifies the user for personalization (defaults to "default").
        - context_window: Size of the context window to consider for predictions (defaults to 2).
        """
        # models/engines
        self.mk = MarkovPredictor()
        self.emb = Embeddings()
        self.bk = BKTree()
        self.ctx = CtxPersonal(user)
        self.sem = SemanticEngine()

        # rankers/adaptors
        self.base_ranker = FusionRanker(preset="balanced")
        self.learner = AdaptiveLearner()
        # feedback tracker wired to adaptive learner (best-effort)
        self.feedback = FeedbackTracker(learner=self.learner, verbose=feedback_verbose)

        # optional plugin registry 
        self.registry = registry

        # misc state
        self.context_window = max(1, int(context_window))
        self.accepted = Counter()        # local reinforcement counts
        self.freq_stats = Counter()      # word frequency seen in training
        self._trained = False

        # bk cache for fuzzy lookups: (q,maxd) -> list[(word,dist)]
        self._bk_cache: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
        self._bk_cache_time: Dict[Tuple[str, int], float] = {}
        self._bk_cache_ttl = 60.0

        # mapping from last suggestion to source (best-effort inference)
        self.last_sources: Dict[str, str] = {}
        self.last_ranked: List[Candidate] = []

    # Training/incremental -----------------------------------------------------
    def train(self, lines: List[str]) -> None:
        """
        Train the model using a provided corpus of text (demo_copus.txt).
        Args:
        - lines: List of strings to train the model on.
        """
        if not lines:
            return
        with Log.time_block("Hybrid.train"):
            for ln in lines:
                self._train_line(ln)
            # plugin bulk hook
            if self.registry:
                try:
                    self.registry.call_train(lines)
                except Exception as e:
                    Log.write(f"[plugins] train hook error: {e}")
            # persist ctx
            try:
                self.ctx.save()
            except Exception:
                pass
            self._trained = True
            Log.write("[Hybrid] train finished")

    def retrain(self, sentence: str) -> None:
        """
        Incrementally retrain the model with a new sentence during active use.
        Args:
        - sentence: The new sentence to add to the model's knowledge.
        """
        # incremental learning, learns when user confirms or writes a sentence
        self._train_line(sentence)
        if self.registry:
            try:
                self.registry.call_retrain(sentence)
            except Exception as e:
                Log.write(f"[plugins] retrain hook error: {e}")
        try:
            self.ctx.save()
        except Exception:
            pass
        self._trained = True

    def _train_line(self, ln: str) -> None:
        """
        Train model on a single line of text.
        Args:
        - ln: The line of text to process and train with.
        """
        # keep tokenization simple
        self.mk.train_sentence(ln)
        self.ctx.learn(ln)
        for tok in self._tokens(ln):
            self.freq_stats[tok] += 1
            try:
                self.bk.insert(tok)
            except Exception:
                # BK tree can be picky about input so ignore bad tokens
                pass

    # Helpers ---------------------------------------------------------------
    @staticmethod
    def _tokens(s: str) -> List[str]:
        # small tokenizer for training usage
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

    def _bk_query_cached(self, q: str, maxd: int) -> List[Tuple[str, int]]:
        key = (q, maxd)
        now = time.time()
        if key in self._bk_cache and now - self._bk_cache_time.get(key, 0.0) < self._bk_cache_ttl:
            return self._bk_cache[key]
        try:
            res = self.bk.query(q, max_dist=maxd)
        except Exception:
            res = []
        self._bk_cache[key] = res
        self._bk_cache_time[key] = now
        return res

    # Main suggestion API ----------------------------------------------------------------------
    def suggest(self, text_fragment: str, topn: int = 6, fuzzy: bool = True) -> List[Candidate]:
        """
        Produce ranked suggestions for the last token in text_fragment.
        Returns list[(word, score)] (keeps API stable for older callers).
        Internally also fills self.last_sources and self.last_ranked for feedback mapping.
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

            # Markov top next (word, count)
            mk_list = self.mk.top_next(last, topn=topn * 3)
            markov_scores: Scores = {w: float(c) for w, c in mk_list}

            # Embeddings/semantic similar words (word,score)
            emb_list = self.emb.similar(last, topn=topn * 3) if hasattr(self.emb, "similar") else []
            embed_scores: Scores = {w: float(s) for w, s in emb_list}

            # Personal scores (freq in personal context + recent)
            personal_candidates = set(markov_scores) | set(embed_scores) | set(self.ctx.recent)
            personal_scores: Scores = {w: float(self.ctx.freq.get(w, 0.0)) for w in personal_candidates}

            # global frequency
            freq_scores: Scores = {w: float(self.freq_stats.get(w, 0)) for w in set(markov_scores) | set(embed_scores) | set(personal_scores)}

            # fuzzy BK fallback, add small boost into embed_scores
            fuzzy_results: List[Tuple[str, int]] = []
            if fuzzy:
                maxd = self._adaptive_maxd(last)
                if maxd > 0:
                    fuzzy_results = self._bk_query_cached(last, maxd)
                    for w, dist in fuzzy_results:
                        bonus = max(0.0, (maxd + 1) - dist)
                        embed_scores[w] = max(embed_scores.get(w, 0.0), bonus * 0.5)

            # reinforcement from accepted dictionary
            for w, c in self.accepted.items():
                personal_scores[w] = personal_scores.get(w, 0.0) + math.log1p(c) * 0.35

            # Fuse signals, preferentially use AdaptiveLearner ->FusionRanker
            ranked: List[Tuple[str, float]]
            try:
                weights = self.learner.get_weights() if self.learner else None
                if weights:
                    # FusionRanker might use different param shapes
                    # try to use its high-level rank API, otherwise fallback below.
                    tmp_weights = {"markov": weights.get("markov", 0.0),
                                   "embed": weights.get("semantic", 0.0),
                                   "personal": weights.get("personal", 0.0),
                                   "freq": 0.05}
                    tmp_ranker = FusionRanker()
                    # If FusionRanker exposes a 'fuse' or 'rank' that accepts dicts, attempt
                    if hasattr(tmp_ranker, "rank"):
                        ranked = tmp_ranker.rank(markov=markov_scores,
                                                 embed=embed_scores,
                                                 personal=personal_scores,
                                                 freq=freq_scores,
                                                 topn=topn * 2)
                    else:
                        # fallback generic combine
                        raise RuntimeError("FusionRanker.rank not available as expected")
                else:
                    # base ranker fallback, try to call rank similarly
                    if hasattr(self.base_ranker, "rank"):
                        ranked = self.base_ranker.rank(markov=markov_scores,
                                                       embed=embed_scores,
                                                       personal=personal_scores,
                                                       freq=freq_scores,
                                                       topn=topn * 2)
                    else:
                        raise RuntimeError("base_ranker.rank not available")
            except Exception:
                # final fallback, linear combine with alpha weighting
                merged: Scores = {}
                alpha = 0.45
                for w in set(markov_scores) | set(embed_scores):
                    merged[w] = markov_scores.get(w, 0.0) * (1 - alpha) + embed_scores.get(w, 0.0) * alpha
                ranked = sorted(merged.items(), key=lambda kv: -kv[1])[:topn * 2]

            # plugin pipeline, let plugins rerank or add candidates
            if self.registry:
                try:
                    # registry expects list[(w,score)] and returns list[(w,score)] or (w,score,src)
                    bundle = {"context": context, "user": getattr(self.ctx, "user", None), "last": last}
                    ranked = self.registry.run_suggest_pipeline(last, ranked, bundle)  # allow plugins to mutate
                except Exception as e:
                    Log.write(f"[plugins] suggest pipeline error: {e}")

            # bias by personal heuristics
            try:
                ranked = self.ctx.bias_words(ranked)
            except Exception:
                # if ctx.bias_words has different expectations, ignore
                pass

            # Build last_sources mapping with best-effort source inference
            self.last_sources.clear()
            final_list: List[Candidate] = []
            seen = set()
            for w, sc in ranked:
                if w in seen:
                    continue
                seen.add(w)
                # infer source preference
                if w in markov_scores:
                    src = "markov"
                elif w in embed_scores:
                    src = "semantic"
                elif w in personal_scores:
                    src = "personal"
                elif any(w == fw for fw, _ in fuzzy_results):
                    src = "fuzzy"
                else:
                    src = "plugin"
                self.last_sources[w] = src
                final_list.append((w, float(sc)))

            # trim to requested topn
            out = [(w, round(float(sc), 3)) for w, sc in final_list][:topn]
            self.last_ranked = out

            Log.metric("suggest_latency", round(time.time() - start, 3), "s")
            return out

        except Exception as e:
            Log.write(f"[HybridPredictor] suggest error: {e}")
            return []

    # Reinforcement/feedback ----------------------------------------------------------------
    def accept(self, word: str, context: Optional[str] = None, source: Optional[str] = None) -> None:
        """
        Called when a user accepts a suggestion.
        If source isn't provided, tries to use last suggestions mapping.
        Records feedback and rewards learner.
        """
        if not word:
            return
        w = word.lower().strip()
        self.accepted[w] += 1
        # cap the reinforcement counter
        if self.accepted[w] > 10000:
            self.accepted[w] = 10000

        # resolve source (prefer provided source)
        resolved_source = source or self.last_sources.get(w, "unknown")

        # record in feedback tracker (use public API)
        try:
            # FeedbackTracker may expose different method names across versions 
            if hasattr(self.feedback, "record_accept"):
                self.feedback.record_accept(context or "", w, resolved_source)
            elif hasattr(self.feedback, "record"):
                self.feedback.record(context or "", w, True, resolved_source)
            else:
                # generic push
                try:
                    self.feedback.push("accepted", {"word": w, "src": resolved_source, "context": context or ""})
                except Exception:
                    pass
        except Exception as e:
            Log.write(f"[HybridPredictor] feedback record failed: {e}")

        # reward adaptive learner for that source
        try:
            # normalize some aliases
            src_key = "semantic" if resolved_source in ("semantic", "embed") else ("markov" if resolved_source == "markov" else "personal")
            if hasattr(self.learner, "reward"):
                self.learner.reward(src_key)
        except Exception:
            pass

        # plugin hook
        if self.registry:
            try:
                self.registry.call_accept(w, {"user": getattr(self.ctx, "user", None), "source": resolved_source})
            except Exception as e:
                Log.write(f"[plugins] accept hook error: {e}")

    def reject(self, word: str, context: Optional[str] = None, source: Optional[str] = None) -> None:
        """Record a rejection by penalizing the source and notifying plugins."""
        if not word:
            return
        w = word.lower().strip()
        resolved_source = source or self.last_sources.get(w, "unknown")

        # record reject
        try:
            if hasattr(self.feedback, "record_reject"):
                self.feedback.record_reject(context or "", w, resolved_source)
            elif hasattr(self.feedback, "record"):
                self.feedback.record(context or "", w, False, resolved_source)
            else:
                try:
                    self.feedback.push("rejected", {"word": w, "src": resolved_source, "context": context or ""})
                except Exception:
                    pass
        except Exception:
            pass

        # penalize adaptive learner
        try:
            if hasattr(self.learner, "penalize"):
                self.learner.penalize(resolved_source)
        except Exception:
            pass

        # plugin hook, optionally call rechook if it exists
        if self.registry:
            try:
                # registry might not implement call_reject, best-effort
                if hasattr(self.registry, "call_reject"):
                    self.registry.call_reject(w, {"user": getattr(self.ctx, "user", None), "source": resolved_source})
            except Exception as e:
                Log.write(f"[plugins] reject hook error: {e}")

    # Persistence ---------------------------------------------------------------
    def save_state(self, path: str) -> None:
        """Save model and its state."""
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump({
                    "markov": self.mk,
                    "ctx": self.ctx,
                    "bk": self.bk,
                    "freq": self.freq_stats,
                    "accepted": self.accepted,
                }, fh)
            Log.write(f"[HybridPredictor] state saved -> {path}")
        except Exception as e:
            Log.write(f"[HybridPredictor] save_state failed: {e}")

    def load_state(self, path: str) -> None:
        """Load previously saved model from a file."""
        if not os.path.exists(path):
            Log.write(f"[HybridPredictor] load_state: path not found {path}")
            return
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            self.mk = data.get("markov", self.mk)
            self.ctx = data.get("ctx", self.ctx)
            self.bk = data.get("bk", self.bk)
            self.freq_stats = data.get("freq", self.freq_stats)
            self.accepted = data.get("accepted", self.accepted)
            self._trained = True
            Log.write(f"[HybridPredictor] state loaded from {path}")
        except Exception as e:
            Log.write(f"[HybridPredictor] load_state failed: {e}")

    # Explainability & debug ----------------------------------------------------------
    def explain(self, fragment: str, topn: int = 6) -> Dict[str, Any]:
        """
        Give a debugging breakdown of contributions from each component.
        For /explain CLI command or debugging.
        """
        toks = [t.lower() for t in fragment.strip().split() if t]
        if not toks:
            return {}
        last = toks[-1]
        try:
            mk = dict(self.mk.top_next(last, topn=topn))
        except Exception:
            mk = {}
        try:
            emb = dict(self.emb.similar(last, topn=topn)) if hasattr(self.emb, "similar") else {}
        except Exception:
            emb = {}
        per = {w: self.ctx.freq.get(w, 0) for w in set(list(mk) + list(emb))}
        freq = {w: self.freq_stats.get(w, 0) for w in set(list(mk) + list(emb) + list(per))}
        try:
            fused = self.base_ranker.fuse(mk, emb, per, freq)
        except Exception:
            fused = {}
        return {"markov": mk, "embed": emb, "personal": per, "freq": freq, "fused": fused}

    def get_weights(self) -> Dict[str, float]:
        """Expose current learner weights (semantic/markov/personal/plugin)."""
        try:
            return self.learner.get_weights()
        except Exception:
            # fallback to base preset
            return {"semantic": 0.45, "markov": 0.35, "personal": 0.15, "plugin": 0.05}
