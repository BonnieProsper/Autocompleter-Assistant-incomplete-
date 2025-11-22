# intelligent_autocompleter/hybrid_predictor.py
# Integrated HybridPredictor, combines:
# - markov (syntax/contextual predictions), embeddings (semantic), bk-tree (fuzzy), personal context
# - FusionRanker (weights can be overridden by AdaptiveLearner)
# - FeedbackTracker and AdaptiveLearner wired in
# - Plugin registry hooks supported to extend behaviour

import time
import math
import pickle
from collections import Counter
from typing import List, Tuple, Dict, Optional

# imports, local or package
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
    # fallback to relative imports if run from package root
    from core.markov_predictor import MarkovPredictor
    from core.embeddings import Embeddings
    from core.bktree import BKTree
    from core.semantic_engine import SemanticEngine
    from core.fusion_ranker import FusionRanker
    from core.feedback_tracker import FeedbackTracker
    from core.adaptive_learner import AdaptiveLearner
    from context_personal import CtxPersonal
    from utils.logger_utils import Log

# optional plugin registry
try:
    from intelligent_autocompleter.plugins.registry import PluginRegistry
except Exception:
    try:
        from plugins.registry import PluginRegistry
    except Exception:
        PluginRegistry = None

Scores = Dict[str, float]
Candidate = Tuple[str, float]

class HybridPredictor:
    """
    Hybrid predictor with feedback + adaptive learner + plugin support.
    """
    def __init__(self,
                 user: str = "default",
                 context_window: int = 2,
                 registry: Optional[object] = None,
                 feedback_verbose: bool = False):
        """
        Initializes the hybrid predictor with default user and context window size.
        Args:
        - user: Identifies the user for personalization (defaults to "default").
        - context_window: Size of the context window to consider for predictions (defaults to 3).
        """
        # models
        self.mk = MarkovPredictor()
        self.emb = Embeddings()
        self.bk = BKTree()
        self.ctx = CtxPersonal(user)
        self.sem = SemanticEngine()

        # fusion
        self.base_ranker = FusionRanker(preset="balanced")

        # feedback + learner
        self.learner = AdaptiveLearner()
        self.feedback = FeedbackTracker(learner=self.learner, verbose=feedback_verbose)

        # plugin registry (optional)
        self.registry = registry

        # bookkeeping
        self.context_window = max(1, int(context_window))
        self.accepted = Counter()
        self.freq_stats = Counter()
        self.trained = False

        # BK cache
        self.bk_cache = {}
        self.bk_time = {}
        self.bk_cache_ttl = 60.0

        # last suggestions mapping (word to source) for accept mapping
        self.last_sources = {}  # word to source string
        self.last_ranked = []   # last ranked list used (for debugging)

    # training ----------------------------------------------------------
    def train(self, lines: List[str]):
        """
        Train the model using a provided corpus of text (demo_copus.txt).
        Args:
        - lines: List of strings to train the model on.
        """
        if not lines:
            return
        with Log.time_block("Hybrid.train"):
            for ln in lines:
                self.train_line(ln)
            # plugin hook
            if self.registry:
                try:
                    self.registry.call_train(lines)
                except Exception as e:
                    Log.write(f"[plugins] train hook error: {e}")
            try:
                self.ctx.save()
            except Exception:
                pass
            self.trained = True

    def retrain(self, sentence: str):
        """
        Incrementally retrain the model with a new sentence during active use.
        Args:
        - sentence: The new sentence to add to the model's knowledge.
        """
        self.train_line(sentence)
        if self.registry:
            try:
                self.registry.call_retrain(sentence)
            except Exception as e:
                Log.write(f"[plugins] retrain hook error: {e}")
        try:
            self.ctx.save()
        except Exception:
            pass
        self.trained = True

    def train_line(self, ln: str):
        """
        Train the model on a single line of text by processing tokens and updating various components.
        Args:
        - ln: The line of text to process.
        """
        self.mk.train_sentence(ln)
        self.ctx.learn(ln)
        for tok in self.get_tokens(ln):
            self.freq_stats[tok] += 1
            try:
                self.bk.insert(tok)
            except Exception:
                pass

    # helpers --------------------------------------------------------------
    @staticmethod
    def get_tokens(s: str) -> List[str]:
        return [t.lower() for t in s.strip().split() if t.isalpha()]

    def adaptive_maxd(self, q: str) -> int:
        L = len(q)
        if L <= 3: 
            return 0
        if L <= 6: 
            return 1
        if L <= 12: 
            return 2
        return 3

    def bk_query_cached(self, q: str, maxd: int):
        key = (q, maxd)
        now = time.time()
        if key in self.bk_cache and now - self.bk_time.get(key, 0.0) < self.bk_cache_ttl:
            return self.bk_cache[key]
        try:
            res = self.bk.query(q, max_dist=maxd)
        except Exception:
            res = []
        self.bk_cache[key] = res
        self.bk_time[key] = now
        return res

    # Main Suggestion API -----------------------------------------------------------------------
    def suggest(self, text_fragment: str, topn: int = 6, fuzzy: bool = True) -> List[Candidate]:
        """
        Generate ranked suggestions based on the text fragment string. 
        Each candidate is annotated internally with a primary source.
        Sources: 'markov', 'semantic', 'personal', 'fuzzy', 'plugin'
        Args:
        - fragment: Partial text or word fragment to generate suggestions for.
        - topn: Number of top suggestions to return (default = 6).
        - fuzzy: Whether to include fuzzy matches from the BK-tree (default = True).
        Returns:
        - A list of tuples (word, score) where 'word' is a suggested word and 'score' is its prediction score.
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

            # gather candidate suggestions from sources
            mk_list = self.mk.top_next(last, topn=topn * 3)
            markov_scores = {w: float(s) for w, s in mk_list}

            emb_list = self.emb.similar(last, topn=topn * 3) if hasattr(self.emb, "similar") else []
            embed_scores = {w: float(s) for w, s in emb_list}

            # personal: include candidates + some recent words
            personal_set = set(list(markov_scores) + list(embed_scores)) | set(self.ctx.recent)
            personal_scores = {w: float(self.ctx.freq.get(w, 0.0)) for w in personal_set}

            # frequency map
            freq_scores = {w: float(self.freq_stats.get(w, 0)) for w in set(list(markov_scores) + list(embed_scores) + list(personal_scores))}

            # fuzzy fallback
            fuzzy_list = []
            if fuzzy:
                maxd = self.adaptive_maxd(last)
                if maxd > 0:
                    fuzzy_res = self.bk_query_cached(last, maxd)
                    for w, dist in fuzzy_res:
                        bonus = max(0.0, (maxd + 1) - dist)
                        # treat fuzzy as small semantic bump
                        embed_scores[w] = max(embed_scores.get(w, 0.0), bonus * 0.4)
                        fuzzy_list.append((w, dist))

            # reinforcement from accepted words
            for w, c in self.accepted.items():
                personal_scores[w] = personal_scores.get(w, 0.0) + math.log1p(c) * 0.3

            # fusion: use adaptive weighting if adaptivelearner is present
            try:
                weights = self.learner.get_weights() if self.learner else None
                if weights:
                    tmp_ranker = FusionRanker(weights={k + "_weight": v for k, v in weights.items()}, preset=None, personalizer=self.ctx)
                    ranked = tmp_ranker.rank(markov=list(markov_scores.items()),
                                             embeddings=list(embed_scores.items()),
                                             fuzzy=fuzzy_list,
                                             base_freq=freq_scores,
                                             recency_map={}, topn=topn)
                else:
                    ranked = self.base_ranker.rank(markov=list(markov_scores.items()),
                                                   embeddings=list(embed_scores.items()),
                                                   fuzzy=fuzzy_list,
                                                   base_freq=freq_scores,
                                                   recency_map={}, topn=topn)
            except Exception:
                # fallback to simple combined
                merged = {}
                for w in set(list(markov_scores) + list(embed_scores)):
                    merged[w] = markov_scores.get(w, 0.0) * 0.6 + embed_scores.get(w, 0.0) * 0.4
                ranked = sorted(merged.items(), key=lambda kv: -kv[1])[:topn]

            # if plugins are available, allow them to inspect/modify results
            try:
                if self.registry:
                    bundle = {"context": context, "user": getattr(self.ctx, "user", None), "last": last}
                    ranked = self.registry.run_suggest_pipeline(last, ranked, bundle)
            except Exception as e:
                Log.write(f"[plugins] suggest pipeline error: {e}")

            # bias by personal heuristics and format
            ranked = self.ctx.bias_words(ranked)

            # store source hints for accept() mapping in best effort format
            self.last_sources.clear()
            for w, _ in ranked:
                # guess source: prefer markov > embed > personal > fuzzy > plugin
                if w in markov_scores:
                    src = "markov"
                elif w in embed_scores:
                    src = "semantic"
                elif w in personal_scores:
                    src = "personal"
                else:
                    src = "plugin"
                self.last_sources[w] = src

            # final trimming
            out = [(w, round(float(sc), 3)) for w, sc in ranked][:topn]
            self.last_ranked = out

            Log.metric("Suggest latency", round(time.time() - start, 3), "s")
            return out

        except Exception as e:
            Log.write(f"[HybridPredictor] suggest error: {e}")
            return []

    # reinforcement hooks --------------------------------------------------------------------------
    def accept(self, suggestion: str, context: Optional[str] = None, source: Optional[str] = None):
        """
        Called when a user accepts a suggestion.
        If source isn't provided, tries to use last suggestions mapping.
        Records feedback and rewards learner.
        """
        if not suggestion:
            return
        s = suggestion.lower().strip()
        # increment local counter
        self.accepted[s] += 1
        if self.accepted[s] > 10000:
            self.accepted[s] = 10000

        # infer source if not provided
        resolved_source = source or self.last_sources.get(s, "unknown")

        # record to feedback tracker
        try:
            ctx = context or ""
            self.feedback.record_accept(ctx, s, resolved_source)
        except Exception as e:
            Log.write(f"[HybridPredictor] feedback record failed: {e}")

        # best effort reward learner for that source
        try:
            self.learner.reward(resolved_source)
        except Exception:
            pass

        # plugin hook
        if self.registry:
            try:
                self.registry.call_accept(s, {"user": getattr(self.ctx, "user", None), "source": resolved_source})
            except Exception as e:
                Log.write(f"[plugins] accept hook error: {e}")

    def reject(self, suggestion: str, context: Optional[str] = None, source: Optional[str] = None):
        """Record a rejection by penalizing the source and notifying plugins."""
        if not suggestion:
            return
        s = suggestion.lower().strip()
        resolved_source = source or self.last_sources.get(s, "unknown")
        try:
            self.feedback.record_reject(context or "", s, resolved_source)
        except Exception:
            pass
        try:
            self.learner.penalize(resolved_source)
        except Exception:
            pass
        if self.registry:
            try:
                self.registry.call_reject(s, {"user": getattr(self.ctx, "user", None), "source": resolved_source})
            except Exception as e:
                Log.write(f"[plugins] reject hook error: {e}")

    # Persistence Helpers ----------------------------------------------------------------
    def save_state(self, path: str):
        """
        Save the model and its state to a file for later use.
        Args:
        - path: The path to save the model state to.
        """
        try:
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

    def load_state(self, path: str):
        """
        Load a previously saved model from a file.
        Args:
        - path: The path to load the model state from.
        Returns:
        - An instance of HybridPredictor loaded from the file.
        """
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            self.mk = data.get("markov", self.mk)
            self.ctx = data.get("ctx", self.ctx)
            self.bk = data.get("bk", self.bk)
            self.freq_stats = data.get("freq", self.freq_stats)
            self.accepted = data.get("accepted", self.accepted)
            Log.write(f"[HybridPredictor] state loaded from {path}")
            self.trained = True
        except Exception as e:
            Log.write(f"[HybridPredictor] load_state failed: {e}")

    # Debugging Helpers --------------------------------------------------------------
    def explain(self, fragment: str, topn: int = 6):
        """Return breakdown of contributions for the fragment (useful for /explain CLI command)."""
        toks = [t.lower() for t in fragment.strip().split() if t]
        if not toks:
            return {}
        last = toks[-1]
        mk = dict(self.mk.top_next(last, topn=topn))
        emb = dict(self.emb.similar(last, topn=topn)) if hasattr(self.emb, "similar") else {}
        per = {w: self.ctx.freq.get(w, 0) for w in set(list(mk) + list(emb))}
        freq = {w: self.freq_stats.get(w, 0) for w in set(list(mk) + list(emb) + list(per))}
        fused = self.base_ranker.fuse(mk, emb, per, freq)
        return {"markov": mk, "embed": emb, "personal": per, "freq": freq, "fused": fused}
