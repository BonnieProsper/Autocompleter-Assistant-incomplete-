# hybrid_predictor.py 
# Combines:
# - Markov (syntactic/contextual predictions)
# - Embeddings (semantic similarity)
# - BK-tree (fuzz matchingy)
# - User personalization (bias + recent context, CtxPersonal)
# - FusionRanker (adaptive weighting/presets)
# Exposes methods for training, retraining, suggestion generation, acceptance/rejection feedback, and state persistence.

import time
import math
import pickle
from collections import Counter, deque
from typing import List, Dict, Tuple, Optional

from intelligent_autocompleter.core.markov_predictor import MarkovPredictor
from intelligent_autocompleter.core.embeddings import Embeddings
from intelligent_autocompleter.core.bktree import BKTree
from intelligent_autocompleter.core.semantic_engine import SemanticEngine
from intelligent_autocompleter.core.feedback_tracker import FeedbackTracker
from intelligent_autocompleter.context_personal import CtxPersonal
from intelligent_autocompleter.core.fusion_ranker import FusionRanker
from intelligent_autocompleter.utils.logger_utils import Log

Scores = Dict[str, float]

class HybridPredictor:
    def __init__(self, user: str = "default", context_window: int = 3):
        """
        Initializes the hybrid predictor with default user and context window size.
        Args:
        - user: Identifies the user for personalization (defaults to "default").
        - context_window: Size of the context window to consider for predictions (defaults to 3).
        """
        self.user = user
        self.context_window = max(1, context_window)

        # initialise all prediction models/parts
        self.markov = MarkovPredictor()
        self.emb = Embeddings()
        self.bk = BKTree()
        self.ctx = CtxPersonal(user) # personal context for bias and user preference
        self.sem = SemanticEngine()
        self.rank = FusionRanker(preset="balanced") # initialise to balanced weight for fusion
        self.feedback = FeedbackTracker()

        # initialise state tracking objects
        self.accepted = Counter() # tracks accepted suggestions
        self.freq_stats = Counter() # tracks word frequencies
        self.session_context = deque(maxlen=20) # stores recent context for session based predictions
        self._trained = False # flag to show whether model is trained

    # Training and incremental updates
    # ------------------------------------------------------------------
    def train(self, corpus: List[str]):
        """
        Train the model using a provided corpus of text (demo_copus.txt).
        Args:
        - corpus: List of strings to train the model on.
        """
        if not corpus:
            Log.write("[HybridPredictor] empty corpus — nothing to train.")
            return

        with Log.time_block("HybridPredictor.train"): # log training time
            for line in corpus:
                self._train_line(line) # process each line in corpus
        self.ctx.save() # save learned context (user specific knowledge)
        self._trained = True
        Log.write("[HybridPredictor] training complete.")

    def retrain(self, sentence: str):
        """
        Incrementally retrain the model with a new sentence during active use.
        Args:
        - sentence: The new sentence to add to the model's knowledge.
        """
        self._train_line(sentence) # train on new sentence incrimentally
        self.ctx.save() # update and save context

    def _train_line(self, text: str):
        """
        Train the model on a single line of text by processing tokens and updating various components.
        Args:
        - text: The line of text to process.
        """
        tokens = [t.lower() for t in text.strip().split() if t.isalpha()] # tokenise, make lowercase
        self.markov.train_sentence(text) # train markov model with sentence
        self.ctx.learn(text) # learn user context from sentence
        for t in tokens:
            self.freq_stats[t] += 1 # update word frequency
            try:
                self.bk.insert(t) # insert token into BK tree for fuzzy matching
            except Exception:
                pass

    # Suggestion engine
    # ------------------------------------------------------------------
    def suggest(self, fragment: str, topn: int = 6, fuzzy: bool = True) -> List[Tuple[str, float]]:
        """
        Generate a list of word suggestions based on the provided text fragment.
        Args:
        - fragment: Partial text or word fragment to generate suggestions for.
        - topn: Number of top suggestions to return (default = 6).
        - fuzzy: Whether to include fuzzy matches from the BK-tree (default = True).
        Returns:
        - A list of tuples (word, score) where 'word' is a suggested word and 'score' is its prediction score.
        """
        if not fragment:
            return []

        start = time.time() # track time taken for suggestions
        toks = [t.lower() for t in fragment.strip().split()] # tokenise
        if not toks:
            return []

        context = toks[-self.context_window :] # use last context_window tokens
        last = context[-1] # last word in context
        self.session_context.extend(context) # update session context

        try:
            # Core signal sources: combine different model outputs
            markov_scores = {w: float(s) for w, s in self.markov.top_next(last, topn=topn * 2)}
            embed_scores = {w: float(s) for w, s in self.emb.similar(last, topn=topn * 2)} if hasattr(self.emb, "similar") else {}
            personal_scores = {w: float(self.ctx.freq.get(w, 0)) for w in set(markov_scores) | set(embed_scores)}
            freq_scores = {w: float(self.freq_stats.get(w, 0)) for w in set(markov_scores) | set(embed_scores) | set(personal_scores)}

            # Optional fuzzy match using BK tree
            if fuzzy:
                maxd = self._adaptive_maxd(last)
                for w, dist in self.bk.query(last, max_dist=maxd):
                    bonus = max(0.0, (maxd + 1 - dist)) * 0.4
                    embed_scores[w] = max(embed_scores.get(w, 0.0), bonus)

            # Reinforcement from accepted words
            for w, c in self.accepted.items():
                personal_scores[w] = personal_scores.get(w, 0.0) + math.log1p(c) * 0.5

            # Fuse all signals together into final ranking
            ranked = self.rank.rank(
                markov_scores, embed_scores, personal_scores, freq_scores, topn=topn
            )

            # Adaptive bias based on feedback for personalised ranking
            ranked = self._apply_feedback_bias(ranked, last)

            # Log latency for performance tracking
            Log.metric("suggest_latency", round(time.time() - start, 3), "s")
            return ranked

        except Exception as e:
            Log.write(f"[HybridPredictor] suggest error: {e}")
            return []

    # Adaptive feedback and learning
    # ------------------------------------------------------------------
    def accept(self, word: str):
        """
        Called when a user accepts or confirms a suggestion.
        Args:
        - word: The accepted suggestion.
        """
        w = word.lower().strip()
        self.accepted[w] += 1 # increase acceptance count for word by 1
        self.feedback.record_accept(w) # record acceptance in feedback tracker

        # Adjust ranker dynamically based on ongoing feedback trend
        self.rank.autotune(self.feedback.trends())

    def reject(self, word: str):
        """
        Optional rejection tracking for a word.
        Args:
        - word: The rejected suggestion.
        """
        w = word.lower().strip()
        self.feedback.record_reject(w)

    def _apply_feedback_bias(self, ranked: List[Tuple[str, float]], last_word: str) -> List[Tuple[str, float]]:
        """
        Adjust scores using feedback-based multipliers for personalized ranking.
        Args:
        - ranked: List of (word, score) tuples to adjust based on feedback.
        - last_word: The last word in the user's input, used for bias adjustments.
        Returns:
        - A list of (word, adjusted_score) tuples.
        """
        adjusted = []
        for w, score in ranked:
            ratio = self.feedback.acceptance_ratio(w) # calculate feedback acceptance rate for word
            multiplier = 0.9 + (ratio * 0.25)  # adjust mulitplier based on acceptance ratio 
            adjusted.append((w, round(score * multiplier, 4))) # apply multiplier to score
        return adjusted

    # Helpers and diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _adaptive_maxd(word: str) -> int:
        """
        Determine the maximum allowable distance for fuzzy matching based on session context.
        Args:
        - last_word: The last word in the user's input, used to calculate distance.
        Returns:
        - A maximum distance value for BK-tree querying.
        """
        L = len(word)
        return 0 if L <= 3 else 1 if L <= 6 else 2 if L <= 12 else 3

    def explain(self, fragment: str, topn: int = 5) -> Dict[str, Dict[str, float]]:
        """Return a breakdown of component contributions for transparency."""
        toks = [t.lower() for t in fragment.strip().split()]
        if not toks:
            return {}

        last = toks[-1]
        mk = dict(self.markov.top_next(last, topn=topn))
        emb = dict(self.emb.similar(last, topn=topn)) if hasattr(self.emb, "similar") else {}
        per = {w: self.ctx.freq.get(w, 0) for w in set(mk) | set(emb)}
        freq = {w: self.freq_stats.get(w, 0) for w in set(mk) | set(emb) | set(per)}
        fused = self.rank.fuse(mk, emb, per, freq)

        return {
            "markov": mk,
            "embed": emb,
            "personal": per,
            "freq": freq,
            "fused": dict(sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:topn]),
        }

    # Persistence
    # ------------------------------------------------------------------
    def save_state(self, path: str):
        """
        Save the model and its state to a file for later use.
        Args:
        - path: The path to save the model state to.
        """
        data = {
            "markov": self.markov,
            "ctx": self.ctx,
            "bk": self.bk,
            "freq": self.freq_stats,
            "accepted": self.accepted,
            "feedback": self.feedback.data,
            "rank_preset": self.rank.preset,
        }
        try:
            with open(path, "wb") as fh:
                pickle.dump(data, fh)
            Log.write(f"[HybridPredictor] state saved to {path}")
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
            self.markov = data.get("markov", self.markov)
            self.ctx = data.get("ctx", self.ctx)
            self.bk = data.get("bk", self.bk)
            self.freq_stats = data.get("freq", self.freq_stats)
            self.accepted = data.get("accepted", self.accepted)
            self.feedback.data = data.get("feedback", self.feedback.data)
            if preset := data.get("rank_preset"):
                self.rank.update_preset(preset)
            self._trained = True
            Log.write(f"[HybridPredictor] state loaded from {path}")
        except Exception as e:
            Log.write(f"[HybridPredictor] load_state failed: {e}")

    # Reporting
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, object]:
        return {
            "user": self.user,
            "vocab_size": len(self.freq_stats),
            "accepted": sum(self.accepted.values()),
            "feedback_acceptance_rate": round(self.feedback.global_acceptance_rate(), 3),
            "mode": self.rank.preset,
            "context_window": self.context_window,
        }
