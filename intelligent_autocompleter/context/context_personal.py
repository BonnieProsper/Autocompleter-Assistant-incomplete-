# context_personal.py
# Per-user personalization engine for adaptive autocomplete behavior.
# Learns from user text to bias future suggestions using:
#  - Word frequency
#  - Recency-weighted preferences
#  - POS distribution (if spaCy available)
#  - Semantic similarity (if spaCy vectors available)
# Integrates with Trie + ContextPredictor for a context-aware language assistant.
# ---------------------------------------------------------------------

from __future__ import annotations
import json
import os
import math
import time
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque

from logger_utils import Log

# Lazy spaCy loader
NLP = None


def _ensure_spacy():
    global NLP
    if NLP is not None:
        return NLP

    try:
        import spacy

        NLP = spacy.load("en_core_web_sm", disable=["ner"])
        Log.write("[CtxPersonal] spaCy loaded successfully.")
    except Exception:
        NLP = None
        Log.write("[CtxPersonal] spaCy unavailable; POS + semantic disabled.")
    return NLP


# Storage paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "userdata")
os.makedirs(DATA_DIR, exist_ok=True)


class CtxPersonal:
    """
    User-specific context engine.

    Tracks individualized statistics:
        - Word frequency with log-scaled weighting
        - Recency (sliding window)
        - POS usage patterns (optional)
        - Semantic similarity (optional)
        - Time-based decay to prevent stale influence

    Has scoring layer that applies personalized bias to
    autocomplete suggestions.
    """

    VERSION = 1
    TOKEN_RE = re.compile(r"[a-zA-Z']+")  # allows contractions

    def __init__(self, user: str = "default"):
        self.user = user
        self.path = os.path.join(DATA_DIR, f"{user}.json")

        self.freq: Dict[str, float] = defaultdict(float)
        self.recent: deque[str] = deque(maxlen=50)
        self.pos_hist: Dict[str, float] = defaultdict(float)
        self.last_decay: float = time.time()

        # Cache for semantic embeddings
        self._vec_cache: Dict[str, Optional[List[float]]] = {}

        self._load()

    def learn(self, text: str) -> None:
        """Update user preferences from raw text input."""
        if not isinstance(text, str) or not text.strip():
            return

        tokens = [t.lower() for t in self.TOKEN_RE.findall(text)]
        if not tokens:
            return

        # Decay once an hour
        now = time.time()
        if now - self.last_decay > 3600:
            self._decay()
            self.last_decay = now

        # Frequency & recency
        for w in tokens:
            self.freq[w] += 1.0
            self.recent.append(w)

        # POS learning (lazy spaCy)
        nlp = _ensure_spacy()
        if nlp:
            try:
                doc = nlp(text)
                for tok in doc:
                    self.pos_hist[tok.pos_] += 1.0
            except Exception:
                pass

        Log.write(f"[CtxPersonal] Learned {len(tokens)} tokens for user '{self.user}'.")

    def _decay(self, factor: float = 0.97) -> None:
        """Apply exponential decay to maintain adaptiveness."""
        for table in (self.freq, self.pos_hist):
            for k in list(table.keys()):
                table[k] *= factor
                if table[k] < 0.01:
                    del table[k]

        Log.write(f"[CtxPersonal] Applied decay factor={factor:.2f}")

    def bias_words(
        self, suggestions: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Re-rank suggestions using personal preference statistics.

        Each input item: (word, base_score)
        Returns: sorted list of (word, adjusted_score)
        """
        if not suggestions:
            return suggestions

        total_pos = sum(self.pos_hist.values()) or 1.0
        nlp = NLP or _ensure_spacy()

        adjusted = []

        for word, score in suggestions:
            wl = word.lower()

            # Frequency boost (logarithmic)
            f = self.freq.get(wl, 0.0)
            freq_boost = 1.0 + math.log1p(f) * 0.12

            # Recency boost (decays by position)
            recency_boost = 1.0
            if wl in self.recent:
                idx = list(self.recent).index(wl)
                recency_boost = 1.0 + (1 - idx / len(self.recent)) * 0.25

            # POS affinity
            pos_boost = 1.0
            if nlp:
                try:
                    tok = nlp(wl)[0]
                    p = self.pos_hist.get(tok.pos_, 0.0)
                    pos_boost = 1.0 + (p / total_pos) * 0.1
                except Exception:
                    pass

            # Semantic similarity
            sem_boost = 1.0
            if nlp:
                sem_boost = self._semantic_boost(wl)

            total = score * freq_boost * recency_boost * pos_boost * sem_boost
            adjusted.append((word, total))

        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted

    def _semantic_boost(self, word: str) -> float:
        """Boost score if word is semantically related to recent words."""
        nlp = NLP or _ensure_spacy()
        if not nlp or not self.recent:
            return 1.0

        try:
            # Cache embedding
            if word not in self._vec_cache:
                doc = nlp(word)
                self._vec_cache[word] = doc.vector if doc.has_vector else None

            w_vec = self._vec_cache[word]
            if w_vec is None:
                return 1.0

            recent_words = list(self.recent)[-5:]
            sims = 0.0
            count = 0

            for rw in recent_words:
                if rw not in self._vec_cache:
                    doc = nlp(rw)
                    self._vec_cache[rw] = doc.vector if doc.has_vector else None

                r_vec = self._vec_cache[rw]
                if r_vec is None:
                    continue

                # spaCy explicitly optimizes this internally
                sims += doc.similarity(nlp(rw))
                count += 1

            return 1.0 + (sims / max(count, 1)) * 0.05

        except Exception:
            return 1.0

    def save(self) -> None:
        """Save user context."""
        data = {
            "version": self.VERSION,
            "freq": dict(self.freq),
            "recent": list(self.recent),
            "pos_hist": dict(self.pos_hist),
            "last_decay": self.last_decay,
        }
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            Log.write(f"[ERROR] Saving context for user '{self.user}': {e}")

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Version guard
            if data.get("version") != self.VERSION:
                Log.write("[CtxPersonal] Version mismatch; starting fresh.")
                return

            self.freq.update(data.get("freq", {}))
            self.pos_hist.update(data.get("pos_hist", {}))
            self.recent.extend(data.get("recent", []))
            self.last_decay = data.get("last_decay", time.time())

            Log.write(f"[CtxPersonal] Loaded context ({len(self.freq)} known words).")

        except Exception as e:
            Log.write(f"[ERROR] Loading context for user '{self.user}': {e}")

    def reset(self) -> None:
        """Completely clear user-specific data."""
        self.freq.clear()
        self.recent.clear()
        self.pos_hist.clear()
        self._vec_cache.clear()
        self.save()
        Log.write(f"[CtxPersonal] Reset data for '{self.user}'.")
