# context_personal.py - context + user pref and vocabulary tracking + recent context

import json
import os
import math
import time
from collections import defaultdict, deque
from logger_utils import Log

try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
except Exception:
    NLP = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "userdata")


class CtxPersonal:
    """
    Context-aware personalization engine.

    Learns from user input to bias future word suggestions.
    - Tracks word frequency and recency.
    - Learns POS (part-of-speech) distributions if spaCy is available.
    - Applies decay to older data for adaptive behavior.
    - Persists user data between sessions.
    """

    def __init__(self, user: str = "default"):
        self.user = user
        os.makedirs(DATA_DIR, exist_ok=True)
        self.path = os.path.join(DATA_DIR, f"{user}.json")

        self.freq = defaultdict(float)     # word frequency
        self.recent = deque(maxlen=50)     # recent context
        self.pos_hist = defaultdict(float) # POS tag frequencies
        self.last_decay = time.time()

        self._load()

    # Learning and Decay --------------------------------------------
    def learn(self, text: str):
        """
        Update learned preferences from a user text input.
        """
        if not text or not isinstance(text, str):
            return

        tokens = [w.lower() for w in text.split() if w.isalpha()]
        if not tokens:
            return

        # Apply occasional decay to simulate adaptive forgetting
        if time.time() - self.last_decay > 3600:  # every hour
            self._decay()
            self.last_decay = time.time()

        # Learn frequencies
        for word in tokens:
            self.freq[word] += 1
            self.recent.append(word)

        # Learn linguistic context
        if NLP:
            doc = NLP(text)
            for tok in doc:
                self.pos_hist[tok.pos_] += 1

        Log.write(f"[CtxPersonal] Learned {len(tokens)} words for user '{self.user}'")

    def _decay(self, factor: float = 0.97):
        """
        Gradually decay old data to keep context adaptive and avoid overfitting.
        """
        for d in (self.freq, self.pos_hist):
            for k in list(d.keys()):
                d[k] *= factor
                if d[k] < 0.01:
                    del d[k]
        Log.write(f"[CtxPersonal] Applied decay factor {factor:.2f}")

    # Biasing and Scoring -------------------------------------------------------
    def bias_words(self, suggestions):
        """
        Adjust suggestion scores based on user frequency, recency, and linguistic patterns.
        Each element of 'suggestions' should be (word, score).
        """
        adjusted = []
        for word, score in suggestions:
            # Frequency boost (log-based)
            freq_boost = 1 + math.log1p(self.freq.get(word, 0)) * 0.15

            # Recency boost
            recent_boost = 1.15 if word in self.recent else 1.0

            # POS pattern alignment (if known)
            if NLP:
                try:
                    pos = NLP(word)[0].pos_
                    pos_weight = 1 + (self.pos_hist.get(pos, 0) / (sum(self.pos_hist.values()) or 1)) * 0.1
                except Exception:
                    pos_weight = 1.0
            else:
                pos_weight = 1.0

            # Semantic boost (optional)
            sim_boost = self._semantic_boost(word) if NLP else 1.0

            total_score = score * freq_boost * recent_boost * pos_weight * sim_boost
            adjusted.append((word, total_score))

        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted

    def _semantic_boost(self, word: str) -> float:
        """
        If spaCy vectors are available, boost words similar to recent ones.
        """
        if not NLP or not self.recent:
            return 1.0
        try:
            wv = NLP(word).vector
            if wv is None or not len(wv):
                return 1.0

            # Average similarity to recent words
            sims, n = 0.0, 0
            for recent_word in list(self.recent)[-5:]:
                rv = NLP(recent_word).vector
                if rv is not None and len(rv):
                    sims += wv.dot(rv) / (math.sqrt((wv**2).sum()) * math.sqrt((rv**2).sum()) + 1e-8)
                    n += 1
            return 1 + (sims / max(n, 1)) * 0.05
        except Exception:
            return 1.0

    # Persistence-----------------------------------------------
    def save(self):
        data = {
            "freq": dict(self.freq),
            "recent": list(self.recent),
            "pos_hist": dict(self.pos_hist),
            "last_decay": self.last_decay,
        }
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            Log.write(f"[ERROR] Saving user context: {e}")

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.freq.update(data.get("freq", {}))
            self.pos_hist.update(data.get("pos_hist", {}))
            self.recent.extend(data.get("recent", []))
            self.last_decay = data.get("last_decay", time.time())
            Log.write(f"[CtxPersonal] Loaded context for user '{self.user}' ({len(self.freq)} words)")
        except Exception as e:
            Log.write(f"[ERROR] Loading user context: {e}")

    # Utilities -------------------------------------------------------
    def reset(self):
        """Clear all user-learned data."""
        self.freq.clear()
        self.recent.clear()
        self.pos_hist.clear()
        self.save()
        Log.write(f"[CtxPersonal] Reset personalization for '{self.user}'")
