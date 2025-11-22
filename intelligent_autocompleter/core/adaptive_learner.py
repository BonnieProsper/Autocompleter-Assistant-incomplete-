"""
adaptive_learner.py

An adaptive learning module that adjusts weighting (the influence of each prediction source)
between each prediction source (semantic, markov, personal, plugin-based predictions)
based on user behaviour. This makes the autocompleter more personalized to the user over 
time by adjusting weights based on real user feedback. This module's purpose is:
- to reward/penalize updates
- category specific reinforcement
- normalize weights with slight decay to prevent runaway dominance
- exponential decay (prevents runaway dominance)
- temperature scaling (for long term stability under noisy feedback)
- persist learning state on disk
"""

import os
import json
from typing import Dict

# try to reuse logger if available
try:
    from intelligent_autocompleter.utils.logger_utils import Log
except Exception:
    class Log:
        @staticmethod
        def write(m): print(m)

DEFAULT_PATH = os.path.join("userdata", "adaptive_profile.json")

# some sane constants
DEFAULT_PROFILE = {
    "semantic_weight": 0.5,
    "markov_weight": 0.3,
    "personal_weight": 0.15,
    "plugin_weight": 0.05,
    "boosts": {}
}

DECAY = 0.995        # slight decay each update (to prevent runaway/domination of any one source)
LEARN_STEP = 0.03   # controls weight adjustment for each reward/penalty
MIN_WEIGHT = 0.01   # minimum floor for stabilising

class AdaptiveLearner:
    """
    Keep normalized weights for prediction sources.
    - reward(source) nudges the given source up a little
    - penalize(source) nudges it down
    - normalize() keeps sum at approx 1
    - persistent learning profile saved to disk
    """
    def __init__(self, path: str = None):
        self.path = path or DEFAULT_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.profile = self.load()  # dict

    def load(self) -> Dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # ensure keys exist
                    for k, v in DEFAULT_PROFILE.items():
                        data.setdefault(k, v)
                    return data
            except Exception:
                Log.write("[AdaptiveLearner] failed to load profile, resetting to defaults")
        # write defaults
        self.save(DEFAULT_PROFILE.copy())
        return DEFAULT_PROFILE.copy()

    def save(self, data: Dict):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            Log.write(f"[AdaptiveLearner] save failed: {e}")

    # Public API ------------------------------------------------
    def get_weights(self) -> Dict[str, float]:
        """Return normalized weights mapping (semantic, markov, personal, plugin)."""
        # get current values
        keys = ["semantic_weight", "markov_weight", "personal_weight", "plugin_weight"]
        vals = [max(MIN_WEIGHT, float(self.profile.get(k, 0.0))) for k in keys]
        tot = sum(vals) or 1.0
        return {k.replace("weight", ""): v / tot for k, v in zip(keys, vals)}

    def reward(self, source: str, amount: float = LEARN_STEP):
        """Nudge the weight of a source upward to reward it for a correct prediction."""
        key = f"{source}weight"
        if key not in self.profile:
            # ignore unknown sources
            return
        # apply slight decay to all, then bump selected
        self.apply_decay()
        self.profile[key] = min(1.0, float(self.profile.get(key, 0.0)) + amount)
        self.normalize_and_persist()

    def penalize(self, source: str, amount: float = LEARN_STEP):
        """Nudge the weight of a source downward to penalize it for an incorrect prediction."""
        key = f"{source}weight"
        if key not in self.profile:
            return
        self.apply_decay()
        self.profile[key] = max(MIN_WEIGHT, float(self.profile.get(key, 0.0)) - amount)
        self.apply_decay()

    def reinforce_category(self, category: str, amount: float = 0.02):
        """Apply a persistent boost for a category (plugins use this)."""
        boosts = self.profile.setdefault("boosts", {})
        boosts[category] = float(boosts.get(category, 0.0)) + amount
        self.save(self.profile)

    # Internals -------------------------------------------------
    def apply_decay(self):
        # decay for every update to keep the system drifting slowly
        for k in ["semantic_weight", "markov_weight", "personal_weight", "plugin_weight"]:
            self.profile[k] = max(MIN_WEIGHT, float(self.profile.get(k, 0.0)) * DECAY)

    def normalize_and_persist(self):
        keys = ["semantic_weight", "markov_weight", "personal_weight", "plugin_weight"]
        vals = [max(MIN_WEIGHT, float(self.profile.get(k, 0.0))) for k in keys]
        s = sum(vals) or 1.0
        for k, v in zip(keys, vals):
            self.profile[k] = float(v) / s
        self.save(self.profile)

    def reset(self):
        self.profile = DEFAULT_PROFILE.copy()
        self.save(self.profile)
