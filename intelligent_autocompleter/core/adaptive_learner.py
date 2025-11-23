# intelligent_autocompleter/core/adaptive_learner.py
"""
AdaptiveLearner
---------------
Small online learning layer that adjusts the relative weights (influence) of
each prediction source (semantic, markov, personal, plugin) based on user behaviour.
Design goals:
 - small & understandable (not a research project (but make it one in future?)
 - reward/penalize hook for FeedbackTracker
 - decay keeps the system stable (prevents one source from dominating forever)
 - weights are always normalized to sum of approx 1
 - persistent JSON profile so behaviour survives restarts
"""

import os
import json
from typing import Dict, Any

# optional logger
try:
    from intelligent_autocompleter.utils.logger_utils import Log
except Exception:
    class Log:
        @staticmethod
        def write(m): print(m)

# Defaults/hyperparams ------------------------------------
DEFAULT_PROFILE = {
    "semantic_weight": 0.45,
    "markov_weight": 0.35,
    "personal_weight": 0.15,
    "plugin_weight": 0.05,
    "boosts": {},   # category / tag boosts (plugins may use)
}

REWARD_STEP = 0.03     # small nudges keep things smooth
PENALTY_STEP = 0.03
DECAY = 0.995         # tiny exponential decay per update
MIN_WEIGHT = 0.01      # floor to prevent collapse

PROFILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "userdata",
    "adaptive_profile.json"
)

class AdaptiveLearner:
    """
    Keeps track of weights for prediction sources.
    Public API:
        get_weights() for normalized mapping
        reward(source)
        penalize(source)
        reinforce_category(...)
        reset()
    """
    def __init__(self, path: str = None):
        self.path = path or PROFILE_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._profile = self._load_or_init()

    # Loading/saving ----------------------------------------------------------
    def _load_or_init(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # ensure all keys exist
                for k, v in DEFAULT_PROFILE.items():
                    data.setdefault(k, v)
                return data
            except Exception as e:
                Log.write(f"[AdaptiveLearner] load failed, using defaults: {e}")

        # fallback to defaults
        self._save(DEFAULT_PROFILE.copy())
        return DEFAULT_PROFILE.copy()

    def _save(self, data: Dict[str, Any]):
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception as e:
            Log.write(f"[AdaptiveLearner] save failed: {e}")

    # Public API -----------------------------------------------------------------
    def get_weights(self) -> Dict[str, float]:
        """
        Return normalized weight dict:
            {"semantic": w1, "markov": w2, ...}
        Stored keys are e.g. "semantic_weight".
        """
        keys = ["semantic_weight", "markov_weight", "personal_weight", "plugin_weight"]
        raw = [max(MIN_WEIGHT, float(self._profile.get(k, 0.0))) for k in keys]
        s = sum(raw) or 1.0

        # map "semantic_weight" to "semantic"
        out = {}
        for k, v in zip(keys, raw):
            label = k.replace("_weight", "")
            out[label] = v / s
        return out

    def reward(self, source: str, amount: float = REWARD_STEP):
        """
        Small positive bump to a source.
        Example: reward("semantic"), reward("plugin"), etc.
        """
        key = f"{source}_weight"
        if key not in self._profile:
            # Unknown source so silently ignore
            return

        # decay globally then add reward
        self._apply_decay()
        self._profile[key] = min(1.0, float(self._profile.get(key, 0.0)) + amount)
        self._normalize_and_save()

    def penalize(self, source: str, amount: float = PENALTY_STEP):
        """
        Small negative bump to a source.
        """
        key = f"{source}_weight"
        if key not in self._profile:
            return

        self._apply_decay()
        self._profile[key] = max(MIN_WEIGHT, float(self._profile.get(key, 0.0)) - amount)
        self._normalize_and_save()

    def reinforce_category(self, category: str, delta: float = 0.02):
        """
        Plugins can call this to express long-term bias.
        Not currently used by HybridPredictor itself, but available.
        """
        b = self._profile.setdefault("boosts", {})
        b[category] = float(b.get(category, 0.0)) + delta
        self._save(self._profile)

    def reset(self):
        """
        Reset everything back to built-in defaults.
        """
        self._profile = DEFAULT_PROFILE.copy()
        self._save(self._profile)

    # Internal helpers ------------------------------------------------
    def _apply_decay(self):
        """Apply mild exponential decay to all core weights."""
        for k in ("semantic_weight", "markov_weight",
                  "personal_weight", "plugin_weight"):
            v = float(self._profile.get(k, 0.0)) * DECAY
            self._profile[k] = max(MIN_WEIGHT, v)

    def _normalize_and_save(self):
        """
        Normalize all weights to sum to ~1 and persist to disk.
        """
        keys = ["semantic_weight", "markov_weight",
                "personal_weight", "plugin_weight"]

        vals = [max(MIN_WEIGHT, float(self._profile.get(k, 0.0))) for k in keys]
        s = sum(vals) or 1.0

        for k, v in zip(keys, vals):
            self._profile[k] = v / s

        self._save(self._profile)
