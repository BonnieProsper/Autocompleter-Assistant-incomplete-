import json
import os
from typing import Dict, Optional

class AdaptiveLearner:
    """
    Learns from user behavior by tracking which prediction sources perform well.
    It rewards sources that lead to correct predictions, penalizes those that don’t,
    and keeps the internal weighting system balanced over time.
    """
    def __init__(self, path="userdata/learning_profile.json"):
        self.path = path
        self.profile = self.load_or_init()

    # Profile Loading -----------------------------------------------------
    def load_or_init(self) -> Dict:
        """
        Loads the learning profile from disk if it exists.
        Otherwise creates a default profile and saves it.
        """
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)

        default_profile = {
            "semantic_weight": 0.50,
            "markov_weight": 0.30,
            "personal_weight": 0.15,
            "plugin_weight": 0.05,
            "boosts": {},              # category specific reinforcement boosts
            "history_len": 0           # reserved for future usage if needed
        }
        self.save(default_profile)
        return default_profile

    def save(self, data):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # Update weight factors --------------------------------------------------
    def reward(self, source: str, amount: float = 0.03):
        """
        Increases the weight for a prediction source that performed well.
        """
        key = f"{source}_weight"
        if key in self.profile:
            self.profile[key] = min(1.0, self.profile[key] + amount)
            self.normalize()
            self.save(self.profile)

    def penalize(self, source: str, amount: float = 0.03):
        """
        Decreases the weight for a prediction source that performed poorly.
        """
        key = f"{source}_weight"
        if key in self.profile:
            self.profile[key] = max(0.0, self.profile[key] - amount)
            self.normalize()
            self.save(self.profile)

    # Category Reinforcement ---------------------------------------
    def reinforce_category(self, category: str, delta: float = 0.02):
        """
        Applies a small persistent boost to a given command category.
        Useful for learning frequent user patterns (e.g prefers docker commands).
        """
        self.profile["boosts"][category] = self.profile["boosts"].get(category, 0.0) + delta
        self.save(self.profile)

    # Normalize Weights  -----------------------------------------------------
    def normalize(self):
        """
        Ensures all source weights sum to 1, prevents runaway values
        and keeps the decision system balanced.
        """
        total = (
            self.profile["semantic_weight"]
            + self.profile["markov_weight"]
            + self.profile["personal_weight"]
            + self.profile["plugin_weight"]
        )
        if total == 0:
            return

        for key in ["semantic_weight", "markov_weight", "personal_weight", "plugin_weight"]:
            self.profile[key] /= total

    # Public Accessors ---------------------------------------------------------------
    def get_weights(self):
        """Returns the normalized weights for each prediction source."""
        return {
            "semantic": self.profile["semantic_weight"],
            "markov": self.profile["markov_weight"],
            "personal": self.profile["personal_weight"],
            "plugin": self.profile["plugin_weight"],
        }

    def category_boost(self, category: str) -> float:
        return self.profile["boosts"].get(category, 0.0)
