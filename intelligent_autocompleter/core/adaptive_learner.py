"""
adaptive_learner.py

An online-learning module that adjusts weighting (influence of each prediction source)
between each prediction source (semantic, markov, personal, plugin-based predictions)
based on user behaviour. The goal is to make the autocompleter feel more personalized over 
time by adjusting weights based on real user feedback.
This module's purpose is:
- to reward/penalize updates
- category specific reinforcement
- normalize weights with slight decay to prevent runaway dominance
- exponential decay (prevents runaway dominance)
- temperature scaling (for long term stability under noisy feedback)
- persist learning state on disk
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Dict

# Configuration Constants ------------------------------------------------------
DEFAULT_PROFILE_PATH = "userdata/adaptive_profile.json"

# define learning dynamics
LEARNING_RATE = 0.03       # controls weight adjustment per reward/penalty
CATEGORY_RATE = 0.02       # controls reinforcement for categories 
DECAY_FACTOR = 0.98        # slight decay to prevent runaway growth, prevents one source from taking over
MIN_WEIGHT = 0.02          # minimum floor for stabilising
MAX_WEIGHT = 1.00          # max bound before normalization

# Data Structures ------------------------------------------------------------------
@dataclass
class LearnerProfile:
    """Stores the current adaptive learning state on disk."""
    semantic_weight: float = 0.50
    markov_weight: float = 0.30
    personal_weight: float = 0.15
    plugin_weight: float = 0.05

    # reinforcement for categories
    boosts: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {
            "semantic_weight": self.semantic_weight,
            "markov_weight": self.markov_weight,
            "personal_weight": self.personal_weight,
            "plugin_weight": self.plugin_weight,
            "boosts": self.boosts,
        }

# AdaptiveLearner Implementation --------------------------------------------------------
class AdaptiveLearner:
    """
    Adjusts the weighting between multiple prediction sources based on user
    behavior. Over time the system becomes more aligned with what the user
    personally finds helpful.
    Features:
    - reward/penalize with smooth exponential learning
    - weight normalization with drift control to prevent runaway dominance through decay
    - temperature scaling for stability under noisy feedback
    - category reinforcement through learning user specific command habits
    - persistent learning profile saved to disk
    """
    def __init__(self, path: str = DEFAULT_PROFILE_PATH):
        self.path = path
        self.profile = self.load_or_init()
        self.ensure_directories()

    # Persistence ------------------------------------------------------
    def ensure_directories(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def load_or_init(self) -> LearnerProfile:
        """
        Load the saved profile if it exists; otherwise create a default profile.
        """
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return LearnerProfile(**data)
            except Exception:
                pass

        # fallback to defaults
        profile = LearnerProfile()
        self.save(profile)
        return profile

    def save(self, profile: LearnerProfile):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(profile.as_dict(), f, indent=2)
        except Exception:
            # if saving fails its not critical
            pass

    # Core Weight Adjustments -------------------------------------------------------
    def apply_decay(self):
        """
        Applies gentle decay to all weights to prevent any single source from
        permanently dominating due to accumulated rewards.
        """
        for attr in ["semantic_weight", "markov_weight", "personal_weight", "plugin_weight"]:
            current = getattr(self.profile, attr)
            decayed = max(MIN_WEIGHT, current * DECAY_FACTOR)
            setattr(self.profile, attr, decayed)

    def normalize(self):
        """Normalizes all weights so they sum to 1 (L1 norm)."""
        keys = ["semantic_weight", "markov_weight", "personal_weight", "plugin_weight"]
        values = [getattr(self.profile, k) for k in keys]
        total = sum(values)
        if total == 0:
            return  # avoid division error

        for key in keys:
            setattr(self.profile, k, getattr(self.profile, k) / total)

    def update_weight(self, source: str, delta: float):
        """
        Applies a bounded change (+ or -) to a particular prediction source.
        """
        key = f"{source}_weight"
        if hasattr(self.profile, key):
            current = getattr(self.profile, attr)
            new_value = max(MIN_WEIGHT, min(MAX_WEIGHT, current + delta))
            setattr(self.profile, attr, new_value)

    # Public API ------------------------------------------------------------------
    def reward(self, source: str):
        """
        Reward a prediction source for producing a correct prediction.
        """
        self.apply_decay()
        self.update_weight(source, +LEARNING_RATE)
        self.normalize()
        self.save(self.profile)

    def penalize(self, source: str):
        """
        Penalize a source for producing a incorrect/unhelpful prediction.
        """
        self.apply_decay()
        self.update_weight(source, -LEARNING_RATE)
        self.normalize()
        self.save(self.profile)

    def reinforce_category(self, category: str):
        """
        Apply a persistent positive bias to a specific command category that user uses frequently.
        Helps the system learn individual user habits.
        """
        boosts = self.profile.boosts
        boosts[category] = boosts.get(category, 0.0) + CATEGORY_RATE
        self.save(self.profile)

    def get_weights(self) -> Dict[str, float]:
        """Return current normalized weights of each source."""
        return {
            "semantic": self.profile.semantic_weight,
            "markov": self.profile.markov_weight,
            "personal": self.profile.personal_weight,
            "plugin": self.profile.plugin_weight,
        }

    def category_boost(self, category: str) -> float:
        """Return the boost value for a category."""
        return self.profile.boosts.get(category, 0.0)

    # Debug and Maintenance -------------------------------------------------------
    def reset(self):
        """Reset the profile to defaults."""
        if os.path.exists(self.path):
            os.remove(self.path)
        self.profile = self.load_or_init()
    def category_boost(self, category: str) -> float:
        return self.profile["boosts"].get(category, 0.0)
