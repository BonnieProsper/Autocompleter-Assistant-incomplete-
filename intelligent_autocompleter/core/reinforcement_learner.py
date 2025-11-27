# intelligent_autocompleter/core/reinforcement_learner.py
"""
ReinforcementLearner - merged feedback tracker + adaptive learner.

This module consolidates:
 - Feedback recording (append-only CSV + in-memory counters + buffer)
 - Online adaptive learner (weights for semantic / markov / personal / plugin)
 - Simple context window + confirmed_words (folded-in from ContextManager)
 - Persistence (CSV for events, JSON for learner profile)

Module purpose:
 - Clear API for the rest of the system:
    record_accept(context, word, source=None)
    record_reject(context, word, source=None)
    acceptance_ratio(word) -> float
    get_weights() -> dict
    reinforce_category(category)
    save()/load()/reset()
 - Robust: best-effort I/O, quiet failures logged with Log
 - Inspectability: dump_recent(), top_accepted()
"""

from __future__ import annotations
import os
import csv
import json
import time
import math
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

# Try project logger, fallback to simple print
try:
    from intelligent_autocompleter.utils.logger_utils import Log
except Exception:
    class Log:
        @staticmethod
        def write(m: str):
            print(m)
        @staticmethod
        def metric(*a, **k):
            pass

# Paths & defaults
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "userdata")
EVENTS_PATH = os.path.join(BASE_DIR, "feedback_events.csv")
PROFILE_PATH = os.path.join(BASE_DIR, "reinforcement_profile.json")

# Internal learner defaults
DEFAULT_PROFILE = {
    "semantic_weight": 0.45,
    "markov_weight": 0.35,
    "personal_weight": 0.15,
    "plugin_weight": 0.05,
    "boosts": {},  # category boosts
}

REWARD_STEP = 0.03
PENALTY_STEP = 0.03
DECAY = 0.995
MIN_WEIGHT = 0.01


@dataclass
class FeedbackEvent:
    timestamp: str
    context: str
    suggestion: str
    accepted: bool
    source: str

    def to_row(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "context": self.context,
            "suggestion": self.suggestion,
            "accepted": "True" if self.accepted else "False",
            "source": self.source or "",
        }


# -------------------------
# Internal: Event store
# -------------------------
class FeedbackStore:
    """
    Buffered append-only CSV writer + in-memory counters for quick stats.
    Optimized for simplicity/robustness.
    """

    def __init__(self, path: Optional[str] = None, buffer_max: int = 1000):
        os.makedirs(BASE_DIR, exist_ok=True)
        self.path = path or EVENTS_PATH
        self._buffer = deque(maxlen=buffer_max)
        self._accept_counts = defaultdict(int)
        self._reject_counts = defaultdict(int)
        self._recent = deque(maxlen=500)
        # load existing counts
        self._load_from_disk()

    # Recording
    def push(self, event: FeedbackEvent) -> None:
        self._buffer.append(event)
        self._recent.append(event)
        if event.accepted:
            self._accept_counts[event.suggestion] += 1
        else:
            self._reject_counts[event.suggestion] += 1
        # opportunistic flush if buffer big
        if len(self._buffer) >= (self._buffer.maxlen // 2):
            self.save()

    def record_accept(self, context: str, suggestion: str, source: Optional[str] = None) -> None:
        ev = FeedbackEvent(timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                           context=context or "",
                           suggestion=suggestion,
                           accepted=True,
                           source=source or "")
        self.push(ev)

    def record_reject(self, context: str, suggestion: str, source: Optional[str] = None) -> None:
        ev = FeedbackEvent(timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                           context=context or "",
                           suggestion=suggestion,
                           accepted=False,
                           source=source or "")
        self.push(ev)

    # Queries
    def acceptance_ratio(self, suggestion: str) -> float:
        a = self._accept_counts.get(suggestion, 0)
        r = self._reject_counts.get(suggestion, 0)
        total = a + r
        return (a / total) if total > 0 else 0.5

    def global_acceptance_rate(self) -> float:
        a = sum(self._accept_counts.values())
        r = sum(self._reject_counts.values())
        total = a + r
        return (a / total) if total > 0 else 0.5

    def top_accepted(self, n: int = 20) -> List[Tuple[str, int]]:
        return sorted(self._accept_counts.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def dump_recent(self, n: int = 200) -> List[Dict[str, Any]]:
        out = []
        for ev in list(self._recent)[-n:]:
            out.append(ev.to_row())
        # also include tail of file for inspection 
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as fh:
                    tail = fh.readlines()[-min(n, 200):]
                for ln in tail:
                    out.append({"raw": ln.strip()})
        except Exception:
            pass
        return out

    # Persistence
    def save(self) -> None:
        if not self._buffer:
            return
        try:
            write_header = not os.path.exists(self.path) or os.stat(self.path).st_size == 0
            with open(self.path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["timestamp", "context", "suggestion", "accepted", "source"])
                if write_header:
                    writer.writeheader()
                while self._buffer:
                    ev = self._buffer.popleft()
                    writer.writerow(ev.to_row())
            Log.write(f"[FeedbackStore] flushed events to {self.path}")
        except Exception as e:
            Log.write(f"[FeedbackStore] save failed: {e}")
            # best-effort, don't throw

    def _load_from_disk(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    sug = row.get("suggestion", "")
                    accepted = row.get("accepted", "False") == "True"
                    if accepted:
                        self._accept_counts[sug] += 1
                    else:
                        self._reject_counts[sug] += 1
            Log.write(f"[FeedbackStore] loaded counters from {self.path} (accepted={sum(self._accept_counts.values())})")
        except Exception as e:
            Log.write(f"[FeedbackStore] load failed: {e}")

    def reset(self) -> None:
        self._buffer.clear()
        self._accept_counts.clear()
        self._reject_counts.clear()
        self._recent.clear()
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
            Log.write("[FeedbackStore] reset persisted log")
        except Exception as e:
            Log.write(f"[FeedbackStore] reset failed: {e}")


# -------------------------
# Internal: Weight learner
# -------------------------
class WeightLearner:
    """
    Small adaptive learner that keeps normalized weights and category boosts.
    Persisted to JSON.
    """

    def __init__(self, path: Optional[str] = None):
        os.makedirs(BASE_DIR, exist_ok=True)
        self.path = path or PROFILE_PATH
        self._profile = DEFAULT_PROFILE.copy()
        self._load_or_init()

    def _load_or_init(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # backfill keys
                for k, v in DEFAULT_PROFILE.items():
                    data.setdefault(k, v)
                self._profile = data
                return
            except Exception:
                Log.write("[WeightLearner] load failed, reinitializing profile")
        # write default
        self._save()

    def _save(self) -> None:
        try:
            # apply small decay to weights on each save to avoid runaway dominance
            for k in ("semantic_weight", "markov_weight", "personal_weight", "plugin_weight"):
                self._profile[k] = max(MIN_WEIGHT, float(self._profile.get(k, 0.0)) * DECAY)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self._profile, fh, indent=2)
        except Exception:
            # quiet
            pass

    # API
    def get_weights(self) -> Dict[str, float]:
        keys = ["semantic_weight", "markov_weight", "personal_weight", "plugin_weight"]
        vals = [max(MIN_WEIGHT, float(self._profile.get(k, 0.0))) for k in keys]
        s = sum(vals) or 1.0
        # return short names as before (semantic, markov, personal, plugin)
        return {k.replace("_weight", ""): v / s for k, v in zip(keys, vals)}

    def _normalize_and_save(self) -> None:
        keys = ["semantic_weight", "markov_weight", "personal_weight", "plugin_weight"]
        vals = [max(MIN_WEIGHT, float(self._profile.get(k, 0.0))) for k in keys]
        s = sum(vals) or 1.0
        for k, v in zip(keys, vals):
            self._profile[k] = float(v) / s
        self._save()

    def reward(self, source: str, amount: float = REWARD_STEP) -> None:
        key = f"{source}_weight" if not source.endswith("_weight") else source
        if key not in self._profile:
            # unknown source, ignore quietly
            return
        self._profile[key] = min(1.0, float(self._profile.get(key, 0.0)) + float(amount))
        self._normalize_and_save()

    def penalize(self, source: str, amount: float = PENALTY_STEP) -> None:
        key = f"{source}_weight" if not source.endswith("_weight") else source
        if key not in self._profile:
            return
        self._profile[key] = max(MIN_WEIGHT, float(self._profile.get(key, 0.0)) - float(amount))
        self._normalize_and_save()

    def reinforce_category(self, category: str, delta: float = 0.02) -> None:
        boosts = self._profile.setdefault("boosts", {})
        boosts[category] = float(boosts.get(category, 0.0)) + float(delta)
        self._save()

    def category_boost(self, category: str) -> float:
        return float(self._profile.get("boosts", {}).get(category, 0.0))

    def reset(self) -> None:
        self._profile = DEFAULT_PROFILE.copy()
        self._save()


# -------------------------
# Public facade
# -------------------------
class ReinforcementLearner:
    """
    The facade used by HybridPredictor and the CLI.
    Example usage:
        rl = ReinforcementLearner()
        rl.record_accept("git push", "origin", source="markov")
        rl.get_weights()  # use to tune FusionRanker
    """

    def __init__(self,
                 events_path: Optional[str] = None,
                 profile_path: Optional[str] = None,
                 buffer_max: int = 1000,
                 verbose: bool = False):
        self.store = FeedbackStore(path=events_path, buffer_max=buffer_max)
        self.learner = WeightLearner(path=profile_path)
        self.verbose = bool(verbose)

        # basic context window + confirmed words (folded from ContextManager)
        self.context_window = deque(maxlen=20)
        self.confirmed = Counter()

    # Recording API  ----------------------------------------------------------------
    def record_accept(self, context: str, suggestion: str, source: Optional[str] = None) -> None:
        suggestion = suggestion.lower().strip()
        self.context_window.extend(w.lower() for w in (context or "").split() if w.isalpha())
        self.confirmed[suggestion] += 1
        self.store.record_accept(context or "", suggestion, source)
        # reward learner by source (best-effort)
        try:
            src = (source or "semantic").lower()
            # map common aliases
            if src == "embed":
                src = "semantic"
            if src not in ("semantic", "markov", "personal", "plugin"):
                src = "semantic"
            self.learner.reward(src)
        except Exception:
            pass
        if self.verbose:
            Log.write(f"[Reinforcement] accept '{suggestion}' src={source}")

    def record_reject(self, context: str, suggestion: str, source: Optional[str] = None) -> None:
        suggestion = suggestion.lower().strip()
        self.store.record_reject(context or "", suggestion, source)
        try:
            src = (source or "semantic").lower()
            if src == "embed":
                src = "semantic"
            if src not in ("semantic", "markov", "personal", "plugin"):
                src = "semantic"
            self.learner.penalize(src)
        except Exception:
            pass
        if self.verbose:
            Log.write(f"[Reinforcement] reject '{suggestion}' src={source}")

    # Queries -------------------------------------------------------
    def acceptance_ratio(self, suggestion: str) -> float:
        return self.store.acceptance_ratio(suggestion)

    def global_acceptance_rate(self) -> float:
        return self.store.global_acceptance_rate()

    def top_accepted(self, n: int = 20) -> List[Tuple[str, int]]:
        return self.store.top_accepted(n)

    def dump_recent(self, n: int = 200) -> List[Dict[str, Any]]:
        return self.store.dump_recent(n)

    def get_weights(self) -> Dict[str, float]:
        return self.learner.get_weights()

    def reinforce_category(self, category: str) -> None:
        try:
            self.learner.reinforce_category(category)
        except Exception:
            pass

    # Persistence ---------------------------------------------------
    def save(self) -> None:
        # flush buffered events + save learner profile
        try:
            self.store.save()
        except Exception:
            pass
        try:
            self.learner._save()
        except Exception:
            pass

    def reset(self) -> None:
        self.store.reset()
        self.learner.reset()
        self.context_window.clear()
        self.confirmed.clear()
        Log.write("[Reinforcement] reset all state")

