# intelligent_autocompleter/core/reinforcement_learner.py
"""
ReinforcementLearner - merged feedback store + lightweight (extend it further?? check against original adaptiveLearner file to do so) adaptive learner.

- API used by HybridPredictor:
    record_accept(context, suggestion, source=None)
    record_reject(context, suggestion, source=None)
    get_weights() -> Dict[str,float]
    acceptance_ratio(suggestion) -> float
    save(), reset(), dump_recent(), top_accepted()
- Thread-safe flush and profile save (simple lock)
- Atomic JSON profile writes
- Keeps CSV append-only feedback log and in-memory counters for faster queries
"""

from __future__ import annotations

import csv
import json
import os
import time
import threading
from collections import defaultdict, deque, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

try:
    from intelligent_autocompleter.utils.logger_utils import Log
except Exception:
    class Log:
        @staticmethod
        def write(msg: str) -> None:
            print(msg)
        @staticmethod
        def metric(*a, **k) -> None:
            pass

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "userdata")
EVENTS_FILENAME = "feedback_events.csv"
PROFILE_FILENAME = "reinforcement_profile.json"
EVENTS_PATH = os.path.join(BASE_DIR, EVENTS_FILENAME)
PROFILE_PATH = os.path.join(BASE_DIR, PROFILE_FILENAME)

# Defaults and hyperparams
DEFAULT_PROFILE = {
    "semantic_weight": 0.45,
    "markov_weight": 0.35,
    "personal_weight": 0.15,
    "plugin_weight": 0.05,
    "boosts": {},
}
REWARD_STEP = 0.03
PENALTY_STEP = 0.03
DECAY = 0.995
MIN_WEIGHT = 0.01

# Dataclasses
@dataclass
class FeedbackEvent:
    timestamp: str
    context: str
    suggestion: str
    accepted: bool
    source: str

    def to_row(self) -> Dict[str, str]:
        return {
            "timestamp": self.timestamp,
            "context": self.context,
            "suggestion": self.suggestion,
            "accepted": "True" if self.accepted else "False",
            "source": self.source or "",
        }


# -----------------------------
# Feedback store (CSV + memory)
# -----------------------------
class FeedbackStore:
    """
    Buffered append-only CSV writer + in-memory counters for fast queries.
    Best-effort persistence, failures are logged but do not crash the app.
    """

    def __init__(self, path: Optional[str] = None, buffer_max: int = 1000):
        os.makedirs(BASE_DIR, exist_ok=True)
        self.path = path or EVENTS_PATH
        self._buffer = deque(maxlen=buffer_max)
        self._accept_counts: Dict[str, int] = defaultdict(int)
        self._reject_counts: Dict[str, int] = defaultdict(int)
        self._recent: deque[FeedbackEvent] = deque(maxlen=500)
        self._lock = threading.Lock()
        self._load_counters_from_disk()

    def _load_counters_from_disk(self) -> None:
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
            Log.write(f"[FeedbackStore] loaded counters (accepted={sum(self._accept_counts.values())})")
        except Exception as e:
            Log.write(f"[FeedbackStore] load failed: {e}")

    def push(self, ev: FeedbackEvent) -> None:
        with self._lock:
            self._buffer.append(ev)
            self._recent.append(ev)
            if ev.accepted:
                self._accept_counts[ev.suggestion] += 1
            else:
                self._reject_counts[ev.suggestion] += 1
            # opportunistic flush if buffer grows large
            if len(self._buffer) >= (self._buffer.maxlen // 2):
                self.save()

    def record_accept(self, context: str, suggestion: str, source: Optional[str] = None) -> None:
        ev = FeedbackEvent(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            context=context or "",
            suggestion=suggestion,
            accepted=True,
            source=source or "",
        )
        self.push(ev)

    def record_reject(self, context: str, suggestion: str, source: Optional[str] = None) -> None:
        ev = FeedbackEvent(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            context=context or "",
            suggestion=suggestion,
            accepted=False,
            source=source or "",
        )
        self.push(ev)

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
        items = sorted(self._accept_counts.items(), key=lambda kv: kv[1], reverse=True)
        return items[:n]

    def dump_recent(self, n: int = 200) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with self._lock:
            for ev in list(self._recent)[-n:]:
                out.append(ev.to_row())
            # append last lines of persistent file for inspection
            try:
                if os.path.exists(self.path):
                    with open(self.path, "r", encoding="utf-8") as fh:
                        tail = fh.readlines()[-min(n, 200):]
                    for ln in tail:
                        out.append({"raw": ln.strip()})
            except Exception:
                pass
        return out

    def save(self) -> None:
        """Flush buffer to disk (append). Thread-safe."""
        with self._lock:
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

    def reset(self) -> None:
        with self._lock:
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


# -----------------------------
# Simple adaptive weight learner
# -----------------------------
class WeightLearner:
    """
    Stores normalized weights and simple boosts. Persists to JSON.
    """

    def __init__(self, path: Optional[str] = None):
        os.makedirs(BASE_DIR, exist_ok=True)
        self.path = path or PROFILE_PATH
        self._profile = dict(DEFAULT_PROFILE)  # copy
        self._lock = threading.Lock()
        self._load_or_init()

    def _load_or_init(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # Backfill keys to be tolerant of older profiles
                for k, v in DEFAULT_PROFILE.items():
                    data.setdefault(k, v)
                self._profile = data
                return
            except Exception:
                Log.write("[WeightLearner] failed to load profile, reinitializing")
        # persist default
        self._save_locked()

    def _save_locked(self) -> None:
        # Called with lock held
        try:
            # apply slight decay to avoid runaway dominance
            for k in ("semantic_weight", "markov_weight", "personal_weight", "plugin_weight"):
                self._profile[k] = max(MIN_WEIGHT, float(self._profile.get(k, 0.0)) * DECAY)
            tmp_path = self.path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self._profile, fh, indent=2)
            os.replace(tmp_path, self.path)
        except Exception as e:
            Log.write(f"[WeightLearner] save failed: {e}")

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _normalize_profile(self) -> None:
        # Ensure weights sum to 1 and respect MIN_WEIGHT
        keys = ["semantic_weight", "markov_weight", "personal_weight", "plugin_weight"]
        vals = [max(MIN_WEIGHT, float(self._profile.get(k, 0.0))) for k in keys]
        s = sum(vals) or 1.0
        for k, v in zip(keys, vals):
            self._profile[k] = float(v) / s

    def get_weights(self) -> Dict[str, float]:
        with self._lock:
            keys = ["semantic_weight", "markov_weight", "personal_weight", "plugin_weight"]
            vals = [max(MIN_WEIGHT, float(self._profile.get(k, 0.0))) for k in keys]
            s = sum(vals) or 1.0
            return {k.replace("_weight", ""): float(v / s) for k, v in zip(keys, vals)}

    def _adjust(self, key: str, delta: float) -> None:
        with self._lock:
            if key not in self._profile:
                # ignore unknown keys
                return
            self._profile[key] = max(MIN_WEIGHT, float(self._profile.get(key, 0.0)) + float(delta))
            self._normalize_profile()
            self._save_locked()

    def reward(self, source: str, amount: float = REWARD_STEP) -> None:
        key = f"{source}_weight" if not source.endswith("_weight") else source
        self._adjust(key, float(amount))

    def penalize(self, source: str, amount: float = PENALTY_STEP) -> None:
        key = f"{source}_weight" if not source.endswith("_weight") else source
        self._adjust(key, -float(amount))

    def reinforce_category(self, category: str, delta: float = 0.02) -> None:
        with self._lock:
            boosts = self._profile.setdefault("boosts", {})
            boosts[category] = float(boosts.get(category, 0.0)) + float(delta)
            self._save_locked()

    def category_boost(self, category: str) -> float:
        with self._lock:
            return float(self._profile.get("boosts", {}).get(category, 0.0))

    def reset(self) -> None:
        with self._lock:
            self._profile = dict(DEFAULT_PROFILE)
            self._save_locked()


# -----------------------------
# Public facade
# -----------------------------
class ReinforcementLearner:
    """
    Combines FeedbackTracker + AdaptiveLearner.

    Usage by HybridPredictor:
      rl = ReinforcementLearner()
      rl.record_accept(context, suggestion, source)
      weights = rl.get_weights()
    """

    def __init__(self, events_path: Optional[str] = None, profile_path: Optional[str] = None, buffer_max: int = 1000, verbose: bool = False):
        self.store = FeedbackStore(path=events_path, buffer_max=buffer_max)
        self.learner = WeightLearner(path=profile_path)
        self.verbose = bool(verbose)
        # small in-memory context & confirmed suggestions (helpful telemetry)
        self.context_window: deque[str] = deque(maxlen=20)
        self.confirmed: Counter = Counter()

    # Recording API
    def record_accept(self, context: str, suggestion: str, source: Optional[str] = None) -> None:
        suggestion = suggestion.lower().strip()
        self.context_window.extend(w.lower() for w in (context or "").split() if w.isalpha())
        self.confirmed[suggestion] += 1
        self.store.record_accept(context or "", suggestion, source)
        # reward learner by source (normalized mapping)
        try:
            src = (source or "semantic").lower()
            if src == "embed":
                src = "semantic"
            if src not in ("semantic", "markov", "personal", "plugin"):
                src = "semantic"
            self.learner.reward(src)
        except Exception:
            pass
        if self.verbose:
            Log.write(f"[ReinforcementLearner] accept '{suggestion}' src={source}")

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
            Log.write(f"[ReinforcementLearner] reject '{suggestion}' src={source}")

    # Queries
    def acceptance_ratio(self, suggestion: str) -> float:
        return self.store.acceptance_ratio(suggestion)

    def global_acceptance_rate(self) -> float:
        return self.store.global_acceptance_rate()

    def top_accepted(self, n: int = 20) -> List[Tuple[str, int]]:
        return self.store.top_accepted(n)

    def dump_recent(self, n: int = 200) -> List[Dict[str, Any]]:
        return self.store.dump_recent(n)

    def get_weights(self) -> Dict[str, float]:
        # returns short names: semantic, markov, personal, plugin
        return self.learner.get_weights()

    def reinforce_category(self, category: str) -> None:
        try:
            self.learner.reinforce_category(category)
        except Exception:
            pass

    # Persistence
    def save(self) -> None:
        # flush store and learner profile
        try:
            self.store.save()
        except Exception:
            pass
        try:
            self.learner.save()
        except Exception:
            pass

    def reset(self) -> None:
        self.store.reset()
        self.learner.reset()
        self.context_window.clear()
        self.confirmed.clear()
        Log.write("[ReinforcementLearner] reset all state")
