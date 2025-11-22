# intelligent_autocompleter/core/feedback_tracker.py
# Persistent feedback tracking and light telemetry for the autocompleter.
# Data on user responses can be used to refine future predictions adaptively.

import os
import csv
import time
from collections import defaultdict, deque
from typing import Optional, Dict, Any

# try local package logger, fallback to print
try:
    from intelligent_autocompleter.utils.logger_utils import Log
except Exception:
    try:
        from utils.logger_utils import Log
    except Exception:
        class _DummyLog:
            @staticmethod
            def write(msg): print(msg)
            @staticmethod
            def metric(*a, **k): pass
        Log = _DummyLog()

DEFAULT_PATH = os.path.join("data", "feedback_log.csv")


class FeedbackTracker:
    """
    Tracks user suggestion feedback e.g whether suggestion was accepted/rejected/ignored.
    - Keeps an in-memory buffer, writes append-only to CSV.
    - Tracks per-suggestion accept/reject counts for quick queries.
    - Optionally notifies AdaptiveLearner
    """
    def __init__(self, path: str = None, learner: Optional[object] = None, verbose: bool = False):
        self.path = path or DEFAULT_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # lightweight in-memory circular buffer, flush on save
        self.buffer = deque(maxlen=1000)

        # counts used for fast queries
        self.accepts = defaultdict(int)
        self.rejects = defaultdict(int)

        # last events, for debugging
        self.recent = deque(maxlen=200)

        # optional adaptive learner to call 
        self.learner = learner

        # feedback is silent as default (doesn't spam terminal). CLI can toggle.
        self.verbose = bool(verbose)

        # load historical stats from file if present
        self.load_counts()

    # recording API ------------------------------------------------
    def record(self, context: str, suggestion: str, accepted: bool, source: Optional[str] = None) -> None:
        """
        Records a single feedback event.
        Parameters:
         context: short context string (last token or sentence)
         suggestion: the suggested token/phrase
         accepted: True if accepted by user, False if rejected
         source: optional label for which source suggested it (markov/semantic/plugin/etc)
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        event = {"timestamp": ts, "context": context, "suggestion": suggestion, "accepted": accepted, "source": source or ""}
        self.buffer.append(event)
        self.recent.append(event)

        # update counts
        if accepted:
            self.accepts[suggestion] += 1
        else:
            self.rejects[suggestion] += 1

        # notify learner in a best-effort, non-blocking style (synchronous call)
        if self.learner is not None:
            try:
                if accepted:
                    self.learner.reward(source or "unknown")
                else:
                    self.learner.penalize(source or "unknown")
            except Exception as e:
                # keep silent, learner failing should not break feedback flow
                Log.write(f"[FeedbackTracker] learner hook failed: {e}")

        # optionally print a message to console using Log.write
        if self.verbose:
            st = "ACCEPT" if accepted else "REJECT"
            Log.write(f"[Feedback] {st} source={source} sug='{suggestion}' ctx='{context}'")

    def record_accept(self, context: str, suggestion: str, source: Optional[str] = None):
        self.record(context, suggestion, True, source)

    def record_reject(self, context: str, suggestion: str, source: Optional[str] = None):
        self.record(context, suggestion, False, source)

    # queries ------------------------------------------------------
    def acceptance_ratio(self, suggestion: str) -> float:
        """
        Returns acceptance ratio for a suggestion (how often a given suggestion has been accepted)
        Ratio between 0 and 1.
        A neutral score of 0.5 is returned if no other data is returned.
        """
        a = self.accepts.get(suggestion, 0)
        r = self.rejects.get(suggestion, 0)
        if (a + r) == 0:
            return 0.5
        return a / (a + r)

    def stats(self) -> Dict[str, Any]:
        """Quick stats summary."""
        total_events = sum(self.accepts.values()) + sum(self.rejects.values())
        top_accepted = sorted(self.accepts.items(), key=lambda kv: -kv[1])[:10]
        top_rejected = sorted(self.rejects.items(), key=lambda kv: -kv[1])[:10]
        return {
            "events_in_memory": len(self.buffer),
            "total_events": total_events,
            "unique_suggestions_tracked": len(set(list(self.accepts) + list(self.rejects))),
            "top_accepted": top_accepted,
            "top_rejected": top_rejected,
        }

    # Persistence --------------------------------------------------
    def save(self):
        """Persist buffer to CSV (append)."""
        if not self.buffer:
            return
        try:
            first_write = not os.path.exists(self.path)
            with open(self.path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["timestamp", "context", "suggestion", "accepted", "source"])
                if first_write:
                    writer.writeheader()
                while self.buffer:
                    ev = self.buffer.popleft()
                    # ensure accepted is written as True/False
                    ev = ev.copy()
                    ev["accepted"] = str(bool(ev["accepted"]))
                    writer.writerow(ev)
            if self.verbose:
                Log.write(f"[Feedback] flushed events to {self.path}")
        except Exception as e:
            Log.write(f"[Feedback] save failed: {e}")

    def load_counts(self):
        """Load existing CSV and reconstruct counts. Does this quietly/with best effort."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    sug = row.get("suggestion", "")
                    accepted = row.get("accepted", "False") == "True"
                    if accepted:
                        self.accepts[sug] += 1
                    else:
                        self.rejects[sug] += 1
            Log.write(f"[Feedback] loaded stats from {self.path} (accepted={sum(self.accepts.values())})")
        except Exception as e:
            Log.write(f"[Feedback] load failed: {e}")

    # Convenience/debug -----------------------------------------
    def dump_recent(self, n: int = 50):
        """Return the last n recorded events (list of dicts)."""
        return list(self.recent)[-n:]

            Log.write(f"[Feedback] Error loading feedback: {e}")
