# intelligent_autocompleter/core/feedback_tracker.py
"""
FeedbackTracker
Lightweight telemetry + feedback recording for the autocompleter.
 - quiet by default (no terminal spam unless enabled)
 - append-only CSV log (safe for crashes)
 - quick in-memory stats (accept/reject counts)
 - optional hook into AdaptiveLearner
 - small API for HybridPredictor and plugins
"""

import os
import csv
import time
from collections import defaultdict, deque
from typing import Optional, Dict, Any, List

# Best-effort logger import
try:
    from intelligent_autocompleter.utils.logger_utils import Log
except Exception:
    class _Dummy:
        @staticmethod
        def write(msg): print(msg)
        @staticmethod
        def metric(*a, **k): pass
    Log = _Dummy()

DEFAULT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "feedback_log.csv"
)


class FeedbackTracker:
    """
    Tracks user reactions to suggestions.
    Public API:
      record_accept(word, src, context)
      record_reject(word, src, context)
      acceptance_ratio(word)
      stats()
      save()
      dump_recent()
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        learner: Optional[object] = None,
        verbose: bool = False,
        buffer_max: int = 1000,
    ):
        self.path = path or DEFAULT_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # ring buffer storing unsaved events
        self._buffer = deque(maxlen=buffer_max)

        # quick accept/reject counters
        self._accept = defaultdict(int)
        self._reject = defaultdict(int)

        # debugging tail buffer
        self._recent = deque(maxlen=200)

        # integration point: AdaptiveLearner instance
        self._learner = learner

        # quiet by default
        self.verbose = bool(verbose)

        # load historical stats
        self._load_previous()

    # Event recording --------------------------------------------------------------
    def record_accept(self, word: str, src: str = "", context: str = ""):
        self._record("accepted", word, src, context)

    def record_reject(self, word: str, src: str = "", context: str = ""):
        self._record("rejected", word, src, context)

    def _record(self, event_type: str, word: str, src: str, context: str):
        """Internal helper for adding an event."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        ev = {
            "timestamp": ts,
            "type": event_type,
            "word": word,
            "src": src or "",
            "context": context or "",
        }

        self._buffer.append(ev)
        self._recent.append(ev)

        # update stats
        if event_type == "accepted":
            self._accept[word] += 1
        elif event_type == "rejected":
            self._reject[word] += 1

        # optional learning hook
        if self._learner:
            try:
                if event_type == "accepted":
                    self._learner.reward(src)
                else:
                    self._learner.penalize(src)
            except Exception as e:
                Log.write(f"[Feedback] learner hook failed: {e}")

        if self.verbose:
            Log.write(f"[Feedback] {event_type.upper()} '{word}' from {src}")

        # opportunistic flush (keeps memory stable)
        if len(self._buffer) >= self._buffer.maxlen // 2:
            self.save()

    # Stats/queries -------------------------------------------------------------------------
    def acceptance_ratio(self, word: str) -> float:
        """Probability score describing how often a suggestion is accepted."""
        a = self._accept.get(word, 0)
        r = self._reject.get(word, 0)
        total = a + r
        return (a / total) if total else 0.5

    def stats(self) -> Dict[str, Any]:
        """Simple stats bundle used by CLI debug view."""
        total = sum(self._accept.values()) + sum(self._reject.values())
        return {
            "total_events": total,
            "unique_words": len(set(list(self._accept) + list(self._reject))),
            "top_accepted": sorted(self._accept.items(), key=lambda kv: -kv[1])[:10],
            "top_rejected": sorted(self._reject.items(), key=lambda kv: -kv[1])[:10],
        }

    # Persistence ----------------------------------------------------------------------
    def save(self):
        """Flush buffered events to CSV."""
        if not self._buffer:
            return

        write_header = not os.path.exists(self.path)

        try:
            with open(self.path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["timestamp", "type", "word", "src", "context"]
                )
                if write_header:
                    writer.writeheader()

                while self._buffer:
                    writer.writerow(self._buffer.popleft())

            if self.verbose:
                Log.write(f"[Feedback] flushed to {self.path}")

        except Exception as e:
            Log.write(f"[Feedback] save failed: {e}")

    def _load_previous(self):
        """Reconstruct accept/reject counters from disk."""
        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    t = (row.get("type") or "").lower()
                    word = row.get("word") or ""
                    if t == "accepted":
                        self._accept[word] += 1
                    elif t == "rejected":
                        self._reject[word] += 1

            Log.write(f"[Feedback] loaded historic feedback")
        except Exception as e:
            Log.write(f"[Feedback] load failed: {e}")

    # Debug helpers -------------------------------------------------------------------
    def dump_recent(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return the last n in-memory events."""
        return list(self._recent)[-n:]
