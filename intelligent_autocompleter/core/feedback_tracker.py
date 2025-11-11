# feedback_tracker.py - Persistent feedback tracking & adaptive learning support
# Tracks how users respond to model suggestions (accepted, ignored, or rejected), updates HybridPredictor
# This data can be used to refine future predictions adaptively.

import os
import csv
import time
from collections import defaultdict, deque

from logger_utils import Log


class FeedbackTracker:
    """
    Keeps track of user feedback to improve model performance over time.

    - Records each suggestion and whether it was accepted or rejected.
    - Computes acceptance ratios for individual suggestions or words.
    - Persists feedback to disk so it can be reloaded and reused.
    """

    def __init__(self, path: str = "data/feedback_log.csv"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Keep a temporary in-memory buffer before writing to disk
        self.buffer = deque(maxlen=500)

        # Count how often each suggestion is accepted or rejected
        self.accept_counts = defaultdict(int)
        self.reject_counts = defaultdict(int)

        # Load existing feedback history
        self.load()

    # Record feedback -----------------------------------------------
    def record(self, context: str, suggestion: str, accepted: bool):
        """Record a single feedback event."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        event = {
            "timestamp": timestamp,
            "context": context,
            "suggestion": suggestion,
            "accepted": accepted,
        }
        self.buffer.append(event)

        # Update in-memory statistics
        if accepted:
            self.accept_counts[suggestion] += 1
        else:
            self.reject_counts[suggestion] += 1

        status = "ACCEPTED" if accepted else "REJECTED"
        Log.write(f"[Feedback] {status}: '{suggestion}' (context: '{context}')")

    def acceptance_ratio(self, suggestion: str) -> float:
        """
        Returns how often a given suggestion has been accepted.

        The ratio is between 0 and 1.
        If there’s no feedback yet, a neutral score (0.5) is returned.
        """
        accepted = self.accept_counts[suggestion]
        rejected = self.reject_counts[suggestion]
        total = accepted + rejected

        return (accepted / total) if total > 0 else 0.5

    # Persistence ------------------------------------------------------
    def save(self):
        """Persist buffered feedback events to disk."""
        try:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "context", "suggestion", "accepted"])
                
                # Write header if file is new
                if f.tell() == 0:
                    writer.writeheader()
                
                # Flush all buffered entries
                while self.buffer:
                    writer.writerow(self.buffer.popleft())

        except Exception as e:
            Log.write(f"[Feedback] Error saving feedback: {e}")

    def load(self):
        """Load saved feedback from disk and rebuild acceptance statistics."""
        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    suggestion = row["suggestion"]
                    accepted = row["accepted"] == "True"

                    if accepted:
                        self.accept_counts[suggestion] += 1
                    else:
                        self.reject_counts[suggestion] += 1

            Log.write(f"[Feedback] Loaded {len(self.accept_counts)} tracked suggestions from history")

        except Exception as e:
            Log.write(f"[Feedback] Error loading feedback: {e}")
