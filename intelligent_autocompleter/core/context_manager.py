# possibly uneeded, check later

import json
from collections import deque, Counter
from pathlib import Path
from typing import List, Dict, Optional


class ContextManager:
    """
    Tracks user typing context across sessions:
    - Recent tokens and topics
    - Word confirmation frequency
    - Short-term context weighting
    """

    def __init__(self, userdata_path: Path, memory_size: int = 5):
        self.userdata_path = userdata_path
        self.memory_size = memory_size
        self.context_window = deque(maxlen=memory_size)
        self.confirmed_words = Counter()
        self.topics = Counter()

        self._load_context()

    def _load_context(self):
        if not self.userdata_path.exists():
            return
        try:
            data = json.loads(self.userdata_path.read_text())
            self.context_window.extend(data.get("context_window", []))
            self.confirmed_words.update(data.get("confirmed_words", {}))
            self.topics.update(data.get("topics", {}))
        except json.JSONDecodeError:
            pass  # Fail gracefully if corrupt

    def _save_context(self):
        data = {
            "context_window": list(self.context_window),
            "confirmed_words": dict(self.confirmed_words),
            "topics": dict(self.topics),
        }
        self.userdata_path.write_text(json.dumps(data, indent=2))

    def add_token(self, token: str):
        """Track token in the context window."""
        self.context_window.append(token.lower())
        self._save_context()

    def confirm_word(self, word: str):
        """User selected this word — boost its future probability."""
        word = word.lower()
        self.confirmed_words[word] += 1
        # Infer topics heuristically (you could link to NLP later)
        if len(word) > 4:
            self.topics[word[:3]] += 1
        self._save_context()

    def get_context_bias(self) -> Dict[str, float]:
        """
        Return adaptive bias weights that can be applied by the predictor.
        These weights subtly tilt results based on user habits.
        """
        bias = {}
        for word, count in self.confirmed_words.most_common(10):
            bias[word] = 1 + (count / 10)  # modest boost
        return bias

    def summarize_context(self) -> str:
        """Return a short summary for display in CLI."""
        recent = ", ".join(self.context_window) or "None"
        top_words = ", ".join([w for w, _ in self.confirmed_words.most_common(3)]) or "None"
        return f"Recent: {recent} | Top words: {top_words}"
