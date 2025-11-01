# context_predictor.py
# ---------------------------------------------------------------------
# Learns word co-occurrence patterns for context-aware prediction.
# Acts as a lightweight n-gram model without external dependencies.
# ---------------------------------------------------------------------

from collections import defaultdict, Counter
import random


class ContextPredictor:
    """Learns word sequences and predicts likely next words."""

    def __init__(self):
        self.bigram = defaultdict(Counter)
        self.unigram = Counter()

    def learn(self, sentence: str):
        """Learn from a sentence."""
        words = [w.lower() for w in sentence.strip().split() if w.isalpha()]
        for i in range(len(words) - 1):
            self.bigram[words[i]][words[i + 1]] += 1
            self.unigram[words[i]] += 1
        if words:
            self.unigram[words[-1]] += 1

    def predict(self, word: str, top_k: int = 3):
        """Predict the most likely next words."""
        next_words = self.bigram.get(word.lower(), None)
        if not next_words:
            # Backoff to most common unigrams
            return [w for w, _ in self.unigram.most_common(top_k)]
        ranked = next_words.most_common(top_k)
        return [w for w, _ in ranked]

    def random_next(self, word: str):
        """Random next word weighted by frequency."""
        next_words = self.bigram.get(word.lower(), {})
        if not next_words:
            return random.choice(list(self.unigram.keys()))
        words, weights = zip(*next_words.items())
        return random.choices(words, weights=weights, k=1)[0]
