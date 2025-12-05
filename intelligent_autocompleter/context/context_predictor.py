# context_predictor.py
# Lightweight bigram-based language model for context-aware predictions.
# Provides next-word suggestions using frequency-weighted estimates.
# Designed to complement the Trie autocomplete engine.
# ---------------------------------------------------------------------

from __future__ import annotations
from collections import defaultdict, Counter
from typing import List, Optional
import random
import re


class ContextPredictor:
    """
    Model using unigram and bigram frequencies.
    Provides context-aware next-word predictions for the autocompleter system.
    Features:
    - Learns word co-occurrence patterns from natural text input.
    - Predicts top-K likely next words using bigram frequencies.
    - Falls back to global unigram frequencies when no bigram context exists.
    - Supports frequency-weighted random sampling for more natural predictions.
    """

    WORD_RE = re.compile(r"[a-zA-Z]+")

    def __init__(self, seed: Optional[int] = None):
        """
        Create a new ContextPredictor.
        Args:
            seed: Optional random seed for deterministic sampling (useful for tests).
        """
        self.bigram: dict[str, Counter] = defaultdict(Counter)
        self.unigram: Counter = Counter()

        if seed is not None:
            random.seed(seed)

    def learn(self, sentence: str) -> None:
        """
        Learn token transitions from a sentence.
        Args:
            sentence: Arbitrary user-supplied text. Only alphabetic tokens are used.
        """
        words = [w.lower() for w in self.WORD_RE.findall(sentence)]
        if not words:
            return

        # Update bigram counts
        for w1, w2 in zip(words, words[1:]):
            self.bigram[w1][w2] += 1

        # Update unigram counts
        self.unigram.update(words)

    def predict(self, word: str, top_k: int = 3) -> List[str]:
        """
        Return the top-K most likely next words following the given word.

        Args:
            word: The preceding word to predict from.
            top_k: Number of predictions to return.

        Returns:
            A list of likely next words (up to top_k).
        """
        word = word.lower()
        next_words = self.bigram.get(word)

        if not next_words:
            # Backoff: return globally common words
            return [w for w, _ in self.unigram.most_common(top_k)]

        return [w for w, _ in next_words.most_common(top_k)]

    def random_next(self, word: str) -> Optional[str]:
        """
        Return a random next word weighted by observed frequencies.

        Args:
            word: The word to sample a successor for.

        Returns:
            A frequency-weighted next word, or None if no vocabulary exists.
        """
        if not self.unigram:
            return None

        word = word.lower()
        next_words = self.bigram.get(word)

        if not next_words:
            # Pure unigram sampling if no bigrams exist
            words, weights = zip(*self.unigram.items())
            return random.choices(words, weights=weights, k=1)[0]

        words, weights = zip(*next_words.items())
        return random.choices(words, weights=weights, k=1)[0]
