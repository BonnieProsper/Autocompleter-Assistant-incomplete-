# trie.py
# ---------------------------------------------------------------------
# Trie implementation for Autocompleter Assistant.
# Prefix-based lookup and weighted suggestion handling.
# ---------------------------------------------------------------------

from collections import defaultdict


class TrieNode:
    """A single node in the Trie, storing children and word frequency."""

    __slots__ = ("children", "is_word", "frequency")

    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_word = False
        self.frequency = 0


class Trie:
    """Prefix-based word storage for fast autocomplete."""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        """Insert a word into the Trie."""
        node = self.root
        for char in word.lower():
            node = node.children[char]
        node.is_word = True
        node.frequency += 1  # Tracks popularity

    def _collect(self, node, prefix, results):
        """Recursive helper to gather words from a given Trie node."""
        if node.is_word:
            results.append((prefix, node.frequency))
        for char, child in node.children.items():
            self._collect(child, prefix + char, results)

    def search_prefix(self, prefix: str):
        """Return all words that start with the given prefix."""
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]

        results = []
        self._collect(node, prefix.lower(), results)
        results.sort(key=lambda x: (-x[1], x[0]))  # Sort by frequency, then alphabetically
        return results
