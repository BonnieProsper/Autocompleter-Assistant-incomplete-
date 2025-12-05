# trie.py
# Simple Trie (prefix tree) for prefix-based autocompletion.
# Keeps word frequencies for lightweight ranking.
# fast enough for approx 100k vocab and easy to reason about.

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple

Word = str
Score = int
Candidate = Tuple[Word, Score]


class TrieNode:
    """
    A singlenode in the Trie.
    children: char -> TrieNode
    is_word: bool marker to know if this path forms a real word
    freq: counts how often user/system inserted this word
    """

    __slots__ = ("children", "is_word", "freq")

    def __init__(self) -> None:
        self.children: Dict[str, TrieNode] = defaultdict(TrieNode)
        self.is_word = False
        self.freq = 0


class Trie:
    """
    Trie to store words for fast prefix lookup, used by the Autocompleter for:
     - prefix-based suggestions
     - fast fallback when ML models fail
     - collecting popularity-weighted completions
    """

    def __init__(self) -> None:
        self._root = TrieNode()

    # insertion -----------------------------------------------------
    def insert(self, word: str) -> None:
        """
        Insert a word into the trie.
        Lowercases everything (simple ASCII normalization).
        Keeps a frequency count for ranking.
        """
        if not word:
            return

        node = self._root
        for ch in word.lower():
            node = node.children[ch]
        node.is_word = True
        node.freq += 1

    # search/traversal ---------------------------------------------------------
    def search_prefix(self, prefix: str, limit: int = 50) -> List[Candidate]:
        """
        Return all words starting with `prefix`.
        Very small limit included to avoid accidental massive dumps.
        Returned: list[(word, freq)] sorted by
         - higher freq first
         - lexicographically second
        """
        if not prefix:
            return []

        node = self._root
        for ch in prefix.lower():
            nxt = node.children.get(ch)
            if nxt is None:
                return []
            node = nxt

        out: List[Candidate] = []
        self._collect(node, prefix.lower(), out, limit)
        out.sort(key=lambda t: (-t[1], t[0]))
        return out

    # internal recursive collector ---------------------------------------------------------
    def _collect(
        self, node: TrieNode, prefix: str, results: List[Candidate], limit: int
    ) -> None:
        """DFS collecting words under a prefix node."""
        if node.is_word:
            results.append((prefix, node.freq))
            if len(results) >= limit:
                return
        for ch, child in node.children.items():
            if len(results) >= limit:
                break
            self._collect(child, prefix + ch, results, limit)

    # convenience/debugging -----------------------------------------------------
    def size(self) -> int:
        """
        Count words in the Trie.
        (Slow: O(N) walk. For inspection, not runtime.)
        """
        acc: List[Candidate] = []
        self._collect(self._root, "", acc, limit=10_000_000)
        return len(acc)

    def __contains__(self, word: str) -> bool:
        """Simple membership check."""
        node = self._root
        for ch in word.lower():
            node = node.children.get(ch)
            if node is None:
                return False
        return node.is_word
