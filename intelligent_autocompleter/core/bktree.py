# bktree.py
# BK-tree for approximate/fuzzy string matching (typo-tolerant lookup).
# Designed for use in an autocompleter: insert words, then query for
# close matches within a given edit distance.
# - Node keeps a simple frequency counter (useful for ranking/popularity).
# - Levenshtein implements an early-exit cutoff to speed up searches.
# Query uses an explicit stack (no recursion) and prunes using triangle property of edit distance.

from typing import List, Tuple, Optional, Iterable
from collections import deque


def _normalize(s: str) -> str:
    """Simple normalizer used across the tree (lowercase + trim)."""
    return s.strip().lower()


def levenshtein_with_cutoff(a: str, b: str, max_dist: Optional[int] = None) -> int:
    """
    Compute Levenshtein distance with optional early exit when distance
    exceeds max_dist. This is faster for fuzzy search where we only care
    about nearby matches.
    Returns the computed distance (may be greater than max_dist if cutoff not used).
    """
    if a == b:
        return 0

    # ensure a is the longer string to simplify indexing
    if len(a) < len(b):
        a, b = b, a

    la, lb = len(a), len(b)

    # bounding: if length difference > max_dist, we can bail early
    if max_dist is not None and abs(la - lb) > max_dist:
        return max_dist + 1

    # initialize previous row
    prev = list(range(lb + 1))

    # iterate rows
    for i in range(1, la + 1):
        ca = a[i - 1]
        curr = [i]
        # keep a local best to check early-exit
        row_min = curr[0]

        for j in range(1, lb + 1):
            cb = b[j - 1]
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (0 if ca == cb else 1)
            val = ins if ins < delete else delete
            if replace < val:
                val = replace
            curr.append(val)
            if val < row_min:
                row_min = val

        # early exit: if row_min > max_dist then stop
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        prev = curr
    return prev[-1]


class BKTree:
    """BK-tree for approximate string lookup."""

    class Node:
        __slots__ = ("word", "children", "count")

        def __init__(self, word: str):
            self.word = word
            self.children: dict[int, "BKTree.Node"] = {}
            self.count = 1  # frequency (useful when ranking)

    def __init__(self):
        self.root: Optional[BKTree.Node] = None
        self._size = 0

    # insertion/building -------------------------------------------------------------
    def insert(self, word: str) -> None:
        """Inserts a single word into the BK-tree and normalizes input."""
        if not word or not isinstance(word, str):
            return

        w = _normalize(word)
        if not w:
            return

        if self.root is None:
            self.root = BKTree.Node(w)
            self._size = 1
            return

        node = self.root
        while True:
            d = levenshtein_with_cutoff(w, node.word)
            if d == 0:
                node.count += 1
                return
            child = node.children.get(d)
            if child is None:
                node.children[d] = BKTree.Node(w)
                self._size += 1
                return
            node = child

    def insert_many(self, words: Iterable[str]) -> None:
        """Convenience bulk insert (faster than repeated single inserts in practice)."""
        for w in words:
            self.insert(w)

    # query ---------------------------------------------------------------------------
    def query(self, word: str, max_dist: int = 2) -> List[Tuple[str, int]]:
        """
        Return list of (word, distance) for words within max_dist of 'word'.
        Result is sorted by (distance, -frequency, word) for stable top-k selection.
        """
        if not word or self.root is None:
            return []

        q = _normalize(word)
        if not q:
            return []

        results: List[Tuple[str, int, int]] = []  # (word, dist, count)

        # iterative stack for traversal: (node)
        stack = [self.root]
        while stack:
            node = stack.pop()
            # compute distance with cutoff (only need to know if distance <= max_dist)
            d = levenshtein_with_cutoff(q, node.word, max_dist)
            if d <= max_dist:
                results.append((node.word, d, node.count))

            # children distances to consider: [d - max_dist, d + max_dist]
            low = max(1, d - max_dist)
            high = d + max_dist
            for dist_key, child in node.children.items():
                if low <= dist_key <= high:
                    stack.append(child)

        # sorting order: prefer smaller distance, then higher frequency, then lexicographic
        results.sort(key=lambda item: (item[1], -item[2], item[0]))
        return [(w, dist) for (w, dist, _count) in results]

    # utilities -------------------------------------------------------------------
    def __contains__(self, word: str) -> bool:
        """True if exact word exists in the tree (case/space normalized)."""
        if not self.root:
            return False
        target = _normalize(word)
        # quick lookup by traversing distances (may be O(n) worst-case, fine for small vocab)
        node = self.root
        stack = [node]
        while stack:
            n = stack.pop()
            if n.word == target:
                return True
            for child in n.children.values():
                stack.append(child)
        return False

    def size(self) -> int:
        return self._size

    def words(self) -> List[Tuple[str, int]]:
        """Return all words stored with their frequencies, unsorted."""
        out: List[Tuple[str, int]] = []
        if not self.root:
            return out
        stack = [self.root]
        while stack:
            n = stack.pop()
            out.append((n.word, n.count))
            for child in n.children.values():
                stack.append(child)
        return out

    def clear(self) -> None:
        """Remove all nodes."""
        self.root = None
        self._size = 0
