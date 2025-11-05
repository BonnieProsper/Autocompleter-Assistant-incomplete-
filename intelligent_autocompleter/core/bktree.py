# bktree.py
# ---------------------------------------------------------------------
# BK-tree implementation for fuzzy (approximate) string matching.
# Used in Autocompleter Assistant to handle typos and similar word queries.
# ---------------------------------------------------------------------

from typing import List, Tuple


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr_row = [i]
        for j, cb in enumerate(b, 1):
            insert_cost = curr_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (ca != cb)
            curr_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = curr_row
    return prev_row[-1]


class BKTree:
    """A BK-tree for fast approximate string matching."""

    def __init__(self):
        self.root = None

    class Node:
        __slots__ = ("word", "children")

        def __init__(self, word):
            self.word = word
            self.children = {}

    def insert(self, word: str):
        """Insert a word into the BK-tree."""
        if not self.root:
            self.root = self.Node(word)
            return

        node = self.root
        while True:
            dist = levenshtein(word, node.word)
            if dist == 0:
                return
            if dist not in node.children:
                node.children[dist] = self.Node(word)
                break
            node = node.children[dist]

    def query(self, word: str, max_dist: int = 2) -> List[Tuple[str, int]]:
        """Find words within max_dist of the given query word."""
        results = []

        def search(node):
            dist = levenshtein(word, node.word)
            if dist <= max_dist:
                results.append((node.word, dist))
            for d in range(max(1, dist - max_dist), dist + max_dist + 1):
                child = node.children.get(d)
                if child:
                    search(child)

        if self.root:
            search(self.root)
        return sorted(results, key=lambda x: (x[1], x[0]))
