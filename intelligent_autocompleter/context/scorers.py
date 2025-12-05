# intelligent_autocompleter/context/scorers.py
# small heuristics used to rank and flag next-step behaviour

import math
from collections import Counter


def score_uncertainty(tokens):
    """
    Measures uncertainty using partial tokens/trailing punctuation
    returns 0..1 where 1 means high uncertainty (user likely needs help)
    """
    if not tokens:
        return 0.0
    last = tokens[-1]
    if len(last) < 3:
        # short token is ambiguous
        return 0.6
    if last.endswith((".", "?", "!")):
        return 0.2
    # crude logic, if token ends with non-space fragment its more uncertain
    return 0.1


def score_repetition(tokens):
    """If user repeats same token a lot, returns a higher score."""
    if not tokens:
        return 0.0
    c = Counter(tokens)
    most = c.most_common(1)[0][1]
    # normalized by token count
    return min(1.0, most / max(1, len(tokens)))
