# intelligent_autocompleter/core/fusion_ranker.py
"""
FusionRanker - multi-signal ranker with optional NumPy acceleration.
Design goals:
 - Takes inputs from different signal sources (markov, embeddings, fuzzy, freq, recency).
 - Normalize each signal to [0,1] and compute weighted sum.
 - Support named presets and custom weight overrides.
 - Use NumPy for faster vector math when available, otherwise fall back to manual
 - Allow 'personalizer' object with 'bias_words(list[(w,score)]) -> list[(w,score)]'.
 - surface API for unit tests and tuning.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional, Sequence
from collections import defaultdict
import math
import logging

logger = logging.getLogger(__name__)

# try to import numpy for acceleration
try:
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

Candidate = Tuple[str, float]
CandidateList = List[Candidate]
FuzzyList = List[Tuple[str, int]]

# preset weight profiles (roughly tuned for demo)
_PRESETS = {
    "strict":   {"markov": 0.6, "embed": 0.1, "fuzzy": 0.05, "freq": 0.2, "personal": 0.05, "recency": 0.0},
    "balanced": {"markov": 0.4, "embed": 0.3, "fuzzy": 0.1,  "freq": 0.15, "personal": 0.05, "recency": 0.0},
    "creative": {"markov": 0.2, "embed": 0.6, "fuzzy": 0.05, "freq": 0.05, "personal": 0.1, "recency": 0.0},
    "personal": {"markov": 0.25,"embed": 0.25,"fuzzy": 0.05, "freq": 0.1, "personal": 0.35, "recency": 0.0},
}


def _minmax_scale_py(values: Sequence[float]) -> List[float]:
    vals = list(values)
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [1.0 for _ in vals]
    rng = hi - lo
    return [(v - lo) / rng for v in vals]


def _log1p_py(values: Sequence[float]) -> List[float]:
    return [math.log1p(max(0.0, float(v))) for v in values]


class FusionRanker:
    """
    Combine multiple signal maps into a ranked list of candidates.

    Usage:
        fr = FusionRanker(preset="balanced", personalizer=ctx)
        ranked = fr.rank(markov=[(w,c),...], embeddings=[(w,score)...], fuzzy=[(w,dist)...], base_freq={w:count}, recency_map={w:ts}, topn=8)
    """

    def __init__(self,
                 preset: str = "balanced",
                 weights: Optional[Dict[str, float]] = None,
                 personalizer: Optional[object] = None):
        base = dict(_PRESETS.get(preset, _PRESETS["balanced"]))
        if weights:
            base.update(weights)
        s = sum(base.values()) or 1.0
        # Normalize so weights sum to 1. Keeps semantics stable.
        self.weights: Dict[str, float] = {k: float(v) / s for k, v in base.items()}
        self.personalizer = personalizer

    # internal helpers (numpy and python implementations) --------------------------
    def _normalize_signals(self, keys: List[str], raw_map: Dict[str, float]) -> Dict[str, float]:
        """
        Given a candidate order 'keys' and raw_map mapping key->value, return normalized dict key->[0,1].
        Uses NumPy when available for speed.
        """
        vals = [float(raw_map.get(k, 0.0)) for k in keys]
        if NUMPY_AVAILABLE:
            arr = np.array(vals, dtype=float)
            lo = float(arr.min()) if arr.size else 0.0
            hi = float(arr.max()) if arr.size else 0.0
            if hi == lo:
                return {k: 1.0 for k in keys}
            scaled = ((arr - lo) / (hi - lo)).tolist()
            return {k: scaled[i] for i, k in enumerate(keys)}
        else:
            return dict(zip(keys, _minmax_scale_py(vals)))

    def _log_and_normalize(self, keys: List[str], raw_map: Dict[str, float]) -> Dict[str, float]:
        """Apply log1p transform then min-max normalize."""
        vals = [max(0.0, float(raw_map.get(k, 0.0))) for k in keys]
        if NUMPY_AVAILABLE:
            arr = np.log1p(np.array(vals, dtype=float))
            lo, hi = float(arr.min()), float(arr.max())
            if hi == lo:
                return {k: 1.0 for k in keys}
            scaled = ((arr - lo) / (hi - lo)).tolist()
            return {k: scaled[i] for i, k in enumerate(keys)}
        else:
            transformed = _log1p_py(vals)
            return dict(zip(keys, _minmax_scale_py(transformed)))

    def _fuzzy_to_similarity(self, keys: List[str], fuzzy_map: Dict[str, int]) -> Dict[str, float]:
        """
        Convert fuzzy distances (Levenshtein-like) into similarity scores (higher = better),
        then min-max normalize.
        If a key is missing in fuzzy_map it's treated as 0 similarity.
        """
        # raw: similarity = 1/(1+dist)
        raw = [ (1.0 / (1.0 + float(fuzzy_map.get(k, 999)))) if k in fuzzy_map else 0.0 for k in keys ]
        if NUMPY_AVAILABLE:
            arr = np.array(raw, dtype=float)
            lo, hi = float(arr.min()), float(arr.max())
            if hi == lo:
                return {k: 1.0 for k in keys}
            scaled = ((arr - lo) / (hi - lo)).tolist()
            return {k: scaled[i] for i, k in enumerate(keys)}
        else:
            return dict(zip(keys, _minmax_scale_py(raw)))

    # public API -----------------------------------------------
    def rank(self,
             markov: Optional[List[Tuple[str, float]]] = None,
             embeddings: Optional[List[Tuple[str, float]]] = None,
             fuzzy: Optional[FuzzyList] = None,
             base_freq: Optional[Dict[str, float]] = None,
             recency_map: Optional[Dict[str, float]] = None,
             topn: int = 8) -> CandidateList:
        """
        Rank candidates using available signals. All inputs are optional.
        - markov: list[(word, count)] or similar
        - embeddings: list[(word, score)]
        - fuzzy: list[(word, distance)]
        - base_freq: dict[word->count]
        - recency_map: dict[word->timestamp]
        Returns topn list[(word, score)] sorted descending by score.
        """
        markov = markov or []
        embeddings = embeddings or []
        fuzzy = fuzzy or []
        base_freq = base_freq or {}
        recency_map = recency_map or {}

        # collect union of candidate words, keep deterministic order
        cand_set = set()
        cand_set.update([w for w, _ in markov])
        cand_set.update([w for w, _ in embeddings])
        cand_set.update([w for w, _ in fuzzy])
        cand_set.update(base_freq.keys())
        if not cand_set:
            return []

        # deterministic list order (stable across runs)
        keys = sorted(cand_set)

        # maps for quick lookup
        m_map = {w: float(v) for w, v in markov}
        e_map = {w: float(v) for w, v in embeddings}
        f_map = {w: int(d) for w, d in fuzzy}
        freq_map = {w: float(v) for w, v in base_freq}
        rec_map = {w: float(v) for w, v in recency_map}

        # normalize signals
        m_norm = self._log_and_normalize(keys, m_map)           # markov -> log1p -> minmax
        e_norm = self._normalize_signals(keys, e_map)           # embeddings -> minmax
        f_norm = self._fuzzy_to_similarity(keys, f_map)         # fuzzy -> invert dist -> minmax
        freq_norm = self._log_and_normalize(keys, freq_map)     # freq -> log1p -> minmax

        # recency: newer is higher, invert age and minmax
        rec_raw = []
        if rec_map:
            newest = max(rec_map.values())
        else:
            newest = None
        for k in keys:
            if newest is not None and k in rec_map:
                age = max(0.0, newest - rec_map[k])
                # use 1 / (1 + log1p(age)) to avoid dominating tiny differences
                rec_raw.append(1.0 / (1.0 + math.log1p(age)))
            else:
                rec_raw.append(0.0)
        if NUMPY_AVAILABLE:
            arr = np.array(rec_raw, dtype=float)
            lo, hi = float(arr.min()), float(arr.max())
            if hi == lo:
                rec_norm = {k: 1.0 for k in keys}
            else:
                scaled = ((arr - lo) / (hi - lo)).tolist()
                rec_norm = {k: scaled[i] for i, k in enumerate(keys)}
        else:
            rec_norm = dict(zip(keys, _minmax_scale_py(rec_raw)))

        # combine weighted sum
        scores: Dict[str, float] = {}
        for k in keys:
            s = 0.0
            s += self.weights.get("markov", 0.0) * m_norm.get(k, 0.0)
            s += self.weights.get("embed", 0.0)  * e_norm.get(k, 0.0)
            s += self.weights.get("fuzzy", 0.0)  * f_norm.get(k, 0.0)
            s += self.weights.get("freq", 0.0)   * freq_norm.get(k, 0.0)
            s += self.weights.get("recency", 0.0) * rec_norm.get(k, 0.0)
            # personal weight is intentionally left for the personalizer to apply (post-processing)
            scores[k] = float(s)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])

        # allow personalizer (CtxPersonal) to bump/resplice results
        if self.personalizer:
            try:
                # personalizer expected to accept list[(w,score)] and return same
                ranked = self.personalizer.bias_words(ranked)
            except Exception as e:
                logger.debug("personalizer failed during ranking: %s", e)

        # return topn rounded scores
        return [(w, round(float(sc), 6)) for w, sc in ranked[:topn]]
