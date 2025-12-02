# intelligent_autocompleter/core/feature_preprocessor.py
"""
FeaturePreprocessor - robust, testable normalization utilities for autocompleter fusion pipeline.

Responsibilities
- Validate input signal shapes (word -> numeric)
- Provide per-feature transforms:
    min-max scaling
    log1p + min-max scaling (for heavy-tailed counts)
    fuzzy distance -> similarity (1 / (1 + dist)) + min-max
    recency (timestamp) -> recency score + min-max
- Produce stable and deterministic outputs:
    normalize_all(...) -> Dict[word, Dict[feature->normalized_score]]
    build_feature_matrix(...) -> (words, list_of_feature_names, 2D-list-or-np-array)
- Optional NumPy acceleration via `use_numpy=True` (auto-falls back if numpy not available).
- Shows errors for invalid inputs.

Keeps numeric logic separated from FusionRanker so:
- preprocessing logic is centralised and testable
- FusionRanker only consumes normalized features
- responsibilities are clean (SRP)

Example:
    pre = FeaturePreprocessor(use_numpy=False)
    features = pre.normalize_all(
        markov={'a': 10, 'b': 1},
        embed={'a': 0.9, 'b': 0.1},
        fuzzy={'a': 1, 'b': 3},
        freq={'a': 100, 'b': 2},
        recency={'a': 1600000000, 'b': 1590000000},
    )
    # features -> {'a': {'markov':..., 'embed':..., ...}, 'b': {...}}
"""

from __future__ import annotations

from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import math
import logging

logger = logging.getLogger(__name__)

# Optional numpy acceleration 
try:
    import numpy as np  # type: ignore
    _NUMPY = True
except Exception:
    _NUMPY = False

# Types
Numeric = float
ScoreMap = Dict[str, Numeric]  # word -> raw score
FeatureVector = Dict[str, Numeric]  # normalized features for a single word
FeatureMatrix = Dict[str, FeatureVector]  # word -> features

DEFAULT_FEATURE_ORDER = ("markov", "embed", "fuzzy", "freq", "recency")

class PreprocessorError(ValueError):
    """Raised when inputs are invalid or inconsistent."""


def _safe_minmax_python(vals: Sequence[float]) -> List[float]:
    vals = list(vals)
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if hi == lo:
        # constant vector -> give uniform 1.0 (keeps signal non-zero but equal)
        return [1.0 for _ in vals]
    rng = hi - lo
    return [(float(v) - lo) / rng for v in vals]


def _safe_log1p_minmax_python(vals: Sequence[float]) -> List[float]:
    # apply log1p then minmax
    transformed = [math.log1p(max(0.0, float(v))) for v in vals]
    return _safe_minmax_python(transformed)


def _fuzzy_distances_to_similarity_python(distances: Sequence[float]) -> List[float]:
    # convert distances to similarities (1 / (1 + dist)) then minmax
    sims = [1.0 / (1.0 + float(d)) if d is not None else 0.0 for d in distances]
    return _safe_minmax_python(sims)


class FeaturePreprocessor:
    """
    Validate and normalize signals for FusionRanker.

    Args:
    required_features: optional iterable of feature names that must appear in output vectors.
                       Default set: ('markov','embed','fuzzy','freq','recency')
    use_numpy: if True, attempt to use numpy for vectorized operations (falls back if numpy
               is not importable).
    """

    def __init__(self, required_features: Optional[Iterable[str]] = None, use_numpy: bool = True):
        self.feature_order = tuple(required_features) if required_features else DEFAULT_FEATURE_ORDER
        self.use_numpy = bool(use_numpy) and _NUMPY

    # Transformations
    def _minmax(self, vals: Sequence[Numeric]) -> List[float]:
        if self.use_numpy:
            arr = np.array(vals, dtype=float)
            if arr.size == 0:
                return []
            lo = float(arr.min())
            hi = float(arr.max())
            if hi == lo:
                return [1.0 for _ in arr.tolist()]
            return ((arr - lo) / (hi - lo)).tolist()
        else:
            return _safe_minmax_python(vals)

    def _log1p_minmax(self, vals: Sequence[Numeric]) -> List[float]:
        if self.use_numpy:
            arr = np.array(vals, dtype=float)
            if arr.size == 0:
                return []
            arr = np.log1p(np.clip(arr, a_min=0.0, a_max=None))
            lo = float(arr.min())
            hi = float(arr.max())
            if hi == lo:
                return [1.0 for _ in arr.tolist()]
            return ((arr - lo) / (hi - lo)).tolist()
        else:
            return _safe_log1p_minmax_python(vals)

    def _fuzzy_to_sim(self, distances: Sequence[Numeric]) -> List[float]:
        if self.use_numpy:
            arr = np.array([float(d) if d is not None else 999.0 for d in distances], dtype=float)
            # avoid division by zero, convert distances to similarity
            sims = 1.0 / (1.0 + arr)
            lo = float(sims.min())
            hi = float(sims.max())
            if hi == lo:
                return [1.0 for _ in sims.tolist()]
            return ((sims - lo) / (hi - lo)).tolist()
        else:
            return _fuzzy_distances_to_similarity_python(distances)

    def _recency_transform(self, timestamps: Sequence[Numeric]) -> List[float]:
        """
        Converts timestamps to recency score where newer => larger value.
        Transforms ages to 1 / (1 + log1p(age)) to compress large ages then min-max.
        Timestamps are expected as POSIX seconds (ints/floats).
        """
        if not timestamps:
            return []
        # convert to ages
        arr = list(float(t) for t in timestamps)
        newest = max(arr)
        ages = [max(0.0, newest - a) for a in arr]
        rec_raw = [1.0 / (1.0 + math.log1p(age)) if age >= 0.0 else 0.0 for age in ages]
        return self._minmax(rec_raw)

    # Helpers -------------------------------------------------
    def _collect_keys(self, *maps: Mapping[str, Numeric]) -> List[str]:
        # union of keys in deterministic order
        keys = set()
        for m in maps:
            if not isinstance(m, Mapping):
                raise PreprocessorError("Each signal must be a mapping (word -> numeric).")
            keys.update(m.keys())
        return sorted(keys)

    # Primary API ----------------------------------------
    def build_feature_matrix(
        self,
        markov: Optional[Mapping[str, Numeric]] = None,
        embed: Optional[Mapping[str, Numeric]] = None,
        fuzzy: Optional[Mapping[str, Numeric]] = None,
        freq: Optional[Mapping[str, Numeric]] = None,
        recency: Optional[Mapping[str, Numeric]] = None,
    ) -> Tuple[List[str], List[str], List[List[float]]]:
        """
        Produce aligned arrays for downstream vectorized processing.

        Returns:
            (words, feature_names, matrix) where:
                - words: deterministic sorted list of candidate words
                - feature_names: same order as self.feature_order
                - matrix: list of rows, each row is a list of normalized floats ordered by feature_names
        """
        markov = dict(markov or {})
        embed = dict(embed or {})
        fuzzy = dict(fuzzy or {})
        freq = dict(freq or {})
        recency = dict(recency or {})

        keys = self._collect_keys(markov, embed, fuzzy, freq, recency)
        if not keys:
            return [], list(self.feature_order), []

        # raw arrays aligned with keys
        m_vals = [float(markov.get(k, 0.0)) for k in keys]
        e_vals = [float(embed.get(k, 0.0)) for k in keys]
        # for fuzzy, treat missing as large distance (we'll convert to low similarity)
        f_vals = [float(fuzzy.get(k, 999.0)) if k in fuzzy else 999.0 for k in keys]
        freq_vals = [float(freq.get(k, 0.0)) for k in keys]
        if recency:
            rec_ts = [float(recency.get(k, 0.0)) if k in recency else 0.0 for k in keys]
        else:
            rec_ts = [0.0 for _ in keys]

        # normalize per-feature
        nm = self._log1p_minmax(m_vals)
        ne = self._minmax(e_vals)
        nf = self._fuzzy_to_sim(f_vals)
        nfq = self._log1p_minmax(freq_vals)
        nr = self._recency_transform(rec_ts)

        # assemble feature rows in consistent feature order
        matrix: List[List[float]] = []
        for i, _k in enumerate(keys):
            row = [
                nm[i],
                ne[i],
                nf[i],
                nfq[i],
                nr[i],
            ]
            matrix.append(row)

        return keys, list(self.feature_order), matrix

    def normalize_all(
        self,
        markov: Optional[Mapping[str, Numeric]] = None,
        embed: Optional[Mapping[str, Numeric]] = None,
        fuzzy: Optional[Mapping[str, Numeric]] = None,
        freq: Optional[Mapping[str, Numeric]] = None,
        recency: Optional[Mapping[str, Numeric]] = None,
    ) -> FeatureMatrix:
        """
        Convenience: produce a dict[word -> {feature: normalized_score}].

        This is what FusionRanker expects if it wants per-candidate
        dictionaries rather than vectorized arrays.
        """
        keys, feature_names, matrix = self.build_feature_matrix(markov, embed, fuzzy, freq, recency)
        out: FeatureMatrix = {}
        for word, row in zip(keys, matrix):
            out[word] = {fname: float(val) for fname, val in zip(feature_names, row)}
        # If required_features specified, ensure each output vector contains them (fill 0.0 if missing)
        if self.feature_order:
            for w in out:
                for f in self.feature_order:
                    out[w].setdefault(f, 0.0)
        return out


# Smoke test - extend/test further
if __name__ == "__main__":  # simple sanity checks
    import time as _t

    pre = FeaturePreprocessor(use_numpy=False)
    sample = pre.normalize_all(
        markov={"a": 10, "b": 2, "c": 0},
        embed={"a": 0.9, "b": 0.25, "d": -0.2},
        fuzzy={"a": 1, "b": 4, "d": 2},
        freq={"a": 1000, "b": 2, "c": 1},
        recency={"a": _t.time(), "b": _t.time() - 3600, "d": _t.time() - 86400},
    )
    print("SAMPLE:", sample)
    # vectorized form
    words, feats, mat = pre.build_feature_matrix(
        markov={"a": 10, "b": 2, "c": 0},
        embed={"a": 0.9, "b": 0.25, "d": -0.2},
        fuzzy={"a": 1, "b": 4, "d": 2},
        freq={"a": 1000, "b": 2, "c": 1},
        recency={"a": _t.time(), "b": _t.time() - 3600, "d": _t.time() - 86400},
    )
    print("WORDS:", words)
    print("FEATS:", feats)
    print("MATRIX SAMPLE (first row):", mat[0] if mat else None)


"""Include?:
# intelligent_autocompleter/core/feature_preprocessor.py
# comment
FeaturePreprocessor - normalize and prepare feature maps for ranking.

API:
 - normalize_all(markov, embed, fuzzy, freq, recency) -> Dict[word -> Dict[feature->float]]
# end comment
from __future__ import annotations
from typing import Dict, Iterable
import math

FEATURES = ("markov", "embed", "personal", "freq", "fuzzy", "recency")


class FeaturePreprocessor:
    def __init__(self):
        # placeholder for future config (clip thresholds, scaling method)
        self.clip_min = 0.0
        self.clip_max = 1.0

    def normalize_all(self,
                      markov: Dict[str, float],
                      embed: Dict[str, float],
                      fuzzy: Dict[str, int],
                      freq: Dict[str, float],
                      recency: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        # Build union of words
        words = set(markov.keys()) | set(embed.keys()) | set(freq.keys()) | set(fuzzy.keys()) | set(recency.keys())
        if not words:
            return {}

        # Convert fuzzy distances to a score (0..1): smaller distance -> larger score
        fuzzy_scores = {}
        for w, d in fuzzy.items():
            try:
                d = float(d)
                # guard: distance 0 -> highest
                fuzzy_scores[w] = 1.0 / (1.0 + d)
            except Exception:
                fuzzy_scores[w] = 0.0

        # Prepare raw feature map
        raw = {}
        for w in words:
            raw[w] = {
                "markov": float(markov.get(w, 0.0)),
                "embed": float(embed.get(w, 0.0)),
                "personal": float(freq.get(w, 0.0)),  # personal uses freq map passed by caller (CtxPersonal)
                "freq": float(freq.get(w, 0.0)),
                "fuzzy": float(fuzzy_scores.get(w, 0.0)),
                "recency": float(recency.get(w, 0.0))
            }

        # Min-max normalization per feature
        mins = {f: float("inf") for f in FEATURES}
        maxs = {f: float("-inf") for f in FEATURES}
        for w, feats in raw.items():
            for f in FEATURES:
                v = feats.get(f, 0.0)
                mins[f] = min(mins[f], v)
                maxs[f] = max(maxs[f], v)

        normalized = {}
        for w, feats in raw.items():
            normalized[w] = {}
            for f in FEATURES:
                lo, hi = mins[f], maxs[f]
                v = float(feats.get(f, 0.0))
                if hi - lo <= 1e-12:
                    # degenerate range: map to 0 or scaled by hi
                    normalized[w][f] = 0.0 if hi == 0.0 else v / hi
                else:
                    normalized[w][f] = (v - lo) / (hi - lo)
                # clip for numerical safety
                normalized[w][f] = max(self.clip_min, min(self.clip_max, normalized[w][f]))
        return normalized


"""
