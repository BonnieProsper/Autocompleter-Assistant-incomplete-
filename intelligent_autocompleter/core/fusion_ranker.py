# changes: remove optional featurepreprocessor e.g always use preprocessor (why keep optional?)
# check: this version against previous, which is better (test)

# fusion_ranker.py
"""
FusionRanker - multi-signal ranker.

Design goals:
 - Per-feature normalization (via FeaturePreprocessor when available). - - remove?? e.g _minmax, log1pminmax, safesoftmax
 - Multiple fusion strategies (linear, softmax, borda).
 - Robust weight management (set, update with lr, clip, normalize).
 - Optional NumPy acceleration when present.
 - Lightweight personalizer hook (personalizer.bias_words(list[(w,score)]) -> list[(w,score)]).
 - Clean, testable public API.

 Recently added:
 - rank_normalized(normalized, weights, personalizer, topn) which accepts a normalized
   mapping word -> {feature: value}, avoids repeated per-call normalization conversions
 - rank() kept for backward compatibility and internally calls normalization helpers
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Optional, Sequence
from collections import defaultdict
import math
import logging

logger = logging.getLogger(__name__)

# optional numpy acceleration
try:
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

Candidate = Tuple[str, float]
CandidateList = List[Candidate]
FuzzyList = List[Tuple[str, int]]

# Default presets
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

    Main APIs:
      - rank(markov, embeddings, fuzzy, base_freq, recency_map, topn)
       (backwards-compatible and accepts raw maps or lists)
      - rank_normalized(normalized: Dict[word-> {markov, embed, fuzzy, freq, recency}], weights, personalizer, topn)
        (new and preferred API for performance)
    """

    def __init__(self, preset: str = "balanced", weights: Optional[Dict[str, float]] = None, personalizer: Optional[object] = None):
        base = dict(_PRESETS.get(preset, _PRESETS["balanced"]))
        if weights:
            base.update(weights)
        s = sum(base.values()) or 1.0
        self.weights: Dict[str, float] = {k: float(v) / s for k, v in base.items()}
        self.personalizer = personalizer

    # ---------------------------
    # API: accept normalized features
    # ---------------------------
    def rank_normalized(self,
                        normalized: Dict[str, Dict[str, float]],
                        weights: Optional[Dict[str, float]] = None,
                        personalizer: Optional[object] = None,
                        topn: int = 8) -> CandidateList:
        """
        Rank using a pre-normalized map:
           normalized = { word: {"markov": v, "embed": v, "fuzzy": v, "freq": v, "recency": v}, ...}
        This avoids repeated min-max/log transforms on the hot path.
        """
        if not normalized:
            return []

        # merge provided weights with defaults
        weight_map = dict(self.weights)
        if weights:
            weight_map.update(weights)
        personalizer = personalizer or self.personalizer

        scores: Dict[str, float] = {}
        for w, feats in normalized.items():
            s = 0.0
            s += weight_map.get("markov", 0.0) * float(feats.get("markov", 0.0))
            s += weight_map.get("embed", 0.0) * float(feats.get("embed", 0.0))
            s += weight_map.get("fuzzy", 0.0) * float(feats.get("fuzzy", 0.0))
            s += weight_map.get("freq", 0.0) * float(feats.get("freq", 0.0))
            s += weight_map.get("recency", 0.0) * float(feats.get("recency", 0.0))
            scores[w] = float(s)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])

        # personalizer hook (post-processing)
        if personalizer:
            try:
                ranked = personalizer.bias_words(ranked)
            except Exception as e:
                logger.debug("personalizer failed: %s", e)

        return [(w, round(float(sc), 6)) for w, sc in ranked[:topn]]

    # -----------------------
    # Backwards-compatible rank() API (keeps existing projects working)
    # -----------------------
    def _normalize_list_inputs(self, keys, raw_map: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalize raw_map according to keys."""
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

    def _log_and_norm(self, keys, raw_map: Dict[str, float]) -> Dict[str, float]:
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

    def _fuzzy_to_similarity(self, keys, fuzzy_map: Dict[str, int]) -> Dict[str, float]:
        raw = [(1.0 / (1.0 + float(fuzzy_map.get(k, 999)))) if k in fuzzy_map else 0.0 for k in keys]
        if NUMPY_AVAILABLE:
            arr = np.array(raw, dtype=float)
            lo, hi = float(arr.min()), float(arr.max())
            if hi == lo:
                return {k: 1.0 for k in keys}
            scaled = ((arr - lo) / (hi - lo)).tolist()
            return {k: scaled[i] for i, k in enumerate(keys)}
        else:
            return dict(zip(keys, _minmax_scale_py(raw)))

    def rank(self,
             markov: Optional[List[Tuple[str, float]]] = None,
             embeddings: Optional[List[Tuple[str, float]]] = None,
             fuzzy: Optional[FuzzyList] = None,
             base_freq: Optional[Dict[str, float]] = None,
             recency_map: Optional[Dict[str, float]] = None,
             topn: int = 8) -> CandidateList:
        """
        Backwards-compatible rank method. Internally builds a 'normalized' dict and calls rank_normalized.
        """
        markov = markov or []
        embeddings = embeddings or []
        fuzzy = fuzzy or []
        base_freq = base_freq or {}
        recency_map = recency_map or {}

        cand_set = set()
        cand_set.update([w for w, _ in markov])
        cand_set.update([w for w, _ in embeddings])
        cand_set.update([w for w, _ in fuzzy])
        cand_set.update(base_freq.keys())
        if not cand_set:
            return []

        keys = sorted(cand_set)

        m_map = {w: float(v) for w, v in markov}
        e_map = {w: float(v) for w, v in embeddings}
        f_map = {w: int(d) for w, d in fuzzy}
        freq_map = {w: float(v) for w, v in base_freq.items()}
        rec_map = {w: float(v) for w, v in recency_map.items()}

        nm = self._log_and_norm(keys, m_map)
        ne = self._normalize_list_inputs(keys, e_map)
        nf = self._fuzzy_to_similarity(keys, f_map)
        nfq = self._log_and_norm(keys, freq_map)

        rec_raw = []
        if rec_map:
            newest = max(rec_map.values())
        else:
            newest = None
        for k in keys:
            if newest is not None and k in rec_map:
                age = max(0.0, newest - rec_map[k])
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

        normalized = {}
        for i, k in enumerate(keys):
            normalized[k] = {
                "markov": nm[i],
                "embed": ne[i],
                "fuzzy": nf[i],
                "freq": nfq[i],
                "recency": rec_norm[k],
            }

        # call new API
        return self.rank_normalized(normalized, topn=topn)

        return [(w, round(float(s), 6)) for w, s in ranked[:topn]]
