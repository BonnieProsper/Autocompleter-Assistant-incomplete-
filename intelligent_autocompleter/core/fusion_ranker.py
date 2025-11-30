# changes: remove optional featurepreprocessor e.g always use preprocessor (why keep optional?)
# check: this version against previous, which is better (test)

# intelligent_autocompleter/core/fusion_ranker.py
"""
FusionRanker - multi-signal ranker

Design goals:
 - Per-feature normalization (via FeaturePreprocessor when available). - - remove?? e.g _minmax, log1pminmax, safesoftmax
 - Multiple fusion strategies (linear, softmax, borda).
 - Robust weight management (set, update with lr, clip, normalize).
 - Lightweight personalizer hook (personalizer.bias_words(list[(w,score)]) -> list[(w,score)]).
 - Clean, testable public API.

Includes:
 - rank_normalized(): fast path that accepts per-feature normalized maps (word -> [0,1]).
 - rank(): compatibility wrapper that accepts raw/unnormalized inputs and normalizes them.
 - deterministic tie-breaking and stable ordering for unit tests.
 - optional NumPy acceleration for heavy normalization functions, but hot path (rank_normalized)
   avoids allocations by operating on dicts and weights.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Sequence, Mapping, Iterable
import math
import logging

logger = logging.getLogger(__name__)

# optional numpy
try:
    import numpy as np  # type: ignore
    NUMPY = True
except Exception:
    NUMPY = False

Candidate = Tuple[str, float]

# sensible presets
_PRESETS = {
    "strict":   {"markov": 0.6, "embed": 0.1, "fuzzy": 0.05, "freq": 0.2, "personal": 0.05, "recency": 0.0},
    "balanced": {"markov": 0.4, "embed": 0.3, "fuzzy": 0.1,  "freq": 0.15, "personal": 0.05, "recency": 0.0},
    "creative": {"markov": 0.2, "embed": 0.6, "fuzzy": 0.05, "freq": 0.05, "personal": 0.1, "recency": 0.0},
    "personal": {"markov": 0.25,"embed": 0.25,"fuzzy": 0.05, "freq": 0.1, "personal": 0.35, "recency": 0.0},
}

EPS = 1e-12


class FusionRanker:
    """
    FusionRanker merges multiple normalized signals into a final score.

    Usage:
        fr = FusionRanker(preset="balanced")
        ranked = fr.rank_normalized(markov_norm, embed_norm, fuzzy_distances_norm, freq_norm, recency_norm, topn)

    Backwards compatible:
        ranked = fr.rank(markov=[(w,c)...], embeddings=[(w,score)...], fuzzy=[(w,dist)...], base_freq={w:count}, recency_map={w:ts})
    """

    def __init__(self, preset: str = "balanced", weights: Optional[Dict[str, float]] = None, personalizer: Optional[object] = None):
        base = dict(_PRESETS.get(preset, _PRESETS["balanced"]))
        if weights:
            base.update(weights)
        s = sum(base.values()) or 1.0
        # normalized weights that sum to 1
        self.weights = {k: float(v) / s for k, v in base.items()}
        self.personalizer = personalizer

    # Fast path: operate on normalized signals (values in [0,1])
    # ------------------------------------------------------------
    def rank_normalized(self,
                        markov_norm: Mapping[str, float],
                        embed_norm: Mapping[str, float],
                        fuzzy_norm: Mapping[str, float],
                        freq_norm: Mapping[str, float],
                        recency_norm: Mapping[str, float],
                        topn: int = 8) -> List[Candidate]:
        """
        Accept normalized per feature dictionaries and return topn ranked candidates.
        Deterministic ordering: sort by (-score, token) to break ties lexicographically.
        """

        # union keys
        keys = set(markov_norm.keys()) | set(embed_norm.keys()) | set(fuzzy_norm.keys()) | set(freq_norm.keys()) | set(recency_norm.keys())
        if not keys:
            return []

        # compute weighted sum per key
        w_markov = self.weights.get("markov", 0.0)
        w_embed = self.weights.get("embed", 0.0)
        w_fuzzy = self.weights.get("fuzzy", 0.0)
        w_freq = self.weights.get("freq", 0.0)
        w_recency = self.weights.get("recency", 0.0)

        scores: List[Tuple[str, float]] = []
        append = scores.append
        for k in keys:
            s = 0.0
            s += w_markov * float(markov_norm.get(k, 0.0))
            s += w_embed * float(embed_norm.get(k, 0.0))
            s += w_fuzzy * float(fuzzy_norm.get(k, 0.0))
            s += w_freq * float(freq_norm.get(k, 0.0))
            s += w_recency * float(recency_norm.get(k, 0.0))
            append((k, float(s)))

        # deterministic sort by score desc, then token asc
        scores.sort(key=lambda kv: (-kv[1], kv[0]))

        # allow personalizer to postprocess (e.g boost words etc)
        ranked: List[Candidate] = scores
        if self.personalizer:
            try:
                ranked = self.personalizer.bias_words(ranked)
            except Exception as e:
                logger.debug("personalizer failed: %s", e)

        # return topn rounded
        out = [(w, round(float(sc), 6)) for w, sc in ranked[:topn]]
        return out

    # -------------------------
    # Backwards-compatible API
    # -------------------------
    def rank(self,
             markov: Optional[List[Tuple[str, float]]] = None,
             embeddings: Optional[List[Tuple[str, float]]] = None,
             fuzzy: Optional[List[Tuple[str, int]]] = None,
             base_freq: Optional[Mapping[str, float]] = None,
             recency_map: Optional[Mapping[str, float]] = None,
             topn: int = 8) -> List[Candidate]:
        """
        Compatibility wrapper that normalizes incoming raw signals then calls rank_normalized.

        Normalization strategies:
          - markov: log1p -> minmax
          - embeddings: minmax
          - fuzzy: distance -> similarity 1/(1+dist) -> minmax
          - base_freq: log1p -> minmax
          - recency_map: timestamp -> recency score 1/(1+log1p(age)) -> minmax
        """
        markov = markov or []
        embeddings = embeddings or []
        fuzzy = fuzzy or []
        base_freq = base_freq or {}
        recency_map = recency_map or {}

        # deterministic key order
        cand_set = set([w for w, _ in markov] + [w for w, _ in embeddings] + [w for w, _ in fuzzy] + list(base_freq.keys()) + list(recency_map.keys()))
        if not cand_set:
            return []

        keys = sorted(cand_set)

        # build arrays
        m_vals = [float(dict(markov).get(k, 0.0)) for k in keys]
        e_vals = [float(dict(embeddings).get(k, 0.0)) for k in keys]
        f_raw = [float(next((d for (w, d) in fuzzy if w == k), 999.0)) for k in keys]
        freq_vals = [float(base_freq.get(k, 0.0)) for k in keys]

        # recency transform (newer = larger)
        if recency_map:
            newest = max(recency_map.values())
            rec_vals = []
            for k in keys:
                if k in recency_map:
                    age = max(0.0, newest - float(recency_map[k]))
                    rec_vals.append(1.0 / (1.0 + math.log1p(age)))
                else:
                    rec_vals.append(0.0)
        else:
            rec_vals = [0.0] * len(keys)

        # normalization helpers (use numpy if available)
        def _minmax(vals: Sequence[float]) -> List[float]:
            arr = list(vals)
            if not arr:
                return []
            lo, hi = min(arr), max(arr)
            if abs(hi - lo) < EPS:
                return [1.0 for _ in arr]
            rng = hi - lo
            return [(v - lo) / rng for v in arr]

        def _log1p_minmax(vals: Sequence[float]) -> List[float]:
            transformed = [math.log1p(max(0.0, float(v))) for v in vals]
            return _minmax(transformed)

        m_norm = _log1p_minmax(m_vals)
        e_norm = _minmax(e_vals)
        f_sim = [1.0 / (1.0 + float(d)) if d is not None else 0.0 for d in f_raw]
        f_norm = _minmax(f_sim)
        freq_norm = _log1p_minmax(freq_vals)
        rec_norm = _minmax(rec_vals)

        # build maps and call fast path
        markov_map = {k: m_norm[i] for i, k in enumerate(keys)}
        embed_map = {k: e_norm[i] for i, k in enumerate(keys)}
        fuzzy_map = {k: int(f_raw[i]) if isinstance(f_raw[i], (int, float)) else 999 for i, k in enumerate(keys)}
        # convert fuzzy distances back to similarities normalized to [0,1]
        fuzzy_sim_map = {k: f_norm[i] for i, k in enumerate(keys)}
        freq_map = {k: freq_norm[i] for i, k in enumerate(keys)}
        recency_map_norm = {k: rec_norm[i] for i, k in enumerate(keys)}

        return self.rank_normalized(markov_map, embed_map, fuzzy_sim_map, freq_map, recency_map_norm, topn=topn)
