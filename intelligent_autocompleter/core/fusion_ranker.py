# changes: remove optional featurepreprocessor e.g always use preprocessor (why keep optional?)
# check: this version against previous, which is better (test)

# intelligent_autocompleter/core/fusion_ranker.py
"""
FusionRanker - multi-signal ranker with fast normalized hot path

Design goals:
 - Per-feature normalization (via FeaturePreprocessor when available). - - remove?? e.g _minmax, log1pminmax, safesoftmax
 - Multiple fusion strategies (linear, softmax, borda).
 - Robust weight management (set, update with lr, clip, normalize).
 - Lightweight personalizer hook (personalizer.bias_words(list[(w,score)]) -> list[(w,score)]).
 - Clean, testable public API.

Includes:
 - rank_normalized(): fast path that accepts per-feature normalized maps (word -> [0,1]), 
   preferred hot-path: accepts normalized per-word feature dicts
 - rank(): compatibility wrapper that accepts raw/unnormalized inputs and normalizes them, 
   backward compatible list based API
 - deterministic tie-breaking and stable ordering for unit tests.
 - optional NumPy acceleration for heavy normalization functions, but hot path (rank_normalized)
   avoids allocations by operating on dicts and weights. Uses NumPy if present for vectorized computation on large candidate sets.
 - FusionRanker.debug_contributions(word, normalized_map, weights) -> per-feature contributions

Design notes:
 - Normalization responsibilities can be delegated to FeaturePreprocessor.
 - Uses NumPy if present for vectorized computation on large candidate sets.
 - Keeps numeric stability and deterministic ordering (sorted keys) for reproducibility.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Optional, Protocol
from collections import defaultdict
import math
import logging

logger = logging.getLogger(__name__)

# Optional numpy acceleration
try:
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

# Types
Candidate = Tuple[str, float]
CandidateList = List[Candidate]
FuzzyList = List[Tuple[str, int]]
FeatureMap = Dict[str, float]      # word -> normalized value
NormalizedPerWord = Dict[str, Dict[str, float]]  # word -> {feature: normalized_value}
Weights = Dict[str, float]         # feature name -> weight (not necessarily normalized)


# Protocols (for static typing of pluggable components)
class PersonalizerProtocol(Protocol):
    def bias_words(self, ranked: CandidateList) -> CandidateList:
        ...


# Preset weight profiles
_PRESETS: Dict[str, Weights] = {
    "strict":   {"markov": 0.6, "embed": 0.1, "fuzzy": 0.05, "freq": 0.2, "personal": 0.05, "recency": 0.0},
    "balanced": {"markov": 0.4, "embed": 0.3, "fuzzy": 0.1,  "freq": 0.15, "personal": 0.05, "recency": 0.0},
    "creative": {"markov": 0.2, "embed": 0.6, "fuzzy": 0.05, "freq": 0.05, "personal": 0.1, "recency": 0.0},
    "personal": {"markov": 0.25,"embed": 0.25,"fuzzy": 0.05, "freq": 0.1, "personal": 0.35, "recency": 0.0},
}


def _safe_normalize_weights(weights: Optional[Weights], preset: str = "balanced") -> Weights:
    base = dict(_PRESETS.get(preset, _PRESETS["balanced"]))
    if weights:
        base.update(weights)
    s = sum(base.values()) or 1.0
    return {k: float(v) / s for k, v in base.items()}


class FusionRanker:
    """
    Combine multiple normalized features per-candidate into a single ranking.

    Two recommended entrypoints:
      - rank_normalized(normalized_map, weights=, personalizer=, topn=)
      - rank(markov=[(w,c)], embeddings=[(w,score)], fuzzy=[(w,dist)], base_freq={}, recency_map={}, topn=)

    The normalized API avoids re-normalizing on the hot-path and is preferred for performance.
    """

    def __init__(self,
                 preset: str = "balanced",
                 weights: Optional[Weights] = None,
                 personalizer: Optional[PersonalizerProtocol] = None):
        self.weights = _safe_normalize_weights(weights or None, preset)
        self.personalizer = personalizer

    # ---------------------------------
    # Hot-path: rank normalized maps
    # ----------------------------------
    def rank_normalized(self,
                        normalized: NormalizedPerWord,
                        weights: Optional[Weights] = None,
                        personalizer: Optional[PersonalizerProtocol] = None,
                        topn: int = 8) -> CandidateList:
        """
        Rank using pre-normalized feature values.

        normalized: dict[word] -> dict[feature_name] -> normalized_value in [0,1]
        weights: optional weights mapping (feature -> weight). If omitted uses self.weights.
        personalizer: optional object to post-process ranked list.
        """
        if not normalized:
            return []

        weights = _safe_normalize_weights(weights or self.weights, "balanced")
        personalizer = personalizer or self.personalizer

        # Deterministic sorted order of candidates
        words = sorted(normalized.keys())

        # Extract per-feature arrays in deterministic ordering
        # Determine feature set of interest from weights keys
        feature_keys = list(weights.keys())

        # Build feature arrays
        if NUMPY_AVAILABLE and len(words) > 16:
            # Vectorized path
            mat = []
            for feat in feature_keys:
                arr = [float(normalized[w].get(feat, 0.0)) for w in words]
                mat.append(arr)
            A = np.array(mat, dtype=float)  # shape: (n_features, n_words)
            w = np.array([weights.get(f, 0.0) for f in feature_keys], dtype=float)[:, None]  # (n_features,1)
            scores = (w * A).sum(axis=0)  # (n_words,)
            scores_list = scores.tolist()
        else:
            # python loop path
            scores_list = []
            for w_idx, w in enumerate(words):
                s = 0.0
                row = normalized[w]
                for feat in feature_keys:
                    s += weights.get(feat, 0.0) * float(row.get(feat, 0.0))
                scores_list.append(s)

        scored = list(zip(words, scores_list))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))  # deterministic: score desc, then word asc

        # Personalizer post-processing (if present)
        if personalizer:
            try:
                scored = personalizer.bias_words(scored)
            except Exception as e:
                logger.debug("personalizer failed during rank_normalized: %s", e)

        # Round numeric scores for stability
        return [(w, round(float(sc), 6)) for w, sc in scored[:topn]]

    # ---------------------------
    # Compatibility: list-based API (keeps previous behavior)
    # -----------------------------------
    def rank(self,
             markov: Optional[List[Tuple[str, float]]] = None,
             embeddings: Optional[List[Tuple[str, float]]] = None,
             fuzzy: Optional[FuzzyList] = None,
             base_freq: Optional[Dict[str, float]] = None,
             recency_map: Optional[Dict[str, float]] = None,
             topn: int = 8) -> CandidateList:
        """
        Backwards compatible API: accepts lists/maps and performs normalization internally.
        This is slower than rank_normalized for repeated calls with same candidate sets.
        """
        markov = markov or []
        embeddings = embeddings or []
        fuzzy = fuzzy or []
        base_freq = base_freq or {}
        recency_map = recency_map or {}

        # Collect candidates in deterministic order
        cand = set([w for w, _ in markov] + [w for w, _ in embeddings] + list(base_freq.keys()))
        if not cand:
            return []
        words = sorted(cand)

        # Build raw maps
        markov_map = {w: float(v) for w, v in markov}
        embed_map = {w: float(v) for w, v in embeddings}
        fuzzy_map = {w: int(d) for w, d in fuzzy}
        freq_map = {w: float(base_freq.get(w, 0.0)) for w in words}
        rec_map = {w: float(recency_map.get(w, 0.0)) for w in words}

        # Basic normalization heuristics (log1p for counts, min-max for embeddings/fuzzy)
        # For simplicity this method uses small helpers (kept inline for readability)
        def _minmax(vals):
            if not vals:
                return {}
            lo, hi = min(vals), max(vals)
            if hi == lo:
                return {i: 1.0 for i in range(len(vals))}
            rng = hi - lo
            return {i: (vals[i] - lo) / rng for i in range(len(vals))}

        # Convert to per-word normalized dict
        mvals = [markov_map.get(w, 0.0) for w in words]
        evals = [embed_map.get(w, 0.0) for w in words]
        fvals = [1.0 / (1.0 + float(fuzzy_map.get(w, 999))) for w in words]  # similarity-like
        fqvals = [math.log1p(freq_map.get(w, 0.0)) for w in words]
        # recency: newer better -> normalized age inverted
        rec_raw = []
        if rec_map:
            newest = max(rec_map.values())
            for w in words:
                if w in rec_map:
                    age = max(0.0, newest - rec_map[w])
                    rec_raw.append(1.0 / (1.0 + math.log1p(age)))
                else:
                    rec_raw.append(0.0)
        else:
            rec_raw = [0.0 for _ in words]

        m_norm = _minmax([math.log1p(v) for v in mvals])
        e_norm = _minmax(evals)
        f_norm = _minmax(fvals)
        fq_norm = _minmax([v for v in fqvals])
        r_norm = _minmax(rec_raw)

        normalized: NormalizedPerWord = {}
        for idx, w in enumerate(words):
            normalized[w] = {
                "markov": float(m_norm.get(idx, 0.0)),
                "embed": float(e_norm.get(idx, 0.0)),
                "fuzzy": float(f_norm.get(idx, 0.0)),
                "freq": float(fq_norm.get(idx, 0.0)),
                "recency": float(r_norm.get(idx, 0.0)),
            }

        return self.rank_normalized(normalized, topn=topn)

    # -----------------------------
    # Helpers: debug contributions
    # ------------------------------
    @staticmethod
    def debug_contributions(word: str, normalized: NormalizedPerWord, weights: Weights) -> Dict[str, float]:
        """
        Return per-feature weighted contributions for a single word given a normalized map and weights.
        Good for explainability/inspecting scores.
        """
        row = normalized.get(word, {})
        s = {}
        for feat, w in weights.items():
            s[feat] = float(w) * float(row.get(feat, 0.0))
        s["final"] = sum(s.values())
        return s
