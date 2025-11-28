# changes: remove optional featurepreprocessor e.g always use preprocessor (why keep optional?)

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
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Mapping, Iterable
from enum import Enum
import math
import logging

logger = logging.getLogger(__name__)

# Optional numpy
try:
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

# Optional FeaturePreprocessor import 
try:
    from intelligent_autocompleter.core.feature_preprocessor import FeaturePreprocessor  # type: ignore
except Exception:
    try:
        from core.feature_preprocessor import FeaturePreprocessor  # local fallback
    except Exception:
        FeaturePreprocessor = None  # type: ignore

# public types
Candidate = Tuple[str, float]
CandidateList = List[Candidate]
FuzzyList = List[Tuple[str, int]]
ScoreMap = Dict[str, float]


class FusionStrategy(Enum):
    LINEAR = "linear"
    SOFTMAX = "softmax"
    BORDA = "borda"


# Default presets 
_DEFAULT_PRESETS: Dict[str, Dict[str, float]] = {
    "strict":   {"markov": 0.6, "embed": 0.1, "fuzzy": 0.05, "freq": 0.2, "personal": 0.05, "recency": 0.0},
    "balanced": {"markov": 0.4, "embed": 0.3, "fuzzy": 0.1,  "freq": 0.15, "personal": 0.05, "recency": 0.0},
    "creative": {"markov": 0.2, "embed": 0.6, "fuzzy": 0.05, "freq": 0.05, "personal": 0.1, "recency": 0.0},
    "personal": {"markov": 0.25,"embed": 0.25,"fuzzy": 0.05, "freq": 0.1, "personal": 0.35, "recency": 0.0},
}


def _minmax_scale_py(vals: Iterable[float]) -> List[float]:
    vals = list(vals)
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [1.0 for _ in vals]
    rng = hi - lo
    return [(v - lo) / rng for v in vals]


class FusionRanker:
    """
    Combine multiple signal maps into a ranked list of candidates.

    Primary methods:
      - rank(...) -> List[(word, score)]
      - get_weights()/set_weights(...)/update_weights(...)
    """

    ALLOWED_FEATURES = {"markov", "embed", "fuzzy", "freq", "recency", "personal"}

    def __init__(self,
                 preset: str = "balanced",
                 weights: Optional[Mapping[str, float]] = None,
                 personalizer: Optional[object] = None,
                 strategy: FusionStrategy = FusionStrategy.LINEAR,
                 use_numpy: bool = NUMPY_AVAILABLE):
        preset_map = dict(_DEFAULT_PRESETS.get(preset, _DEFAULT_PRESETS["balanced"]))
        if weights:
            preset_map.update(weights)
        # ensure all allowed features exist explicitly
        for f in self.ALLOWED_FEATURES:
            preset_map.setdefault(f, 0.0)

        s = sum(float(v) for v in preset_map.values()) or 1.0
        self._weights: Dict[str, float] = {k: float(v) / s for k, v in preset_map.items()}
        self.personalizer = personalizer
        self.strategy = strategy
        self._use_numpy = bool(use_numpy and NUMPY_AVAILABLE)
        # optional preprocessor (use if available)
        self._pre = FeaturePreprocessor(use_numpy=self._use_numpy) if FeaturePreprocessor is not None else None

    # Weight management -----------------------------
    def get_weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def set_weights(self, new: Mapping[str, float], normalize: bool = True) -> None:
        for k in new:
            if k not in self.ALLOWED_FEATURES:
                raise ValueError(f"Unknown feature '{k}'")
        w = {feat: float(new.get(feat, 0.0)) for feat in self.ALLOWED_FEATURES}
        if normalize:
            s = sum(max(0.0, v) for v in w.values()) or 1.0
            self._weights = {k: (max(0.0, v) / s) for k, v in w.items()}
        else:
            self._weights = w

    def update_weights(self, deltas: Mapping[str, float], lr: float = 0.1, clip: Optional[tuple] = (0.0, 1.0), normalize: bool = True) -> None:
        if lr < 0:
            raise ValueError("lr must be >= 0")
        w = dict(self._weights)
        for k, d in deltas.items():
            if k not in self.ALLOWED_FEATURES:
                logger.debug("Ignoring unknown weight key: %s", k)
                continue
            w[k] = float(w.get(k, 0.0)) + float(d) * float(lr)
        if clip is not None:
            lo, hi = float(clip[0]), float(clip[1])
            if lo > hi:
                raise ValueError("clip bounds invalid")
            for k in w:
                w[k] = max(lo, min(hi, float(w[k])))
        if normalize:
            s = sum(max(0.0, v) for v in w.values()) or 1.0
            self._weights = {k: (max(0.0, v) / s) for k, v in w.items()}
        else:
            self._weights = w

    # Internal helpers -------------------------------------
    @staticmethod
    def _collect_keys(markov, embeddings, fuzzy, base_freq, recency):
        keys = set()
        if markov:
            keys.update(w for w, _ in markov)
        if embeddings:
            keys.update(w for w, _ in embeddings)
        if fuzzy:
            keys.update(w for w, _ in fuzzy)
        if base_freq:
            keys.update(base_freq.keys())
        if recency:
            keys.update(recency.keys())
        return sorted(keys)

    def _compute_recency(self, keys: List[str], rec_map: Mapping[str, float]) -> ScoreMap:
        if not rec_map:
            return {k: 0.0 for k in keys}
        newest = max(rec_map.values())
        raw = []
        for k in keys:
            if k in rec_map:
                age = max(0.0, newest - float(rec_map[k]))
                raw.append(1.0 / (1.0 + math.log1p(age)))
            else:
                raw.append(0.0)
        # minmax
        if self._use_numpy:
            arr = np.array(raw, dtype=float)
            lo, hi = float(arr.min()), float(arr.max())
            if hi == lo:
                return {k: 1.0 for k in keys}
            scaled = ((arr - lo) / (hi - lo)).tolist()
            return {k: scaled[i] for i, k in enumerate(keys)}
        else:
            return dict(zip(keys, _minmax_scale_py(raw)))

    def _apply_strategy(self, combined: ScoreMap) -> ScoreMap:
        if self.strategy == FusionStrategy.LINEAR:
            return combined
        if self.strategy == FusionStrategy.SOFTMAX:
            vals = list(combined.values())
            if not vals:
                return combined
            mx = max(vals)
            exps = [math.exp(v - mx) for v in vals]
            s = sum(exps) or 1.0
            out_vals = [e / s for e in exps]
            return {k: out_vals[i] for i, k in enumerate(combined.keys())}
        if self.strategy == FusionStrategy.BORDA:
            # Borda over features: for each feature, rank and award points weighted by feature weight
            features = list(self.ALLOWED_FEATURES)
            score_acc = {k: 0.0 for k in combined.keys()}
            # assume per-candidate normalized features are not available here, so fallback to linear
            # (Borda is more meaningful in rank_from_features API, keep safe fallback)
            return combined
        return combined

    # Public API --------------------------------------------------------
    def rank(self,
             markov: Optional[List[Tuple[str, float]]] = None,
             embeddings: Optional[List[Tuple[str, float]]] = None,
             fuzzy: Optional[FuzzyList] = None,
             base_freq: Optional[Mapping[str, float]] = None,
             recency_map: Optional[Mapping[str, float]] = None,
             topn: int = 8) -> CandidateList:
        """
        Rank candidates using available signals. All inputs are optional.
        Returns topn list[(word, score)] sorted descending by score.
        """
        markov = markov or []
        embeddings = embeddings or []
        fuzzy = fuzzy or []
        base_freq = base_freq or {}
        recency_map = recency_map or {}

        keys = self._collect_keys(markov, embeddings, fuzzy, base_freq, recency_map)
        if not keys:
            return []

        # maps
        m_map: ScoreMap = {w: float(v) for w, v in markov}
        e_map: ScoreMap = {w: float(v) for w, v in embeddings}
        f_map: Dict[str, int] = {w: int(d) for w, d in fuzzy}
        freq_map: ScoreMap = {w: float(v) for w, v in base_freq.items()}
        rec_map: ScoreMap = {w: float(v) for w, v in recency_map.items()}

        # Use FeaturePreprocessor if present otherwise perform inline normalization
        if self._pre:
            try:
                # FeaturePreprocessor.normalize_all returns {word: {feat: val}}
                feature_matrix = self._pre.normalize_all(markov=m_map, embed=e_map, fuzzy=f_map, freq=freq_map, recency=rec_map)
                # if its a full matrix compute weighted sums
                combined: ScoreMap = {}
                for k in keys:
                    fv = feature_matrix.get(k, {})
                    s = 0.0
                    s += self._weights.get("markov", 0.0) * float(fv.get("markov", 0.0))
                    s += self._weights.get("embed", 0.0)   * float(fv.get("embed", 0.0))
                    s += self._weights.get("fuzzy", 0.0)   * float(fv.get("fuzzy", 0.0))
                    s += self._weights.get("freq", 0.0)    * float(fv.get("freq", 0.0))
                    s += self._weights.get("recency", 0.0) * float(fv.get("recency", 0.0))
                    combined[k] = float(s)
            except Exception as e:
                logger.exception("FeaturePreprocessor failed: %s", e)
                # fallback to inline path
                self._pre = None
                return self.rank(markov=markov, embeddings=embeddings, fuzzy=fuzzy, base_freq=base_freq, recency_map=recency_map, topn=topn)
        else:
            # Inline normalization + fallback
            # markov & freq: log1p then minmax
            m_vals = [math.log1p(max(0.0, m_map.get(k, 0.0))) for k in keys]
            e_vals = [float(e_map.get(k, 0.0)) for k in keys]
            f_vals = [ (1.0 / (1.0 + float(f_map.get(k, 999)))) if k in f_map else 0.0 for k in keys ]
            fq_vals = [math.log1p(max(0.0, freq_map.get(k, 0.0))) for k in keys]
            rec_vals = []
            if rec_map:
                newest = max(rec_map.values())
                for k in keys:
                    if k in rec_map:
                        age = max(0.0, newest - rec_map[k])
                        rec_vals.append(1.0 / (1.0 + math.log1p(age)))
                    else:
                        rec_vals.append(0.0)
            else:
                rec_vals = [0.0 for _ in keys]

            # normalize each vector (check)
            if self._use_numpy:
                def _minmax(arr):
                    a = np.array(arr, dtype=float)
                    lo, hi = float(a.min()), float(a.max())
                    if hi == lo:
                        return [1.0 for _ in arr]
                    return ((a - lo) / (hi - lo)).tolist()
                nm = _minmax(m_vals)
                ne = _minmax(e_vals)
                nf = _minmax(f_vals)
                nq = _minmax(fq_vals)
                nr = _minmax(rec_vals)
            else:
                nm = _minmax_scale_py(m_vals)
                ne = _minmax_scale_py(e_vals)
                nf = _minmax_scale_py(f_vals)
                nq = _minmax_scale_py(fq_vals)
                nr = _minmax_scale_py(rec_vals)

            combined = {}
            for i, k in enumerate(keys):
                s = 0.0
                s += self._weights.get("markov", 0.0) * nm[i]
                s += self._weights.get("embed", 0.0)  * ne[i]
                s += self._weights.get("fuzzy", 0.0)  * nf[i]
                s += self._weights.get("freq", 0.0)   * nq[i]
                s += self._weights.get("recency", 0.0) * nr[i]
                combined[k] = float(s)

        # apply fusion strategy (softmax or linear)
        combined = self._apply_strategy(combined)

        # convert to sorted list and let personalizer bump results if present
        ranked = sorted(combined.items(), key=lambda kv: -kv[1])

        if self.personalizer:
            try:
                ranked = self.personalizer.bias_words(ranked)
            except Exception:
                logger.debug("personalizer failed", exc_info=True)

        return [(w, round(float(s), 6)) for w, s in ranked[:topn]]
