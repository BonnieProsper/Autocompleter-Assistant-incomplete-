# intelligent_autocompleter/core/fusion_ranker.py
"""
FusionRanker - robust, production-ready multi-signal ranker.

Design goals:
 - Per-feature normalization (pre-fusion) - remove?? e.g _minmax, log1pminmax, safesoftmax 
 - Strategy pattern (LINEAR, SOFTMAX, BORDA)
 - Robust weight management: set, update (with lr, clip, normalize), validation
 - Optional numpy acceleration with fallback
 - Personalizer hook preserved (personalizer.bias_words(list[(w,score)]) -> list[(w,score)])
 - Clear typing and error messages for debugging/unit tests
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Mapping, Iterable
from enum import Enum
import math
import logging

logger = logging.getLogger(__name__)

# Try optional numpy acceleration
try:
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

# Public types
Candidate = Tuple[str, float]
CandidateList = List[Candidate]
FuzzyList = List[Tuple[str, int]]
ScoreMap = Dict[str, float]
FeatureVector = Dict[str, float]


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


def _minmax_scale(values: Iterable[float]) -> List[float]:
    vals = list(values)
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [1.0 for _ in vals]
    rng = hi - lo
    return [(v - lo) / rng for v in vals]


def _log1p_minmax(values: Iterable[float]) -> List[float]:
    vals = [math.log1p(max(0.0, float(v))) for v in values]
    return _minmax_scale(vals)


def _safe_softmax(values: List[float], temp: float = 1.0) -> List[float]:
    if temp <= 0:
        raise ValueError("softmax temperature must be > 0")
    if not values:
        return []
    mx = max(values)
    exps = [math.exp((v - mx) / temp) for v in values]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


class FusionRanker:
    """
    FusionRanker aggregates prenormalized feature maps and returns the ranked candidates.

    Main API:
      fr = FusionRanker(preset="balanced", strategy=FusionStrategy.LINEAR)
      candidates = fr.rank(markov=[(w,c)], embeddings=[(w,s)], fuzzy=[(w,dist)], base_freq={w:count}, recency_map={w:ts}, topn=8)

    Personalizer is optional and invoked as:
    personalizer.bias_words(list[(w,score)]) -> list[(w,score)].
    """
    ALLOWED_FEATURES = ("markov", "embed", "fuzzy", "freq", "recency", "personal")

    def __init__(
        self,
        preset: str = "balanced",
        weights: Optional[Mapping[str, float]] = None,
        personalizer: Optional[object] = None,
        strategy: FusionStrategy = FusionStrategy.LINEAR,
        use_numpy: bool = NUMPY_AVAILABLE,
    ) -> None:
        preset_map = dict(_DEFAULT_PRESETS.get(preset, _DEFAULT_PRESETS["balanced"]))
        if weights:
            preset_map.update(dict(weights))
        # ensure keys are present
        for k in self.ALLOWED_FEATURES:
            preset_map.setdefault(k, 0.0)
        s = sum(float(v) for v in preset_map.values()) or 1.0
        self._weights: Dict[str, float] = {k: float(preset_map[k]) / s for k in self.ALLOWED_FEATURES}
        self.personalizer = personalizer
        self.strategy = strategy
        self.use_numpy = use_numpy and NUMPY_AVAILABLE

    # Weight Management --------------------------------
    def get_weights(self) -> Dict[str, float]:
        """Return a copy of normalized weights (sums to 1)."""
        return dict(self._weights)

    def set_weights(self, new_weights: Mapping[str, float], normalize: bool = True) -> None:
        """Replace weights. Unknown keys raise ValueError."""
        for k in new_weights.keys():
            if k not in self.ALLOWED_FEATURES:
                raise KeyError(f"Unknown feature '{k}'. Allowed: {self.ALLOWED_FEATURES}")
        # merge and optionally normalize 
        merged = {k: float(new_weights.get(k, 0.0)) for k in self.ALLOWED_FEATURES}
        if normalize:
            s = sum(max(0.0, v) for v in merged.values()) or 1.0
            self._weights = {k: max(0.0, v) / s for k, v in merged.items()}
        else:
            self._weights = merged

    def update_weights(self, deltas: Mapping[str, float], lr: float = 0.1, clip: Optional[Tuple[float, float]] = (0.0, 1.0), normalize: bool = True) -> None:
        """
        Incrementally update weights.

        deltas: map feature->delta
        lr: learning rate multiplier
        clip: (min,max) clip after update (None to disable)
        normalize: renormalize so weights sum to 1 (recommended)
        """
        if lr < 0:
            raise ValueError("lr must be non-negative")
        w = dict(self._weights)
        for k, d in deltas.items():
            if k not in self.ALLOWED_FEATURES:
                logger.debug("update_weights ignored unknown key %s", k)
                continue
            w[k] = float(w.get(k, 0.0)) + float(d) * float(lr)
        if clip is not None:
            lo, hi = float(clip[0]), float(clip[1])
            if lo > hi:
                raise ValueError("clip min must be <= clip max")
            for k in w:
                w[k] = max(lo, min(hi, float(w[k])))
        if normalize:
            s = sum(max(0.0, float(v)) for v in w.values()) or 1.0
            self._weights = {k: max(0.0, float(v)) / s for k, v in w.items()}
        else:
            self._weights = w

    # Internal normalization helpers -----------------------------
    def _collect_keys(self, *lists_or_maps) -> List[str]:
        keys = set()
        for m in lists_or_maps:
            if not m:
                continue
            # if list of tuples
            if isinstance(m, list) and m and isinstance(m[0], tuple):
                keys.update([w for w, _ in m])
            elif isinstance(m, dict):
                keys.update(m.keys())
            else:
                # fallback: iterate if iterable of pairs
                try:
                    for item in m:
                        if isinstance(item, tuple) and item:
                            keys.add(item[0])
                except Exception:
                    pass
        return sorted(keys)

    @staticmethod
    def _minmax_map(keys: List[str], raw_map: Mapping[str, float]) -> ScoreMap:
        vals = [float(raw_map.get(k, 0.0)) for k in keys]
        scaled = _minmax_scale(vals)
        return {k: scaled[i] for i, k in enumerate(keys)}

    @staticmethod
    def _log1p_minmax_map(keys: List[str], raw_map: Mapping[str, float]) -> ScoreMap:
        vals = [float(raw_map.get(k, 0.0)) for k in keys]
        scaled = _log1p_minmax(vals)
        return {k: scaled[i] for i, k in enumerate(keys)}

    @staticmethod
    def _fuzzy_map(keys: List[str], fuzzy_map: Mapping[str, int]) -> ScoreMap:
        raw = [(1.0 / (1.0 + float(fuzzy_map.get(k, 999)))) if k in fuzzy_map else 0.0 for k in keys]
        scaled = _minmax_scale(raw)
        return {k: scaled[i] for i, k in enumerate(keys)}

    def _recency_map(self, keys: List[str], rec_map: Mapping[str, float]) -> ScoreMap:
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
        scaled = _minmax_scale(raw)
        return {k: scaled[i] for i, k in enumerate(keys)}

    # Public ranking API --------------------------------------------
    def rank(
        self,
        markov: Optional[List[Tuple[str, float]]] = None,
        embeddings: Optional[List[Tuple[str, float]]] = None,
        fuzzy: Optional[FuzzyList] = None,
        base_freq: Optional[Mapping[str, float]] = None,
        recency_map: Optional[Mapping[str, float]] = None,
        topn: int = 8,
    ) -> CandidateList:
        """
        Rank candidates based on provided signals.

        Inputs:
          - markov: [(word, count), ...]
          - embeddings: [(word, score), ...]
          - fuzzy: [(word, distance), ...]
          - base_freq: {word: count}
          - recency_map: {word: timestamp}
        """
        markov = markov or []
        embeddings = embeddings or []
        fuzzy = fuzzy or []
        base_freq = base_freq or {}
        recency_map = recency_map or {}

        keys = self._collect_keys(markov, embeddings, fuzzy, base_freq, recency_map)
        if not keys:
            return []

        # quick lookup maps
        m_map: ScoreMap = {w: float(v) for w, v in markov}
        e_map: ScoreMap = {w: float(v) for w, v in embeddings}
        f_map: Dict[str, int] = {w: int(d) for w, d in fuzzy}
        freq_map: ScoreMap = {w: float(v) for w, v in base_freq.items()} if isinstance(base_freq, dict) else dict(base_freq)
        rec_map: ScoreMap = {w: float(v) for w, v in recency_map.items()} if isinstance(recency_map, dict) else dict(recency_map)

        # per-feature normalization before fusion
        try:
            m_norm = self._log1p_minmax_map(keys, m_map)
            e_norm = self._minmax_map(keys, e_map)
            f_norm = self._fuzzy_map(keys, f_map)
            freq_norm = self._log1p_minmax_map(keys, freq_map)
            rec_norm = self._recency_map(keys, rec_map)
        except Exception as exc:
            logger.exception("Feature normalization failed: %s", exc)
            # best-effort zeros
            m_norm = {k: 0.0 for k in keys}
            e_norm = {k: 0.0 for k in keys}
            f_norm = {k: 0.0 for k in keys}
            freq_norm = {k: 0.0 for k in keys}
            rec_norm = {k: 0.0 for k in keys}

        # combine weighted scores
        combined: ScoreMap = {}
        for k in keys:
            s = 0.0
            s += self._weights.get("markov", 0.0) * m_norm.get(k, 0.0)
            s += self._weights.get("embed", 0.0)  * e_norm.get(k, 0.0)
            s += self._weights.get("fuzzy", 0.0)  * f_norm.get(k, 0.0)
            s += self._weights.get("freq", 0.0)   * freq_norm.get(k, 0.0)
            s += self._weights.get("recency", 0.0) * rec_norm.get(k, 0.0)
            combined[k] = float(s)

        # apply strategy
        if self.strategy == FusionStrategy.LINEAR:
            final_map = combined
        elif self.strategy == FusionStrategy.SOFTMAX:
            vals = list(combined.values())
            probs = _safe_softmax(vals, temp=1.0)
            final_map = {k: probs[i] for i, k in enumerate(combined.keys())}
        elif self.strategy == FusionStrategy.BORDA:
            # build feature vectors per candidate and delegate to Borda aggregation
            feature_map: Dict[str, FeatureVector] = {}
            for k in keys:
                feature_map[k] = {
                    "markov": m_norm.get(k, 0.0),
                    "embed": e_norm.get(k, 0.0),
                    "fuzzy": f_norm.get(k, 0.0),
                    "freq": freq_norm.get(k, 0.0),
                    "recency": rec_norm.get(k, 0.0),
                }
            # Borda: rank per feature then weight by feature weight
            score_acc: Dict[str, float] = {k: 0.0 for k in keys}
            features = ["markov", "embed", "fuzzy", "freq", "recency"]
            for feat in features:
                ranked_by_feat = sorted(keys, key=lambda x: -feature_map[x].get(feat, 0.0))
                n = len(ranked_by_feat)
                for pos, word in enumerate(ranked_by_feat):
                    score_acc[word] += (n - 1 - pos) * self._weights.get(feat, 0.0)
            final_map = score_acc
        else:
            logger.warning("Unknown strategy %s - falling back to linear", self.strategy)
            final_map = combined

        # convert to sorted list
        ranked = sorted(final_map.items(), key=lambda kv: -kv[1])

        # allow personalizer last (it may inspect user-specific signals like recency/freq)
        if self.personalizer:
            try:
                ranked = self.personalizer.bias_words(ranked)
            except Exception:
                logger.debug("Personalizer failed to adjust results", exc_info=True)

        # return top-n rounded
        return [(w, round(float(s), 6)) for w, s in ranked[:topn]]

