# feature_preprocessor.py
"""
FeaturePreprocessor
- Input validators and per-feature normalizers for the fusion pipeline.
- Ensures stable, named feature maps before they are fused by FusionRanker.
- Centralises normalization logic to eliminate duplication and inconsistencies.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Sequence, Optional
import math

Numeric = float
ScoreMap = Dict[str, Numeric]  # word -> raw score

class PreprocessorError(ValueError):
    pass

def safe_minmax(values: Sequence[float]) -> List[float]:
    vals = list(values)
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi == lo:
        # constant vector, map to 1.0 to indicate equality
        return [1.0 for _ in vals]
    rng = hi - lo
    return [(v - lo) / rng for v in vals]

def safe_log1p_minmax(values: Sequence[float]) -> List[float]:
    transformed = [math.log1p(max(0.0, float(v))) for v in values]
    return safe_minmax(transformed)

def fuzzy_dist_to_similarity(distances: Sequence[float]) -> List[float]:
    # convert distances to similarity (1/(1+dist)) then minmax
    sims = [1.0 / (1.0 + float(d)) if d is not None else 0.0 for d in distances]
    return safe_minmax(sims)

class FeaturePreprocessor:
    """
    Validate and normalize signals.

    Expected inputs: each signal is a mapping word->value (numbers).
    Output: dict[word] -> NormalizedValues (per-feature)
    Use:
        pre = FeaturePreprocessor()
        norm = pre.normalize_all(keys, {"markov": m_map, "embed": e_map, ...})
    """
    def __init__(self, required_features: Optional[Iterable[str]] = None):
        self.required_features = set(required_features or [])

    def _collect_keys(self, *maps: Dict[str, float]) -> List[str]:
        keys = set()
        for m in maps:
            keys.update(m.keys())
        return sorted(keys)

    def normalize_all(self,
                      markov: ScoreMap = None,
                      embed: ScoreMap = None,
                      fuzzy: Dict[str, float] = None,
                      freq: ScoreMap = None,
                      recency: Dict[str, float] = None) -> Dict[str, Dict[str, float]]:
        markov = markov or {}
        embed = embed or {}
        fuzzy = fuzzy or {}
        freq = freq or {}
        recency = recency or {}

        keys = self._collect_keys(markov, embed, fuzzy, freq, recency)
        if not keys:
            return {}

        # per feature vectors in the deterministic key order
        m_vals = [float(markov.get(k, 0.0)) for k in keys]
        e_vals = [float(embed.get(k, 0.0)) for k in keys]
        f_vals = [float(fuzzy.get(k, 999)) if k in fuzzy else 999.0 for k in keys]
        freq_vals = [float(freq.get(k, 0.0)) for k in keys]
        # recency -> newer is better, convert timestamps to recency-score
        if recency:
            newest = max(recency.values())
            rec_raw = []
            for k in keys:
                if k in recency:
                    age = max(0.0, newest - float(recency[k]))
                    rec_raw.append(1.0 / (1.0 + math.log1p(age)))
                else:
                    rec_raw.append(0.0)
        else:
            rec_raw = [0.0 for _ in keys]

        # normalize each feature independently
        nm = safe_log1p_minmax(m_vals)
        ne = safe_minmax(e_vals)
        nf = fuzzy_dist_to_similarity(f_vals)
        nfq = safe_log1p_minmax(freq_vals)
        nr = safe_minmax(rec_raw)

        out: Dict[str, Dict[str, float]] = {}
        for i, k in enumerate(keys):
            out[k] = {
                "markov": nm[i],
                "embed": ne[i],
                "fuzzy": nf[i],
                "freq": nfq[i],
                "recency": nr[i],
            }
        return out

