# fusion_ranker.py
# intelligent_autocompleter/core/fusion_ranker.py
#
# Simple, testable fusion ranker that normalizes signals and combines them
# using named presets. Designed to be small, readable and easy to tweak.
#
# Signals supported:
#  - markov: list[(word, count)]
#  - embeddings: list[(word, score)]
#  - fuzzy: list[(word, distance)]  (smaller = better)
#  - base_freq: dict[word -> int]
#  - recency_map: dict[word -> epoch_ts]
#
# The personalizer (optional) should implement `bias_words(list[(w,score)]) -> list[(w,score)]`.

from typing import Dict, Iterable, List, Tuple, Optional
from collections import defaultdict
import math

CandidateList = List[Tuple[str, float]]
FuzzyList = List[Tuple[str, int]]


def _minmax_scale(vals: Iterable[float]) -> List[float]:
    vals = list(vals)
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [1.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def _log_plus_one(vals: Iterable[float]) -> List[float]:
    return [math.log1p(max(0.0, float(v))) for v in vals]


class FusionRanker:
    PRESETS = {
        "strict":   {"markov": 0.6, "embed": 0.1, "fuzzy": 0.05, "freq": 0.2, "personal": 0.05},
        "balanced": {"markov": 0.4, "embed": 0.3, "fuzzy": 0.1,  "freq": 0.15, "personal": 0.05},
        "creative": {"markov": 0.2, "embed": 0.6, "fuzzy": 0.05, "freq": 0.05, "personal": 0.1},
        "personal": {"markov": 0.25,"embed": 0.25,"fuzzy": 0.05, "freq": 0.1, "personal": 0.35},
    }

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 preset: str = "balanced",
                 personalizer: Optional[object] = None):
        base = dict(self.PRESETS.get(preset, self.PRESETS["balanced"]))
        if weights:
            base.update(weights)
        s = sum(base.values()) or 1.0
        # normalize to sum to 1.0
        self.w = {k: float(v) / s for k, v in base.items()}
        self.personalizer = personalizer

    def rank(self,
             markov: Optional[CandidateList] = None,
             embeddings: Optional[CandidateList] = None,
             fuzzy: Optional[FuzzyList] = None,
             base_freq: Optional[Dict[str, int]] = None,
             recency_map: Optional[Dict[str, float]] = None,
             topn: int = 8) -> CandidateList:
        markov = markov or []
        embeddings = embeddings or []
        fuzzy = fuzzy or []
        base_freq = base_freq or {}
        recency_map = recency_map or {}

        # collect candidate set
        cand = set()
        cand.update([w for w, _ in markov])
        cand.update([w for w, _ in embeddings])
        cand.update([w for w, _ in fuzzy])
        cand.update(base_freq.keys())
        if not cand:
            return []

        # maps for quick lookup
        m_map = {w: c for w, c in markov}
        e_map = {w: s for w, s in embeddings}
        f_map = {w: d for w, d in fuzzy}
        freq_map = dict(base_freq)
        rec_map = dict(recency_map)

        # normalize each signal into [0,1]
        # markov: log1p -> minmax
        m_vals = _log_plus_one([m_map.get(w, 0) for w in cand])
        m_norm = dict(zip(list(cand), _minmax_scale(m_vals)))

        # embeddings: direct minmax
        e_vals = [float(e_map.get(w, 0.0)) for w in cand]
        e_norm = dict(zip(list(cand), _minmax_scale(e_vals)))

        # fuzzy: convert dist -> similarity (1/(1+dist)) then minmax
        f_raw = [1.0 / (1.0 + float(f_map.get(w, 999))) if w in f_map else 0.0 for w in cand]
        f_norm = dict(zip(list(cand), _minmax_scale(f_raw)))

        # freq: log1p -> minmax
        freq_vals = _log_plus_one([freq_map.get(w, 0) for w in cand])
        freq_norm = dict(zip(list(cand), _minmax_scale(freq_vals)))

        # recency: newer -> higher. rec_map should have epoch timestamps.
        rec_raw = []
        now = max(rec_map.values()) if rec_map else None
        for w in cand:
            if now and w in rec_map:
                age = max(0.0, now - rec_map[w])
                rec_raw.append(1.0 / (1.0 + math.log1p(age)))
            else:
                rec_raw.append(0.0)
        rec_norm = dict(zip(list(cand), _minmax_scale(rec_raw)))

        # combine with weights
        scores = {}
        for w in cand:
            s = 0.0
            s += self.w.get("markov", 0.0) * m_norm.get(w, 0.0)
            s += self.w.get("embed", 0.0)  * e_norm.get(w, 0.0)
            s += self.w.get("fuzzy", 0.0)  * f_norm.get(w, 0.0)
            s += self.w.get("freq", 0.0)   * freq_norm.get(w, 0.0)
            # recency separate key (optional)
            s += (self.w.get("recency", 0.0) * rec_norm.get(w, 0.0)) if "recency" in self.w else 0.0
            # leave 'personal' weight to personalizer post-step
            scores[w] = float(s)

        # convert to list and post-process personalization if available
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        if self.personalizer:
            try:
                ranked = self.personalizer.bias_words(ranked)
            except Exception:
                # personalizer is optional; if it fails, keep original ranking
                pass

        ranked = sorted(ranked, key=lambda kv: -kv[1])
        return [(w, round(float(sc), 4)) for w, sc in ranked[:topn]]

