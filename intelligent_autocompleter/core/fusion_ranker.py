# fusion_ranker.py
# Unified fusion layer for Autocompleter Assistant. - redundant because of hybrid predictor?
# Merges signals from:
#  - Markov predictions
#  - Embedding similarity
#  - Fuzzy distance (BK-tree)
#  - Base word frequency
#  - Recency weighting
#   - optional personalization plugin

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional, Set
from collections import defaultdict
import math
import logging

logger = logging.getLogger(__name__)

CandidateList = List[Tuple[str, float]]
FuzzyList = List[Tuple[str, int]]

# Utility transforms---------------------------------------------------------
def _minmax_scale(values: Iterable[float]) -> List[float]:
    """Normalize numeric values to [0,1]."""
    values = list(values)
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]

def _log1p(values: Iterable[float]) -> List[float]:
    """Apply log(1+x) element-wise with safety against negatives."""
    return [math.log1p(max(0.0, float(v))) for v in values]



# FusionRanker ---------------------------------------------------------------
class FusionRanker:
    """
    A multi-signal ranker that blends Markov, embedding, fuzzy,
    frequency, and recency scores using normalized weights.

    Rank pipeline is:
     - Gather union of all candidate words.
     - Compute per-signal raw value for each candidate.
     - Apply transform (log-scale, invert distance, etc).
     - Min-max normalize every signal -> [0,1].
     - Weighted sum of all signals.
     - Optional personalization pass.
     - Return top-N words.
     
    PRESETS have tuned weight profiles for different behaviour:
    strict    =  (prefix & Markov heavy)
    balanced  =  (best general behaviour)
    creative  =  (semantic exploration)
    personal  =  (heavily biased to user history)
    """
    PRESETS = {
        "strict":   {"markov": 0.6, "embed": 0.1, "fuzzy": 0.05,
                     "freq": 0.2, "personal": 0.05},
        "balanced": {"markov": 0.4, "embed": 0.3, "fuzzy": 0.1,
                     "freq": 0.15, "personal": 0.05},
        "creative": {"markov": 0.2, "embed": 0.6, "fuzzy": 0.05,
                     "freq": 0.05, "personal": 0.1},
        "personal": {"markov": 0.25, "embed": 0.25, "fuzzy": 0.05,
                     "freq": 0.1, "personal": 0.35},
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        preset: str = "balanced",
        personalizer: Optional[object] = None,
    ):
        base_weights = dict(self.PRESETS.get(preset, self.PRESETS["balanced"]))

        if weights:
            base_weights.update(weights)

        total = sum(base_weights.values()) or 1.0
        self.weights = {k: v / total for k, v in base_weights.items()}
        self.personalizer = personalizer

    # Core ranking -----------------------------------------------------------------
    def rank(
        self,
        markov: Optional[CandidateList] = None,
        embeddings: Optional[CandidateList] = None,
        fuzzy: Optional[FuzzyList] = None,
        base_freq: Optional[Dict[str, int]] = None,
        recency_map: Optional[Dict[str, float]] = None,
        topn: int = 8,) -> CandidateList:
        markov = markov or []
        embeddings = embeddings or []
        fuzzy = fuzzy or []
        base_freq = base_freq or {}
        recency_map = recency_map or {}

        # Collect candidate word set
        candidates: Set[str] = set()
        candidates.update([w for w, _ in markov])
        candidates.update([w for w, _ in embeddings])
        candidates.update([w for w, _ in fuzzy])
        candidates.update(base_freq.keys())
        if not candidates:
            return []

        # lookup maps
        M = {w: v for w, v in markov}
        E = {w: v for w, v in embeddings}
        F = {w: d for w, d in fuzzy}
        FREQ = base_freq
        REC = recency_map

        cand_list = list(candidates)

        # Prepare raw signal vectors
        # Markov → log1p -> minmax
        m_raw = _log1p([M.get(w, 0) for w in cand_list])
        m_norm = dict(zip(cand_list, _minmax_scale(m_raw)))

        # Embeddings → direct minmax
        e_norm = dict(zip(
            cand_list,
            _minmax_scale([float(E.get(w, 0.0)) for w in cand_list])
        ))

        # Fuzzy → invert distance -> minmax
        f_raw = [
            1.0 / (1.0 + float(F.get(w, 999))) if w in F else 0.0
            for w in cand_list
        ]
        f_norm = dict(zip(cand_list, _minmax_scale(f_raw)))

        # Frequency → log1p -> minmax
        freq_norm = dict(zip(
            cand_list,
            _minmax_scale(_log1p([FREQ.get(w, 0) for w in cand_list]))
        ))

        # Recency → log-age inversion -> minmax
        if REC:
            newest = max(REC.values())
        rec_raw = []
        for w in cand_list:
            if REC and w in REC:
                age = max(0.0, newest - REC[w])
                rec_raw.append(1.0 / (1.0 + math.log1p(age)))
            else:
                rec_raw.append(0.0)
        rec_norm = dict(zip(cand_list, _minmax_scale(rec_raw)))

        # Weighted combination
        scores: Dict[str, float] = {}
        for w in cand_list:
            s = 0.0
            s += self.weights.get("markov", 0.0) * m_norm.get(w, 0.0)
            s += self.weights.get("embed", 0.0) * e_norm.get(w, 0.0)
            s += self.weights.get("fuzzy", 0.0) * f_norm.get(w, 0.0)
            s += self.weights.get("freq", 0.0) * freq_norm.get(w, 0.0)
            s += self.weights.get("recency", 0.0) * rec_norm.get(w, 0.0)
            scores[w] = s
        ranked = sorted(scores.items(), key=lambda t: -t[1])

        # Optional personalization pass
        if self.personalizer:
            try:
                ranked = self.personalizer.bias_words(ranked)
            except Exception as e:
                logger.warning(f"Personalizer failed: {e}")

        # final sort & return
        return [(w, round(float(s), 4)) for w, s in ranked[:topn]]

