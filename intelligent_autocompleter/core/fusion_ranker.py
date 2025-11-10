# fusion_ranker.py
"""
Fusion Ranker:
Combines multiple suggestion sources into a single ranked list.

Expected input examples:
  - markov: list[(word, count)]
  - embeddings: list[(word, similarity_score)]
  - fuzzy: list[(word, edit_distance)]   # lower is better
  - base_freq: dict[word -> integer]
  - optional: recent_ts or last_used map for recency

Normalizes each signal and computes a weighted ranking.
It also supports named presets for quick experimentation purposes.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional
import logging
import math

# typing persona/scores 
CandidateList = List[Tuple[str, float]]
Scores = Dict[str, float]

def _minmax_scale(vals: Iterable[float]) -> List[float]:
    vals = list(vals)
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [1.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]

def _log_plus_one(vals: Iterable[float]) -> List[float]:
    """Apply log1p and return list."""
    return [math.log1p(max(0.0, v)) for v in vals]


def _inv_distance(dist: float) -> float:
    """Convert edit distance to a similarity score."""
    return 1.0 / (1.0 + dist)

class FusionRanker:
    """
    Combine signals from multiple models into a final ranked list.

    Parameters: 
    weights : dict[str, float]
    Weights for each input signal. Keys supported: "markov", "embed", "fuzzy",
    "freq", "recency", "personal".
    preset : str, optional
    
    Mode preset defines the weighting style, the options are: 
    strict = heavy markov/structural
    balanced = even across sources
    creative = more semantic/embedding based
    personal = emphasise personalisation
    
    personalizer : 
    optional object with `.bias_words(list[(w,score)]) -> list[(w,score)]`, 
    usually the CtxPersonal instance.
    """

    PRESETS = {
        "strict":   {"markov": 0.6, "embed": 0.1, "fuzzy": 0.1, "freq": 0.15, "personal": 0.05},
        "balanced": {"markov": 0.4, "embed": 0.3, "fuzzy": 0.1, "freq": 0.15, "personal": 0.05},
        "creative": {"markov": 0.2, "embed": 0.6, "fuzzy": 0.1, "freq": 0.05, "personal": 0.05},
        "personal": {"markov": 0.25, "embed": 0.25, "fuzzy": 0.1, "freq": 0.15, "personal": 0.25},
    }

    def __init__(
      self,
      weights: Optional[Dict[str, float]] = None,
      preset: Optional[str] = "balanced",
      personalizer: Optional[object] = None,
    ):
        if preset and preset in self.PRESETS:
            base = dict(self.PRESETS[preset])
        else:
            base = dict(self.PRESETS["balanced"]
        if weights:
            base.update(weights)
        # normalize weights to sum 1
        total = sum(base.values()) or 1.0
        self.weights = {k: v / total for k, v in base.items()}
        self.personalizer = personalizer
        self.logger = logging.getLogger(__name__)

    # public API --------------------------------------------
    def rank(self,
             markov: Optional[CandidateList] = None,
             embeddings: Optional[CandidateList] = None,
             fuzzy: Optional[List[Tuple[str, int]]] = None,
             base_freq: Optional[Dict[str, int]] = None,
             recency_map: Optional[Dict[str, float]] = None,
             topn: int = 8) -> CandidateList:
        """
        Build a fused ranking from the available signals from different models.
        Returns a list (word, score) sorted in descending order by final weight.
        """
        # gather candidate set
        markov = markov or []
        embeddings = embeddings or []
        fuzzy = fuzzy or []
        base_freq = base_freq or {}
        recency_map = recency_map or {}
        # candidate words
        candidates = set()
        for src in (markov, embeddings, fuzzy):
            candidates.update([w for w, _ in src])
        candidates.update(base_freq.keys())
        if not candidates:
            return []

        # arrays to normalise each signal
        m_map = {w: c for w, c in markov}
        e_map = {w: s for w, s in embeddings}
        f_map = {w: d for w, d in fuzzy}  # dist -> smaller better

        # normalization ----------------------------------------------
        # markov normalisation - counts to logplusone then minmax
        m_vals = _log_plus_one([m_map.get(w, 0) for w in candidates])
        m_norm = dict(zip(list(candidates), _minmax_scale(m_vals)))

        # embedding - similarity scores then minmax (usually between -1-1 or 0-1)
        e_vals = [e_map.get(w, 0.0) for w in candidates]
        e_norm = dict(zip(candidates, _minmax_scale(e_vals)))

        # fuzzy - edit distance to convert to similarity penalty (smaller dist is higher)
        # map dist to score = 1/(1+dist) then minmax
        f_vals = [_inv_distance(f_map.get(w, 100.0)) for w in candidates]
        f_norm = dict(zip(candidates, _minmax_scale(f_vals)))

        # base frequency normalisation: logplusone then minmax
        freq_vals = _log_plus_one([base_freq.get(w, 0) for w in candidates])
        freq_norm = dict(zip(candidates, _minmax_scale(freq_vals)))

        # Recency normalisation: modify timestamps to recency weight, newer is higher.
        # recency_map value is expected as epoch timestamp so convert it to age
        now = max(recency_map.values()) if recency_map else None
        rec_vals = []
        for w in candidates:
            if now and w in recency_map:
                age = max(0.0, now - recency_map[w])
                rec_vals.append(1.0 / (1.0 + math.log1p(age)))
            else:
                rec_vals.append(0.0)
        rec_norm = dict(zip(candidates, _minmax_scale(rec_vals)))


        # Combine weighted signals --------------------------------------------------
        scores: Dict[str, float] = {}
        for w in candidates:
            s = 0.0
            s += self.weights.get("markov", 0) * m_norm.get(w, 0.0)
            s += self.weights.get("embed", 0) * e_norm.get(w, 0.0)
            s += self.weights.get("fuzzy", 0) * f_norm.get(w, 0.0)
            s += self.weights.get("freq", 0) * freq_norm.get(w, 0.0)
            s += self.weights.get("recency", 0) * rec_norm.get(w, 0.0)
            scores[w] = s

        # apply personal bias
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if self.personalizer:
            # personalizer expects list[(word,score)] and returns same form
            ranked = self.personalizer.bias_words(ranked)

        # final stable sort to descending order & return
        ranked = sorted(ranked, key=lambda kv: kv[1], reverse=True)

        # round for readability
        result = [(w, round(s, 4)) for w, s in ranked[:topn]]
        self.logger.debug(f"[FusionRanker] Returned {len(result)} candidates.")
        return result

    def explain_weights(self) -> str:
        """Return formatted summary of current weight configuration."""
        lines = [f"FusionRanker preset weights:"]
        for k, v in self.weights.items():
            lines.append(f"  {k:<10} = {v:.2f}")
        return "\n".join(lines)

    def update_preset(self, preset: str):
        """Change preset weights dynamically."""
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        self.__init__(preset=preset, personalizer=self.personalizer)
        self.logger.info(f"[FusionRanker] Updated preset → {preset}")


