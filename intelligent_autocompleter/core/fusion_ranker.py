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

This ranker normalizes each signal and computes a weighted sum.
It also supports named presets for quick experimentation purposes.
"""

from typing import Dict, Iterable, List, Tuple, Optional
from collections import defaultdict
import math

# typing alias
CandidateList = List[Tuple[str, float]]


def _minmax_scale(vals: Iterable[float]) -> List[float]:
    vals = list(vals)
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [1.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def _log_plus_one(vals: Iterable[float]) -> List[float]:
    return [math.log1p(max(0.0, v)) for v in vals]


class FusionRanker:
    """
    Combine signals from multiple models into a final ranked list.

    Parameters: 
    weights : dict[str, float]
        Weights for signals. Keys supported: "markov", "embed", "fuzzy",
        "freq", "recency", "personal".
    preset : Optional[str]
        Mode preset -> modifies default weights. One of {"strict","balanced",
        "creative","personal"}.
    personalizer : optional object with `.bias_words(list[(w,score)]) -> list[(w,score)]`, 
    for example the CtxPersonal instance.
    """

    PRESETS = {
        "strict":   {"markov": 0.6, "embed": 0.1, "fuzzy": 0.1, "freq": 0.15, "personal": 0.05},
        "balanced": {"markov": 0.4, "embed": 0.3, "fuzzy": 0.1, "freq": 0.15, "personal": 0.05},
        "creative": {"markov": 0.2, "embed": 0.6, "fuzzy": 0.1, "freq": 0.05, "personal": 0.05},
        "personal": {"markov": 0.25, "embed": 0.25, "fuzzy": 0.1, "freq": 0.15, "personal": 0.25},
    }

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 preset: Optional[str] = "balanced",
                 personalizer: Optional[object] = None):
        if preset and preset in self.PRESETS:
            base = dict(self.PRESETS[preset])
        else:
            base = {"markov": 0.4, "embed": 0.3, "fuzzy": 0.1, "freq": 0.15, "personal": 0.05}
        if weights:
            base.update(weights)
        # normalize weights to sum 1
        s = sum(base.values()) or 1.0
        self.w = {k: float(v) / s for k, v in base.items()}
        self.personalizer = personalizer

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
        Returns top `topn` (word, score) sorted in descending order.
        """
        # gather candidate set
        cand_set = set()
        markov = markov or []
        embeddings = embeddings or []
        fuzzy = fuzzy or []
        base_freq = base_freq or {}
        recency_map = recency_map or {}
        cand_set.update([w for w, _ in markov])
        cand_set.update([w for w, _ in embeddings])
        cand_set.update([w for w, _ in fuzzy])
        cand_set.update(base_freq.keys())
        if not cand_set:
            return []

        # arrays for normalization
        m_map = {w: c for w, c in markov}
        e_map = {w: s for w, s in embeddings}
        f_map = {w: d for w, d in fuzzy}  # dist -> smaller better
        freq_map = dict(base_freq)
        rec_map = dict(recency_map)

        # normalization
        # markov: counts to logplusone then minmax
        m_vals = _log_plus_one([m_map.get(w, 0) for w in cand_set])
        m_norm = dict(zip(list(cand_set), _minmax_scale(m_vals)))

        # embedding - similarity scores then minmax (usually between -1-1 or 0-1)
        e_vals = [e_map.get(w, 0.0) for w in cand_set]
        e_norm_vals = _minmax_scale(e_vals)
        e_norm = dict(zip(list(cand_set), e_norm_vals))

        # fuzzy: edit distance to convert to similarity penalty (smaller dist is higher)
        # map dist to score = 1/(1+dist) then minmax
        f_raw = []
        for w in cand_set:
            if w in f_map:
                f_raw.append(1.0 / (1.0 + float(f_map[w])))
            else:
                f_raw.append(0.0)
        f_norm_vals = _minmax_scale(f_raw)
        f_norm = dict(zip(list(cand_set), f_norm_vals))

        # base frequency: logplusone then minmax
        freq_vals = _log_plus_one([freq_map.get(w, 0) for w in cand_set])
        freq_norm_vals = _minmax_scale(freq_vals)
        freq_norm = dict(zip(list(cand_set), freq_norm_vals))

        # Recency: modify timestamps to recency weight, newer is higher.
        # rec_map value is expected as epoch timestamp so convert it to age
        rec_raw = []
        now = None
        if rec_map:
            now = max(rec_map.values())
        for w in cand_set:
            if w in rec_map and now:
                age = max(0.0, now - rec_map[w])
                # convert to recency score: recent -> ~1, old -> ~0
                rec_raw.append(1.0 / (1.0 + math.log1p(age)))
            else:
                rec_raw.append(0.0)
        rec_norm_vals = _minmax_scale(rec_raw)
        rec_norm = dict(zip(list(cand_set), rec_norm_vals))

        # Combine weighted signals --------------------------------------------------
        scores = {}
        for w in cand_set:
            s = 0.0
            s += self.w.get("markov", 0.0) * m_norm.get(w, 0.0)
            s += self.w.get("embed", 0.0)  * e_norm.get(w, 0.0)
            s += self.w.get("fuzzy", 0.0)  * f_norm.get(w, 0.0)
            s += self.w.get("freq", 0.0)   * freq_norm.get(w, 0.0)
            # personal weight is used via personalizer if present
            s += self.w.get("personal", 0.0) * (1.0 if self.personalizer is None else 0.0)
            # recency is combined with 'freq' or considered separately if present
            s += (self.w.get("recency", 0.0) * rec_norm.get(w, 0.0)) if "recency" in self.w else 0.0
            scores[w] = float(s)

        # apply personalizer as a post-process if available
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        if self.personalizer:
            # personalizer expects list[(word,score)] and returns same form
            ranked = self.personalizer.bias_words(ranked)

        # final stable sort & return
        ranked = sorted(ranked, key=lambda kv: -kv[1])
        return [(w, round(float(sc), 4)) for w, sc in ranked[:topn]]

""" 
Alternative??

# fusion_ranker.py
# fusion ranker for experiments and presets.

from typing import Dict, List, Tuple
from collections import defaultdict

Scores = Dict[str, float]


class FusionRanker:
    """
    Lightweight fusion ranker with a few presets.
    Not fancy — easy to read and tweak.
    """

    PRESETS = {
        "balanced": {"markov": 0.4, "embed": 0.4, "personal": 0.15, "freq": 0.05},
        "strict":   {"markov": 0.7, "embed": 0.15, "personal": 0.1,  "freq": 0.05},
        "personal": {"markov": 0.2, "embed": 0.2,  "personal": 0.55, "freq": 0.05},
        "semantic": {"markov": 0.1, "embed": 0.75, "personal": 0.1,  "freq": 0.05},
    }

    def __init__(self, preset: str = "balanced"):
        self.update_preset(preset)

    def update_preset(self, preset: str):
        if preset not in self.PRESETS:
            raise ValueError(f"unknown preset: {preset!r}")
        self.preset = preset
        self.weights = self.PRESETS[preset]

    def _all_words(self, *score_maps: Scores):
        s = set()
        for m in score_maps:
            s.update(m.keys())
        return s

    def fuse(self, markov: Scores, embed: Scores,
             personal: Scores, freq: Scores) -> Scores:
        out: Scores = {}
        for w in self._all_words(markov, embed, personal, freq):
            m = markov.get(w, 0.0)
            e = embed.get(w, 0.0)
            p = personal.get(w, 0.0)
            f = freq.get(w, 0.0)
            # linear combination — simple and explainable
            out[w] = (self.weights["markov"] * m +
                      self.weights["embed"] * e +
                      self.weights["personal"] * p +
                      self.weights["freq"] * f)
        return out

    def rank(self, markov: Scores, embed: Scores,
             personal: Scores, freq: Scores, topn: int = 5) -> List[Tuple[str, float]]:
        """
        Return top-n (word,score) pairs sorted descending by fused score.
        Ties are kept stable by insertion order of python dicts.
        """
        combined = self.fuse(markov, embed, personal, freq)
        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:topn]

"""

