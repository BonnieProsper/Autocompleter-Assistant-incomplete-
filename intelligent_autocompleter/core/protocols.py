# intelligent_autocompleter/core/protocols.py
"""
Protocol interfaces for the core components of the Intelligent Autocompleter.

These Protocols are intentionally small and focused: they describe the methods the core
HybridPredictor/FusionRanker/tests rely on. Chose to depend on Protocols rather than concrete
implementations for better testability and modularity.
Keep this file stable.
"""

from __future__ import annotations

from typing import Protocol, Iterable, List, Tuple, Dict, Any, Optional, runtime_checkable
from typing_extensions import TypedDict


# Typed structures used across components ------------------------------------

class NormalizedFeatureMap(TypedDict, total=False):
    """
    NormalizedFeatureMap maps a word -> per-feature normalized values in [0.0, 1.0].

    Example:
      {
        "word1": {"markov": 0.9, "embed": 0.1, "personal": 0.0, "freq": 0.02, "recency": 0.0, "fuzzy": 0.0},
        ...
      }
    """
    markov: float
    embed: float
    personal: float
    freq: float
    recency: float
    fuzzy: float


NormalizedMaps = Dict[str, NormalizedFeatureMap]  # mapping: token -> per-feature map


# Protocols ------------------------------------------------------------------

class MarkovProtocol(Protocol):
    """Minimal interface for Markov predictor used by HybridPredictor."""

    def train_sentence(self, s: str) -> None:
        ...

    def top_next(self, prev: str, topn: int = 5) -> List[Tuple[str, int]]:
        """
        Return list of (next_token, count) sorted by descending count.
        """
        ...


class BKTreeProtocol(Protocol):
    """Interface for a BK-Tree used to provide fuzzy matches."""

    def insert(self, token: str) -> None:
        ...

    def query(self, token: str, max_dist: int = 2) -> List[Tuple[str, int]]:
        """
        Return list of (matched_token, distance).
        """
        ...


class SemanticEngineProtocol(Protocol):
    """
    Interface for embedding/semantic search.

    Implementations may provide both 'similar' (word-level) and 'search' (sentence-level).
    """

    def similar(self, token: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Optional: find similar tokens to `token`. Return list[(token, score)] where
        score is higher for more similar tokens.
        """
        ...

    def search(self, text: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Optional: search semantic index for `text` and return (item, score).
        """
        ...


class FeaturePreprocessorProtocol(Protocol):
    """Normalizes raw feature maps into normalized per-word feature dicts."""

    def normalize_all(self,
                      markov: Dict[str, float],
                      embed: Dict[str, float],
                      fuzzy: Dict[str, int],
                      freq: Dict[str, float],
                      recency: Dict[str, float]) -> NormalizedMaps:
        """
        Convert raw scores to normalized feature values.

        Return mapping token -> NormalizedFeatureMap.
        """
        ...


class ReinforcementProtocol(Protocol):
    """Interface for the reinforcement/feedback component."""

    def record_accept(self, context: str, suggestion: str, source: Optional[str] = None) -> None:
        ...

    def record_reject(self, context: str, suggestion: str, source: Optional[str] = None) -> None:
        ...

    def get_weights(self) -> Dict[str, float]:
        """
        Returns a dict of normalized weights, common keys:
          - "semantic" or "embed"
          - "markov"
          - "personal"
          - "plugin"
        Values should be floats that can be renormalized by callers.
        """
        ...


class PersonalizerProtocol(Protocol):
    """
    Personalization interface (e.g. CtxPersonal). Two shapes are supported:

     - high-level: bias_words(ranked: List[(token,score)]) -> List[(token,score)]
     - lower-level: adjust_map(m: Dict[token,float], source: str) -> Dict[token,float]
    """

    def bias_words(self, ranked: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        ...

    def adjust_map(self, m: Dict[str, float], source: str) -> Dict[str, float]:
        ...
    

class FusionRankerProtocol(Protocol):
    """
    Required FusionRanker API used by HybridPredictor and tests.

    Implementations should provide two ranking entrypoints:
      - rank(...) for legacy list-based calls
      - rank_normalized(normalized_maps, weights, personalizer, topn) for hot-path
    """

    def rank(self,
             markov: List[Tuple[str, float]],
             embeddings: List[Tuple[str, float]],
             fuzzy: List[Tuple[str, int]],
             base_freq: Dict[str, float],
             recency_map: Dict[str, float],
             topn: int = 10) -> List[Tuple[str, float]]:
        ...

    def rank_normalized(self,
                        normalized_maps: NormalizedMaps,
                        weights: Dict[str, float],
                        personalizer: Optional[PersonalizerProtocol] = None,
                        topn: int = 10) -> List[Tuple[str, float]]:
        """
        Rank using normalized per-token feature maps for speed.
        Returns list[(token, score)] sorted desc by score.
        """
        ...

    def debug_contributions(self, token: str, normalized_maps: NormalizedMaps, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Return per-feature contribution for `token` for explainability/diagnostics.
        """
        ...


class PluginRegistryProtocol(Protocol):
    """Plugin registry hooks used by HybridPredictor and the CLI/TUI."""

    def call_train(self, lines: Iterable[str]) -> None:
        ...

    def call_retrain(self, sentence: str) -> None:
        ...

    def run_suggest_pipeline(self, last_token: str, ranked: List[Tuple[str, float]], bundle: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Plugins may mutate the scored list or return a new scored list. Keep the return deterministic.
        """
        ...

    def call_accept(self, w: str, meta: Dict[str, Any]) -> None:
        ...

    def call_reject(self, w: str, meta: Dict[str, Any]) -> None:
        ...


# runtime-checkable variants 
# let you do runtime isinstance(protocol_instance, SomeProtocol) checks if you want
runtime_checkable(MarkovProtocol)
runtime_checkable(BKTreeProtocol)
runtime_checkable(SemanticEngineProtocol)
runtime_checkable(FeaturePreprocessorProtocol)
runtime_checkable(ReinforcementProtocol)
runtime_checkable(PersonalizerProtocol)
runtime_checkable(FusionRankerProtocol)
runtime_checkable(PluginRegistryProtocol)
