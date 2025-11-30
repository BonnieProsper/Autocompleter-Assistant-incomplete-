# intelligent_autocompleter/core/protocols.py

from __future__ import annotations
from typing import Protocol, Iterable, List, Tuple, Any, Optional, Dict, TypeAlias


# Shared Type Aliases --------------------------------
Token: TypeAlias = str
Score: TypeAlias = float
RankedList: TypeAlias = List[Tuple[Token, Score]]
Bundle: TypeAlias = Dict[str, Any]


class SemanticEngineProtocol(Protocol):
    """
    Interface for all semantic/embedding-based search engines.

    Implementations must provide:
      - similar(token): nearest-neighbor lookup
      - search(text): semantic retrieval
    """
    def similar(self, token: Token, topn: int = 10) -> RankedList: ...
    def search(self, text: str, k: int = 10) -> RankedList: ...


class MarkovProtocol(Protocol):
    """
    Statistical language model interface.
    """
    def train_sentence(self, sentence: str) -> None: ...
    def top_next(self, prev: Token, topn: int = 5) -> List[Tuple[Token, int]]: ...
    def vocabulary_size(self) -> int: ...


class BKTreeProtocol(Protocol):
    """
    Fuzzy search for typo correction (BK-tree interface).
    """
    def insert(self, token: Token) -> None: ...
    def query(self, token: Token, max_dist: int = 2) -> List[Tuple[Token, int]]: ...


class ReinforcementProtocol(Protocol):
    """
    Records user accept/reject events and exposes learned source weights.
    """
    def record_accept(self, context: str, suggestion: Token, source: Optional[str] = None) -> None: ...
    def record_reject(self, context: str, suggestion: Token, source: Optional[str] = None) -> None: ...
    def get_weights(self) -> Dict[str, float]: ...


class PersonalizerProtocol(Protocol):
    """
    Personalized ranking adjustments (user-frequency bias, contextual prefs).
    """
    def bias_words(self, ranked: RankedList) -> RankedList: ...

    # Optionally: deeper map-level weighting used by advanced Rankers
    def adjust_map(self, m: Dict[Token, Score], source: str) -> Dict[Token, Score]: ...


# Plugin Registry (Pipeline Hooks) -------------------------------------------
class PluginRegistryProtocol(Protocol):
    """
    Extensible plugin pipeline:
      - Training hooks
      - Realtime retraining
      - Suggestion pipeline transformation
      - Accept/reject user feedback plugins
    """
    def run_suggest_pipeline(
        self,
        last_token: Token,
        ranked: RankedList,
        bundle: Bundle
    ) -> RankedList: ...

    def call_train(self, lines: Iterable[str]) -> None: ...
    def call_retrain(self, sentence: str) -> None: ...
    def call_accept(self, token: Token, meta: Bundle) -> None: ...
    def call_reject(self, token: Token, meta: Bundle) -> None: ...
