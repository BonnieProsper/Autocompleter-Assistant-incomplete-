# intelligent_autocompleter/core/protocols.py

from __future__ import annotations
from typing import Protocol, Iterable, List, Tuple, Any, Optional, Dict

class SemanticEngineProtocol(Protocol):
    """
    Protocol for a semantic engine used by HybridPredictor.
    Methods:
      - similar(word: str, topn: int) -> List[Tuple[word, score]]
      - search(text: str, k: int) -> List[Tuple[item, score]]
    """
    def similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]: ...
    def search(self, text: str, k: int = 10) -> List[Tuple[str, float]]: ...

class PluginRegistryProtocol(Protocol):
    """
    Plugin registry hooks used by HybridPredictor.
    """
    def call_train(self, lines: Iterable[str]) -> None: ...
    def call_retrain(self, sentence: str) -> None: ...
    def run_suggest_pipeline(self, last: str, ranked: List[Tuple[str, float]], bundle: Dict[str, Any]) -> List[Tuple[str, float]]: ...
    def call_accept(self, word: str, meta: Dict[str, Any]) -> None: ...
    def call_reject(self, word: str, meta: Dict[str, Any]) -> None: ...
