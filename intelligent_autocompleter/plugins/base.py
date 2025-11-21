# intelligent_autocompleter/plugins/base.py
# small plugin base used across the project
from typing import Any, Dict, List, Tuple, Optional

Candidate = Tuple[str, float]
Bundle = Dict[str, Any]


class PluginBase:
    """
    Small plugin base. Subclass and override hooks needed.

    Common hooks:
      - on_init(registry) : called once when plugin loaded
      - on_train(lines) : bulk training hook
      - on_retrain(sentence) : incremental learning hook
      - on_suggest(fragment, cand, bundle) : modify/extend suggestions
      - on_accept(chosen, bundle) : user accepted suggestion
      - shutdown() : cleanup when host shuts down
    """
    name: str = "base"
    version: str = "0.0"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}

    # lifecycle
    def on_init(self, registry) -> None:
        return None

    def on_train(self, lines: List[str]) -> None:
        return None

    def on_retrain(self, sentence: str) -> None:
        return None

    # main pipeline hook
    def on_suggest(self, fragment: str, candidates: List[Candidate], bundle: Bundle) -> List[Candidate]:
        """
        Called with fragment and current candidates.
        Return a (possibly new) list of (word, score).
        Kept fast and side-effect-free where possible.
        """
        return candidates

    def on_accept(self, chosen: str, bundle: Bundle) -> None:
        return None

    def shutdown(self) -> None:
        return None

