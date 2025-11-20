# intelligent_autocompleter/plugins/base.py
# small plugin base class

from typing import Any, Dict, Optional

class PluginBase:
    """
    Simple plugin base.
    - name: unique id for plugin
    - cfg: plugin config dict
    Methods:
    - on_input(bundle) to run when user submits input (can mutate bundle)
    - on_suggest(bundle, suggestions) can re-rank/add suggestions
    - save()/load() optional persistence hooks
    """
    name = "base"
    version = "0.0"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}

    def on_input(self, bundle: Dict[str, Any]) -> None:
        """Called when user submits text (train/retrain)"""
        # bundle keys: text, tokens, context
        return None

    def on_suggest(self, bundle: Dict[str, Any], suggestions: list) -> list:
        """
        Called when suggestions are produced.
        Should return a possibly modified list of (word, score).
        """
        return suggestions

    def save(self):
        """Optional, persists internal state"""
        return None

    def load(self):
        """Optional, restores state"""
        return None
