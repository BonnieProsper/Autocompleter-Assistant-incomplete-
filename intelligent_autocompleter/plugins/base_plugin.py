"""
PluginBase
Minimal base class that all autocompleter plugins extend.
 - Small API surface
 - All hooks are optional (plugins override only what they need)
 - Side effect free where possible (except train/retrain)

Plugins typically use:
    on_train(lines)
    on_retrain(sentence)
    on_suggest(fragment, suggestions, bundle)
    on_accept(word, bundle)

Optionally:
    configure(cfg)
    save()/load()
    shutdown()

Return conventions for on_suggest():
    - None = no change
    - [(word, score)] = updated suggestion list
    - [(word, score, src)] = same but with source annotation
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

# Core types used in plugin system
Candidate = Tuple[str, float]                # (word, score)
Candidate3 = Tuple[str, float, str]           # (word, score, source label)
Bundle = Dict[str, Any]                    # arbitrary metadata passed through pipeline

class PluginBase:
    """
    Base class for all plugins.
    Subclass and override whatever hooks you need.
    Attributes:
        name: short identifier for plugin ("SpellFix", "Emoji", etc.)
        version: plugin version for debugging or compatibility checks
        cfg: runtime configuration passed from config.json or registry.apply_config()
    """
    name: str = "plugin_base"
    version: str = "0.0"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        # storing cfg as dict so plugins can mutate freely
        self.cfg: Dict[str, Any] = cfg or {}

    # Lifecycle hooks (optional) -----------------------------------------------
    def on_init(self, registry) -> None:
        """Called once the plugin is registered. Registry object is passed in case plugin wants to query others."""
        return None

    def on_train(self, lines: List[str]) -> None:
        """Bulk training hook. Usually builds internal frequency tables, indexes, etc."""
        return None

    def on_retrain(self, sentence: str) -> None:
        """Incremental update hook during interactive use."""
        return None

    def on_accept(self, chosen: str, bundle: Bundle) -> None:
        """Called when user accepts a suggestion."""
        return None

    # Main pipeline hook ----------------------------------------------
    def on_suggest(
        self,
        fragment: str,
        candidates: List[Candidate],
        bundle: Bundle
    ) -> Optional[List]:
        """
        Modify or extend the candidate list.
        Args:
            fragment: the user's current token (e.g. "hel")
            candidates: list[(word, score)] from previous plugins or model
            bundle: dict containing context (cursor pos, metadata, etc.)
        Returns:
            None: keep candidates unchanged
            list[(word, score)]: fully replaced or modified candidate list
            list[(word, score, source)]: same as above, with plugin-specific source label
        Notes:
            - Keep function fast bc it relies on every keystroke.
            - Avoid side effects unless necessary.
        """
        return None

    # Config + persistence (optional) -------------------------------------------------
    def configure(self, cfg: Dict[str, Any]):
        """Called when registry.apply_config() passes new runtime options."""
        self.cfg.update(cfg)

    def save(self) -> None:
        """Persist plugin-specific state (optional). Overridden by plugins that need it."""
        return None

    def load(self) -> None:
        """Restore plugin-specific state (optional)."""
        return None

    # Shutdown (optional) -----------------------------------------------------------
    def shutdown(self) -> None:
        """Called when the host application is shutting down."""
        return None

