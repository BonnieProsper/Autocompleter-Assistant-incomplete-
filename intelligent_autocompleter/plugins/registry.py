# intelligent_autocompleter/plugins/registry.py
"""
Plugin registry that holds plugin instances, runs hooks safely and deterministically. Includes:
- registration/unregistration
- safe lifecycle hooks (train, retrain, accept)
- deterministic suggest pipeline
- per-plugin enable/disable flags
- config injection (plugins/config.json is optional)
- optional source tagging for debugging
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import importlib.util
import traceback
import os

from intelligent_autocompleter.plugins.base import PluginBase
from intelligent_autocompleter.utils.logger_utils import Log

# plugin returns normally (word, score)
Candidate = Tuple[str, float]

class PluginEntry:
    """Tiny wrapper to track enable/disable state."""
    def __init__(self, inst: PluginBase):
        self.inst = inst
        self.enabled = True

    def __repr__(self):
        return f"<PluginEntry {self.inst.name} enabled={self.enabled}>"

class PluginRegistry:
    """
    Central registry for all plugins.
     - Order is insertion order, predictable but not perfect.
     - All plugin calls are wrapped in try/except so one plugin never
     breaks the entire suggestion pipeline.
     - Suggest pipeline lets each plugin mutate candidate list, but we
     normalize back to (word,score,src) so HybridPredictor can show
     where suggestions came from (optional).
    """
    def __init__(self):
        self._plugins: Dict[str, PluginEntry] = {}

    # Registration -------------------------------------------------------
    def register(self, plugin: PluginBase):
        """Register a plugin instance. Replace existing by name."""
        name = getattr(plugin, "name", plugin.__class__.__name__)
        self._plugins[name] = PluginEntry(plugin)
        Log.write(f"[PluginRegistry] registered: {name}")

    def unregister(self, name: str):
        """Unregister a plugin and attempt to call plugin.save()."""
        entry = self._plugins.pop(name, None)
        if not entry:
            return
        try:
            entry.inst.save()
        except Exception:
            pass
        Log.write(f"[PluginRegistry] unregistered: {name}")

    def names(self) -> List[str]:
        return list(self._plugins.keys())

    def get(self, name: str) -> Optional[PluginBase]:
        ent = self._plugins.get(name)
        return ent.inst if ent else None

    def all(self) -> List[PluginEntry]:
        return list(self._plugins.values())

    # Lifecycle -------------------------------------------------------------------------
    def call_train(self, lines: List[str]):
        """Initial corpus ingestion."""
        for name, entry in self._plugins.items():
            if not entry.enabled:
                continue
            try:
                entry.inst.on_train(lines)
            except Exception as e:
                Log.write(f"[Plugin:{name}] train hook error: {e}")

    def call_retrain(self, sentence: str):
        """Called when user completes a sentence."""
        for name, entry in self._plugins.items():
            if not entry.enabled:
                continue
            try:
                entry.inst.on_retrain(sentence)
            except Exception as e:
                Log.write(f"[Plugin:{name}] retrain error: {e}")

    def call_accept(self, word: str, bundle: Optional[Dict[str, Any]] = None):
        """Called when user accepts a suggestion."""
        for name, entry in self._plugins.items():
            if not entry.enabled:
                continue
            try:
                entry.inst.on_accept(word, bundle or {})
            except Exception as e:
                Log.write(f"[Plugin:{name}] accept error: {e}")

    # Suggest Pipeline --------------------------------------------------------------------
    def run_suggest_pipeline(
        self,
        fragment: str,
        suggestions: List[Candidate],
        bundle: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, str]]:
        """
        Run through all plugins in order.
        Each plugin: on_suggest(fragment, suggestions, bundle)
        Plugins may return:
            - None (no change)
            - list[(w,score)]
            - list[(w,score,src)]
        Always normalize to list[(w,score,src)] for HybridPredictor.
        """
        bundle = bundle or {}

        # current pipeline state (normalized to (w,score))
        cur: List[Candidate] = [(w, float(s)) for w, s in suggestions]

        for name, entry in self._plugins.items():
            if not entry.enabled:
                continue
            try:
                new = entry.inst.on_suggest(fragment, cur, bundle)
            except Exception as e:
                Log.write(f"[Plugin:{name}] on_suggest error: {e}")
                continue  # keep pipeline alive
            if new is None:
                # plugin didn't modify anything
                continue

            # Normalize returned items
            normalized: List[Tuple[str, float]] = []
            for item in new:
                if isinstance(item, tuple):
                    if len(item) == 2:
                        normalized.append((item[0], float(item[1])))
                    elif len(item) == 3:
                        normalized.append((item[0], float(item[1])))
                # ignore invalid items

            cur = normalized or cur

        # Add plugin source tag needed by HybridPredictor
        return [(w, s, "plugin") for (w, s) in cur]

    # Config ----------------------------------------------------------
    def apply_config(self, cfg: Dict[str, Any]):
        """
        cfg example:
        {
            "EmojiPlugin": {"enabled": true, "some_key": 5},
            "CodeAssist": {"enabled": false}
        }
        """
        for name, sub in cfg.items():
            entry = self._plugins.get(name)
            if not entry:
                Log.write(f"[PluginRegistry] config: unknown plugin {name}")
                continue

            entry.enabled = bool(sub.get("enabled", entry.enabled))
            # pass plugin-specific config if plugin supports configure()
            try:
                entry.inst.configure(sub)
            except Exception:
                pass

    # Basic Loader ----------------------------------------------------------------
    def discover_and_load(self, folder: str):
        """
        Load all .py modules in a folder and register any PluginBase subclasses.
        This is a convenience loader for small projects.
        """
        if not os.path.isdir(folder):
            return
         
        for fname in os.listdir(folder):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            modname = fname[:-3]
            fpath = os.path.join(folder, fname)
            try:
                spec = importlib.util.spec_from_file_location(modname, fpath)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                for attr in dir(mod):
                    obj = getattr(mod, attr)
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, PluginBase)
                        and obj is not PluginBase
                    ):
                        inst = obj()
                        self.register(inst)

            except Exception as e:
                tb = traceback.format_exc(limit=1)
                Log.write(f"[PluginRegistry] load error {fname}: {e}\n{tb}")
            if cur is None or sc > cur:
                best[w] = sc
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        return ranked
