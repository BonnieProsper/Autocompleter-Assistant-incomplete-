# intelligent_autocompleter/plugins/registry.py
"""
Plugin registry that holds plugin instances, runs hooks safely and deterministically.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import importlib.util
import traceback
import os

from intelligent_autocompleter.plugins.base import PluginBase
from intelligent_autocompleter.utils.logger_utils import Log

Candidate = Tuple[str, float]


class PluginEntry:
    """Wrapper tracking plugin instance and enable/disable state."""

    def __init__(self, inst: PluginBase):
        self.inst = inst
        self.enabled = True

    def __repr__(self):
        return f"<PluginEntry {self.inst.name} enabled={self.enabled}>"


class PluginRegistry:
    """
    Central registry of plugins.
    Deterministic, safe, isolated plugin lifecycle & suggestion pipeline.
    """

    def __init__(self):
        self._plugins: Dict[str, PluginEntry] = {}

    # Registration -------------------------------------------------------
    def register(self, plugin: PluginBase):
        name = getattr(plugin, "name", plugin.__class__.__name__)
        self._plugins[name] = PluginEntry(plugin)
        Log.write(f"[PluginRegistry] registered: {name}")

    def unregister(self, name: str):
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
        entry = self._plugins.get(name)
        return entry.inst if entry else None

    def all(self) -> List[PluginEntry]:
        return list(self._plugins.values())

    # Lifecycle -------------------------------------------------------------------------
    def call_train(self, lines: List[str]):
        for name, entry in self._plugins.items():
            if not entry.enabled:
                continue
            try:
                entry.inst.on_train(lines)
            except Exception as e:
                Log.write(f"[Plugin:{name}] train hook error: {e}")

    def call_retrain(self, sentence: str):
        for name, entry in self._plugins.items():
            if not entry.enabled:
                continue
            try:
                entry.inst.on_retrain(sentence)
            except Exception as e:
                Log.write(f"[Plugin:{name}] retrain error: {e}")

    def call_accept(self, word: str, bundle: Optional[Dict[str, Any]] = None):
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
        bundle: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, str]]:

        bundle = bundle or {}

        # Normalize input suggestions
        cur: List[Candidate] = [(w, float(s)) for w, s in suggestions]

        # Each plugin may mutate the list
        for name, entry in self._plugins.items():
            if not entry.enabled:
                continue

            try:
                new = entry.inst.on_suggest(fragment, cur, bundle)
            except Exception as e:
                Log.write(f"[Plugin:{name}] on_suggest error: {e}")
                continue

            if new is None:
                continue

            normalized: List[Candidate] = []
            for item in new:
                if isinstance(item, tuple):
                    if len(item) >= 2:
                        normalized.append((item[0], float(item[1])))
                # ignore invalid

            if normalized:
                cur = normalized

        # Add "plugin" source tag required by HybridPredictor
        return [(w, s, "plugin") for (w, s) in cur]

    # Config -------------------------------------------------------------------------------
    def apply_config(self, cfg: Dict[str, Any]):
        for name, sub in cfg.items():
            entry = self._plugins.get(name)
            if not entry:
                Log.write(f"[PluginRegistry] config: unknown plugin {name}")
                continue

            entry.enabled = bool(sub.get("enabled", entry.enabled))
            try:
                entry.inst.configure(sub)
            except Exception:
                pass

    # Loader -------------------------------------------------------------------------------
    def discover_and_load(self, folder: str):
        """
        Load all .py modules in a folder and register any PluginBase subclasses.
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
                if spec is None or spec.loader is None:
                    continue
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
