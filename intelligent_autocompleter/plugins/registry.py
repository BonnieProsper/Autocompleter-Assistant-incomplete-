# intelligent_autocompleter/plugins/registry.py
"""
Plugin registry: holds plugin instances, runs hooks safely and deterministically.
"""

import traceback
from typing import Dict, List, Tuple, Any

from .base import PluginBase, Candidate, Bundle


class PluginRegistry:
    def __init__(self):
        # name -> plugin instance
        self._p: Dict[str, PluginBase] = {}
        # keep registration order
        self._order: List[str] = []

    def register(self, inst: PluginBase):
        name = getattr(inst, "name", inst.__class__.__name__)
        # brief guard: replace if exists
        if name in self._p:
            # keep existing order if already present
            self._p[name] = inst
            return
        self._p[name] = inst
        self._order.append(name)

    def unregister(self, name: str):
        self._p.pop(name, None)
        if name in self._order:
            self._order.remove(name)

    def get(self, name: str) -> PluginBase:
        return self._p.get(name)

    def all(self) -> List[PluginBase]:
        return [self._p[n] for n in self._order]

    def names(self) -> List[str]:
        return list(self._order)

    # lifecycle helpers (safe calls)
    def call_init(self):
        for p in self.all():
            try:
                p.on_init(self)
            except Exception:
                traceback.print_exc()

    def call_train(self, lines: List[str]):
        for p in self.all():
            try:
                p.on_train(lines)
            except Exception:
                traceback.print_exc()

    def call_retrain(self, sentence: str):
        for p in self.all():
            try:
                p.on_retrain(sentence)
            except Exception:
                traceback.print_exc()

    def call_accept(self, chosen: str, bundle: Bundle):
        for p in self.all():
            try:
                p.on_accept(chosen, bundle)
            except Exception:
                traceback.print_exc()

    # suggestion pipeline
    def run_suggest_pipeline(self, fragment: str, candidates: List[Candidate], bundle: Bundle = None) -> List[Candidate]:
        """
        Run on_suggest on each plugin in registration order; plugin may add or re-score candidates.
        Pipeline is resilient - plugin exceptions are caught and won't stop the flow.
        Final output is deduped and sorted by score desc.
        """
        bundle = bundle or {}
        out = list(candidates)
        for p in self.all():
            try:
                res = p.on_suggest(fragment, out, bundle)
                if isinstance(res, list):
                    out = res
            except Exception:
                traceback.print_exc()

        # dedupe (keep best score)
        best = {}
        for w, sc in out:
            cur = best.get(w)
            if cur is None or sc > cur:
                best[w] = sc
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        return ranked
