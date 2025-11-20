# small registry to hold plugin instances in-memory

from typing import Dict, List
from .base import PluginBase

class PluginRegistry:
    def __init__(self):
        self._p: Dict[str, PluginBase] = {}

    def register(self, inst: PluginBase):
        key = getattr(inst, "name", inst.__class__.__name__)
        self._p[key] = inst

    def unregister(self, name: str):
        self._p.pop(name, None)

    def get(self, name: str):
        return self._p.get(name)

    def all(self) -> List[PluginBase]:
        return list(self._p.values())

    def names(self):
        return list(self._p.keys())
